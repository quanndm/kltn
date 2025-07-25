import SimpleITK as sitk
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from ..processing.preprocessing import (
    zscore_normalise,
    resize_image,
    truncate_HU,
    get_liver_roi,
    normalize
)
from ..processing.augmentation import train_augmentations, stage2_train_augmentation_2d
import os
import torch.nn.functional as F

class Lits(Dataset):
    def __init__(self, patient_dirs, benchmarking = False, training=True, normalizations="zscores", transformations=False, mode="liver"):
        '''
        Args:
            patient_dirs: list of dict, each dict contains id and the paths to the patient's images/ segmentations
            training: bool, whether the dataset is for training or testing
            benchmarking: bool, whether the dataset is for benchmarking
            normalizations: str, the type of normalization to apply to the images, either "zscores" or "minmax"
            transformations: bool, whether to apply transformations to the images
            mode: all | liver
        '''
        self.training = training
        self.benchmarking = benchmarking
        self.normalizations = normalizations
        self.patient_dirs = patient_dirs
        self.transformations = transformations
        self.mode = mode

    def __len__(self):
        return len(self.patient_dirs)

    def __getitem__(self, idx):
        _patient = self.patient_dirs[idx]
        _image = self.load_nii(_patient["volume"])
        _seg = self.load_nii(_patient["segmentation"])
        root_size = _image.shape
        image, seg = self.preprocessing(_image, _seg, self.training, self.normalizations)

        if self.mode == "liver":
            seg = (seg > 0).astype(np.uint8)

            _seg = np.expand_dims(_seg, axis=0)
            _seg = (_seg > 0).astype(np.uint8)    
        _seg = torch.from_numpy(_seg)


        if self.training and self.transformations:
            image, seg = self.augmentation(image, seg)

        image, seg = image.astype(np.float32), seg.astype(np.uint8)
        image, seg = torch.from_numpy(image), torch.from_numpy(seg)

        return dict(
            idx=idx,
            patient_id=_patient["id"],
            image=image,
            label=seg,
            root_size=root_size,
        )

    @staticmethod
    def load_nii(path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} not found!")
        return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))


    @staticmethod
    def preprocessing(image, seg, training, normalizations):
        '''
        Args:
            image: np.ndarray, the image to preprocess
            seg: np.ndarray, the segmentation to preprocess
            training: bool, whether the dataset is for training or testing
            normalizations: str, the type of normalization to apply to the images, either "zscores" or "minmax"
        Returns:
            image: np.ndarray, the preprocessed image
            seg: np.ndarray, the preprocessed segmentation
        '''
        # get liver ROI
        # image, seg, bbox = get_liver_roi(image, seg)

        # clip HU values
        image = truncate_HU(image)

        # normalizations
        if normalizations == "zscores":
            image = zscore_normalise(image)
        else:
            image = normalize(image)

        # expand dims of image and segmentation - resize image
        image, seg = resize_image(np.expand_dims(image, axis=0), np.expand_dims(seg, axis=0), target_size=(128, 128, 128))  

        return image, seg

    @staticmethod
    def augmentation(image, seg):
        '''
        Args:
            image: np.ndarray, the image to augment
            seg: np.ndarray, the segmentation to augment
        Returns:
            image: np.ndarray, the augmented image
            seg: np.ndarray, the augmented segmentation
        '''
        train_transforms = train_augmentations() 
        data_dict = {"image": image, "label": seg}
        augmented = train_transforms(data_dict)
        
        return augmented["image"], augmented["label"]


class Stage2Dataset2D(Dataset):
    def __init__(self, patient_dirs, training=True, normalizations="zscores", transformations=False):
        '''
        Args:
            patient_dirs: list of dict, each dict contains id and the paths to the patient's images/ segmentations
            training: bool, whether the dataset is for training or testing
            normalizations: str, the type of normalization to apply to the images, either "zscores" or "minmax"
            transformations: bool, whether to apply transformations to the images
            liver_mask: liver mask predict, shape (1, D, H, W)
        '''
        self.training = training
        self.normalizations = normalizations
        self.patient_dirs = patient_dirs
        self.transformations = transformations
    def __len__(self):
        return len(self.patient_dirs)

    def __getitem__(self, idx):
        _patient = self.patient_dirs[idx]
        data = self.load_npz(_patient["file"])
        image = data["image"]
        seg = data["seg"]
        bbox = data["bbox"]

        # preprocessing
        image, seg = self.preprocessing(image, seg, self.training, self.normalizations)


        # augmentation
        if self.training and self.transformations:
            image, seg = np.expand_dims(image, axis=0), np.expand_dims(seg, axis=0)
            image, seg = self.augmentation(image, seg)
            image, seg = image.cpu().numpy().squeeze(0), seg.squeeze(0)

        liver_mask = (seg == 1).astype(np.uint8)
        image, seg = image.astype(np.float32), (seg == 2).astype(np.uint8)

        liver_mask = np.max(liver_mask, axis=0)  # (3, H, W) -> (H, W)
        seg = np.max(seg, axis=0)  # (3, H, W) -> (H, W)
        # # convert to torch tensors
        liver_mask, seg = np.expand_dims(liver_mask, axis=0), np.expand_dims(seg, axis=0)
        liverr_mask = torch.from_numpy(liver_mask)
        image, seg = torch.from_numpy(image), torch.from_numpy(seg)

        return dict(
            idx=idx,
            # source=_patient["source"] if "source" in _patient else None,
            patient_id=_patient["id"],
            slide=_patient["slide"],
            image=image,
            label=seg,
            liver_mask=liver_mask,
            bbox=bbox,
        )

    @staticmethod
    def load_npz(path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} not found!")
        return np.load(path, allow_pickle=True)

    @staticmethod
    def preprocessing(image, seg, training, normalizations):
        '''
        Args:
            image: np.ndarray, the image to preprocess
            seg: np.ndarray, the segmentation to preprocess
            training: bool, whether the dataset is for training or testing
            normalizations: str, the type of normalization to apply to the images, either "zscores" or "minmax"
        Returns:
            image: np.ndarray, the preprocessed image
            seg: np.ndarray, the preprocessed segmentation
        '''
        # clip HU values
        image = truncate_HU(image, 0, 200)

        # normalizations
        if normalizations == "zscores":
            image = zscore_normalise(image)
        else:
            image = normalize(image)

        # resize image
        image, seg = resize_image(image, seg, mode=None, target_size=(256, 256), target_size_seg=(256, 256))
        return image, seg

    @staticmethod
    def augmentation(image, seg):
        '''
        Args:
            image: np.ndarray, the image to augment
            seg: np.ndarray, the segmentation to augment
        Returns:
            image: np.ndarray, the augmented image
            seg: np.ndarray, the augmented segmentation
        
        '''
        train_transforms = stage2_train_augmentation_2d()
        data_dict = {"image": image, "label": seg}
        augmented = train_transforms(data_dict)
        return augmented["image"], augmented["label"]