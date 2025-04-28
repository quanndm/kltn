import SimpleITK as sitk
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from ..utils.utils import inference
from ..processing.preprocessing import (
    pad_or_crop_image,
    zscore_normalise,
    irm_min_max_preprocess,
    resize_image,
    truncate_HU,
    get_liver_roi,
    extract_liver_mask_binary,
    mask_input_with_liver,
    crop_patch_around_tumor,
    normalize
)
from ..processing.augmentation import train_augmentations, stage2_train_augmentation
import os

class Lits(Dataset):
    def __init__(self, patient_dirs, benchmarking = False, training=True, normalizations="zscores", transformations=False, mode="all"):
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
            supervised=True,
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
            image = irm_min_max_preprocess(image)

        # expand dims of image and segmentation
        image = np.expand_dims(image, axis=0)
        seg = np.expand_dims(seg, axis=0)
        
        # resize image
        image, seg = resize_image(image, seg, target_size=(128, 128, 128))  

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

class Stage2Dataset(Dataset):
    def __init__(self, patient_dirs, training=True, normalizations="zscores", transformations=False, liver_masks=None):
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
        self.liver_masks = liver_masks

    def __len__(self):
        return len(self.patient_dirs)

    def __getitem__(self, idx):
        _patient = self.patient_dirs[idx]
        image = self.load_nii(_patient["volume"])
        seg = self.load_nii(_patient["segmentation"])
        root_size = image.shape
        liver_mask = self.liver_masks[idx] if self.liver_masks is not None else None

        image, seg, bbox, liver_mask = self.preprocessing(image, seg, self.training, self.normalizations, liver_mask=liver_mask) # shape: (1, 128, 128, 128)

        if self.training and seg.sum() == 0:
            return self.__getitem__((idx + 1) % self.__len__())

        image, seg = image_mask.astype(np.float32), seg.astype(np.uint8)

        if self.training and self.transformations:
            image, seg = self.augmentation(image, seg)

        # convert to torch tensors
        if self.training:
            image, seg = torch.from_numpy(image.detach().cpu().numpy()), torch.from_numpy(seg.detach().cpu().numpy())
        else:
            image, seg = torch.from_numpy(image), torch.from_numpy(seg)

        return dict(
            idx=idx,
            patient_id=_patient["id"],
            image=image,
            label=seg,
            liver_mask = self.liver_mask,
            supervised=True,
            root_size=root_size,
            bbox = bbox
        )

    @staticmethod
    def load_nii(path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} not found!")
        return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
    
    @staticmethod
    def preprocessing(image, seg, training, normalizations, liver_mask=None):
        '''
        Args:
            image: np.ndarray, the image to preprocess
            seg: np.ndarray, the segmentation to preprocess
            training: bool, whether the dataset is for training or testing
            normalizations: str, the type of normalization to apply to the images, either "zscores" or "minmax"
            liver_mask: np.ndarray, the liver mask predicted from the model stage 1, shape (1, D, H, W)
        Returns:
            image: np.ndarray, the preprocessed image
            seg: np.ndarray, the preprocessed segmentation
        '''           

        # expand dims of image and segmentation and resize image
        image, seg = np.expand_dims(image, axis=0), np.expand_dims(seg, axis=0)
        image, seg = resize_image(image, seg, target_size=(128, 128, 128))  

        # get liver ROI
        liver_mask = np.squeeze(liver_mask, axis=0)
        image, seg = np.squeeze(image, axis=0), np.squeeze(seg, axis=0)
        image, seg, bbox = get_liver_roi(image, liver_mask, margin=15)

        # clip HU values
        image = truncate_HU(image)

        # normalizations
        if normalizations == "zscores":
            image = zscore_normalise(image)
        else:
            image = normalize(image)

        # get tumor mask
        seg = (seg == 2).astype(np.uint8)
    

        # expand dims of image and segmentation and resize image
        image, seg = np.expand_dims(image, axis=0), np.expand_dims(seg, axis=0)
        liver_mask = np.expand_dims(liver_mask, axis=0)

        return image, seg, bbox, liver_mask
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