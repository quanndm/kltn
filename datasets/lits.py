import SimpleITK as sitk
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from ..processing.preprocessing import pad_or_crop_image, zscore_normalise, irm_min_max_preprocess, resize_image, truncate_HU
from ..processing.augmentation import train_augmentations

class Lits(Dataset):
    def __init__(self, patient_dirs, benchmarking = False, training=True, normalizations="zscores", transformations=False):
        '''
        Args:
            patient_dirs: list of dict, each dict contains id and the paths to the patient's images/ segmentations
            training: bool, whether the dataset is for training or testing
            benchmarking: bool, whether the dataset is for benchmarking
            normalizations: str, the type of normalization to apply to the images, either "zscores" or "minmax"
            transformations: bool, whether to apply transformations to the images
        '''
        self.training = training
        self.benchmarking = benchmarking
        self.normalizations = normalizations
        self.patient_dirs = patient_dirs
        self.transformations = transformations

    def __len__(self):
        return len(self.patient_dirs)

    def __getitem__(self, idx):
        _patient = self.patient_dirs[idx]
        image = self.load_nii(_patient["volume"])
        seg = self.load_nii(_patient["segmentation"])

        image, seg, _ = self.preprocessing(image, seg, self.training, self.normalizations)

        if self.training and self.transformations:
            image, seg = self.augmentation(image, seg)

        image, seg = image.astype("float16"), seg.astype("bool")
        image, seg = torch.from_numpy(image), torch.from_numpy(seg)

        return dict(
            idx=idx,
            patient_id=_patient["id"],
            image=image,
            label=seg,
            supervised=True,
            # crop_indexes = (bounds["z"], bounds["y"], bounds["x"])
        )

    @staticmethod
    def load_nii(path):
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
            bounds: dict of tuples, includes min-max of x, y, z
        '''
        # truncate HU values
        image = truncate_HU(image)

        # normalizations
        if normalizations == "zscores":
            image = zscore_normalise(image)
        else:
            image = irm_min_max_preprocess(image)

        # split labels + expand dims of image
        liver_mask = seg > 0
        tumor_mask = seg == 2   

        seg = np.stack([liver_mask, tumor_mask], axis=0)
        image = np.expand_dims(image, axis=0)

        # crop - padding - resize
        # if training:
        #     z_indexes, y_indexes, x_indexes = np.nonzero(np.sum(image, axis=0) != 0)
        #     zmin, ymin, xmin = [max(0, int(np.min(arr) - 1)) for arr in (z_indexes, y_indexes, x_indexes)]
        #     zmax, ymax, xmax = [int(np.max(arr) + 1) for arr in (z_indexes, y_indexes, x_indexes)]

        #     image = image[:, zmin:zmax, ymin:ymax, xmin:xmax]
        #     seg = seg[:, zmin:zmax, ymin:ymax, xmin:xmax]

        #     image, seg = pad_or_crop_image(image, seg, target_size=(128, 128, 128))

        # else:
        #     z_indexes, y_indexes, x_indexes = np.nonzero(np.sum(image, axis=0) != 0)
        #     zmin, ymin, xmin = [max(0, int(np.min(arr) - 1)) for arr in (z_indexes, y_indexes, x_indexes)]
        #     zmax, ymax, xmax = [int(np.max(arr) + 1) for arr in (z_indexes, y_indexes, x_indexes)]

        #     image = image[:, zmin:zmax, ymin:ymax, xmin:xmax]
        #     seg = seg[:, zmin:zmax, ymin:ymax, xmin:xmax]
        image, seg = resize_image(image, seg, target_size=(128, 128, 128))  


        # bounds = {
        #     "x": (xmin, xmax),
        #     "y": (ymin, ymax),
        #     "z": (zmin, zmax)
        # }
        return image, seg, None

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