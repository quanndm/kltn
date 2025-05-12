import SimpleITK as sitk
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from ..processing.preprocessing import (
    zscore_normalise,
    resize_image,
    truncate_HU,
    normalize
)
from ..processing.augmentation import train_augmentations
import os


class CustomDatasetLiver(Dataset):
    """
    Custom dataset class for loading and processing images and labels.
    """
    def __init__(self,patient_dirs,  training=True, normalizations="zscores", transformations=True):
        """
        Args:
            image_paths (list): List of paths to the images.
            label_paths (list): List of paths to the labels. Default is None.
            training (bool): Whether the dataset is for training or testing. Default is True.
            normalizations (str): Normalization method to be applied. Default is "zscores".
            transformations (bool): Whether to apply transformations. Default is True.
            mode (str): Mode of the dataset. Default is "all".
            liver_masks_bbox (list): List of liver masks bounding boxes. Default is None.
        """
        self.training = training
        self.normalizations = normalizations
        self.transformations = transformations
        self.patient_dirs = patient_dirs
        

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Get the item at the specified index.
        Args:
            idx (int): Index of the item to be fetched.
        Returns:
            dict: Dictionary containing the image and label.
        """
        # Load the image and label
        _patient = self.patient_dirs[idx]
        _image = self.load_nii(_patient["volume"])
        _label = self.load_nii(_patient["segmentation"])
        _source = _patient["source"]
        root_size = _image.shape
        
        # preprocess the image
        image, seg = self.preprocess_image(_image, _label, self.normalizations)

        # convert mask to liver mask
        seg = (seg > 0).astype(np.uint8)  # Convert to binary mask

        # aug
        if self.training and self.transformations:
            image, seg = self.augment_image(image, seg)


        image, seg = image.astype(np.float32), seg.astype(np.uint8)
        image, seg = torch.from_numpy(image), torch.from_numpy(seg)
        image, seg = image.unsqueeze(0), seg.unsqueeze(0)  # Add channel dimension
        
        return dict(
            image=image,
            label=seg,
            source=_source,
            patient_id=_patient["id"],
            root_size=root_size
        )


    @staticmethod
    def load_nii(path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} not found!")
        return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))

    @staticmethod
    def preprocess_image(image, normalizations="zscores"):
        """
        Preprocess the image based on the specified normalization method.
        Args:
            image (numpy.ndarray): The input image to be preprocessed.
            normalizations (str): Normalization method to be applied. Default is "zscores".
        Returns:
            numpy.ndarray: The preprocessed image.
        """
        # clip HU
        image = truncate_HU(image, -200, 250)

        # normalize
        if normalizations == "zscores":
            image = zscore_normalise(image)
        elif normalizations == "minmax":
            image = normalize(image)

        # resize
        image, seg = resize_image(np.expand_dims(image, axis=0), np.expand_dims(seg, axis=0), target_size=(128, 128, 128))  

        return image, seg

    @staticmethod   
    def augment_image(image, seg):
        """
        Apply augmentations to the image and segmentation.
        Args:
            image (numpy.ndarray): The input image to be augmented.
            seg (numpy.ndarray): The input segmentation to be augmented.
            training (bool): Whether the dataset is for training or testing. Default is True.
        Returns:
            tuple: Tuple containing the augmented image and segmentation.
        """
        train_transforms = train_augmentations() 
        data_dict = {"image": image, "label": seg}
        augmented = train_transforms(data_dict)
        
        return augmented["image"], augmented["label"]