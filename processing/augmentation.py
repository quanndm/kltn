from monai.transforms import (
    RandRotate90d,
    Compose,
    RandFlipd,
    RandZoomd,
    RandShiftIntensityd,
    RandAffined,
    RandAdjustContrastd,
    RandScaleIntensityd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandBiasFieldd,
    EnsureChannelFirstd
)

def train_augmentations():
    """
    Returns a list of augmentations to apply to the training data
    Augmentations:
        # RandRotate90d: Randomly rotate the image by 90 degrees
        RandFlipd: Randomly flip the image along the specified axis
        RandZoomd: Randomly zoom the image by a specified factor
        RandShiftIntensityd: Randomly shift the intensity of the image
    """
    return Compose([
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandZoomd(keys=["image", "label"], prob=0.4, min_zoom=0.9, max_zoom=1.1, mode=["trilinear", "nearest"]),  
        RandAffined(
            keys=["image", "label"], 
            prob=0.3,
            rotate_range=(0.05, 0.05, 0.05),  
            scale_range=(0.05, 0.05, 0.05),
            mode=["trilinear", "nearest"]
        ),
        RandShiftIntensityd(keys=["image"], prob=0.3, offsets=0.1),

        RandGaussianNoised(keys=["image"], prob=0.15, mean=0.0, std=0.01),
        RandBiasFieldd(keys=["image"], prob=0.15),
    ])
    
def stage2_train_augmentation():
    """
    Returns a list of augmentations to apply to the training data for stage 2
    Augmentations:
        RandFlipd: Randomly flip the image along the specified axis
        RandAffined: Randomly affine transform the image
        RandAdjustContrastd: Randomly adjust the contrast of the image
        RandScaleIntensityd: Randomly scale the intensity of the image
        RandShiftIntensityd: Randomly shift the intensity of the image
        RandGaussianNoised: Randomly add Gaussian noise to the image
    """
    return Compose([
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandZoomd(keys=["image", "label"], prob=0.3, min_zoom=0.9, max_zoom=1.1, mode=["trilinear", "nearest"]),  
        RandShiftIntensityd(keys=["image"], prob=0.3, offsets=0.1),

        RandGaussianNoised(keys=["image"], prob=0.1, mean=0.0, std=0.01),
        RandGaussianSmoothd(keys=["image"], sigma_x=(0.25, 1.0), prob=0.05)
    ])

def stage2_train_augmentation_2d():
    """
    image: [1,C, H,W]
    label: [1, 1, H, W]
    Returns a list of augmentations to apply to the training data for stage 2
    Augmentations:
        RandFlipd: Randomly flip the image along the specified axis
        RandAffined: Randomly affine transform the image
        RandAdjustContrastd: Randomly adjust the contrast of the image
        RandScaleIntensityd: Randomly scale the intensity of the image
        RandShiftIntensityd: Randomly shift the intensity of the image
        RandGaussianNoised: Randomly add Gaussian noise to the image
    """
    return Compose([
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=[2, 3]),
        RandZoomd(keys=["image", "label"], prob=0.3, min_zoom=0.9, max_zoom=1.1, mode=["bilinear", "nearest"], spatial_dims=[2, 3]),  
        RandAffined(
            keys=["image", "label"], 
            prob=0.3,
            rotate_range=0.05, 
            mode=["bilinear", "nearest"],
        ),
        RandShiftIntensityd(keys=["image"], prob=0.3, offsets=0.1),

        RandGaussianNoised(keys=["image"], prob=0.1, mean=0.0, std=0.01),
        RandGaussianSmoothd(keys=["image"], sigma_x=(0.25, 1.0), prob=0.05)
    ])