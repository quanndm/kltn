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
    """
    return Compose([
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandZoomd(keys=["image", "label"], prob=0.4, min_zoom=0.8, max_zoom=1.2, mode=["trilinear", "nearest"]),  
        RandAffined(
            keys=["image", "label"], 
            prob=0.3,
            rotate_range=(0.05, 0.05, 0.05),  
            scale_range=(0.05, 0.05, 0.05),
            mode=["trilinear", "nearest"],
            padding_mode="border"
        ),
        RandShiftIntensityd(keys=["image"], prob=0.3, offsets=0.1),

        RandGaussianNoised(keys=["image"], prob=0.15, mean=0.0, std=0.01),
        RandBiasFieldd(keys=["image"], prob=0.15),
    ])

def stage2_train_augmentation_2d():
    """
    Returns a list of augmentations to apply to the training data for stage 2
    Augmentations:
    """
    return Compose([
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=[1, 2]),
        RandZoomd(keys=["image", "label"], prob=0.35, min_zoom=0.85, max_zoom=1.15, mode=["bilinear", "nearest"], spatial_dims=[1, 2]),  
        RandAffined(
            keys=["image", "label"], 
            prob=0.3,
            rotate_range=0.075, 
            mode=["bilinear", "nearest"],
            padding_mode="border"  
        ),
        RandShiftIntensityd(keys=["image"], prob=0.3, offsets=0.1),

        RandGaussianNoised(keys=["image"], prob=0.1, mean=0.0, std=0.01),
        RandGaussianSmoothd(keys=["image"], sigma_x=(0.25, 1.0), prob=0.05)
    ])