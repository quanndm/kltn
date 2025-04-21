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
        # RandRotate90d(keys=["image", "label"], prob=0.5, max_k=1),
        RandFlipd(keys=["image", "label"], prob=0.4, spatial_axis=2),
        RandZoomd(keys=["image", "label"], prob=0.3, min_zoom=0.9, max_zoom=1.1, mode=["trilinear", "nearest"]),  
        RandShiftIntensityd(keys=["image"], prob=0.3, offsets=0.1),
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
    
        RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.3),
        RandFlipd(keys=["image", "label"], spatial_axis=1, prob=0.3),

        RandAffined(
            keys=["image", "label"],
            rotate_range=(0.05, 0.05, 0.05),   # ±5 degrees
            scale_range=(0.05, 0.05, 0.05),      # ±5%
            # translate_range=(2, 2, 2),       # translate 5 voxel
            mode=("bilinear", "nearest"),
            padding_mode="border",
            prob=0.1
        ),

        RandAdjustContrastd(keys=["image"], prob=0.15, gamma=(0.6, 1.2)),
        RandScaleIntensityd(keys=["image"], prob=0.15, factors=0.1),   # ±10%
        RandShiftIntensityd(keys=["image"], prob=0.15, offsets=0.05),  # ±5%

        RandGaussianNoised(keys=["image"], prob=0.15, mean=0.0, std=0.01),
    ])