from monai.transforms import  RandRotate90d, Compose, RandFlipd, RandZoomd, RandShiftIntensityd

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
        RandZoomd(keys=["image", "label"], prob=0.4, min_zoom=0.9, max_zoom=1.1, mode=["trilinear", "nearest"]),  
        RandShiftIntensityd(keys=["image"], prob=0.4, offsets=0.1),
    ])
    