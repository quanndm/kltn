from monai.transforms import ScaleIntensityRanged, RandRotate90d, Compose, EnsureChannelFirstd

def train_augmentations():
    """
    Returns a list of augmentations to apply to the training data
    Augmentations:
        EnsureChannelFirstd: Ensure the channel is the first dimension of the image
        RandRotate90d: Randomly rotate the image by 90 degrees
        ScaleIntensityRanged: Scale the intensity of the image to a given range, in this case from -200 to 200 - like truncate HU from -200 to 200
    """
    return Compose([
        EnsureChannelFirstd(keys=["image", "label"]),
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=1),
        ScaleIntensityRanged(keys=["image"], a_min=-200, a_max=200, b_min=0.0, b_max=1.0, clip=True),
    ])
    