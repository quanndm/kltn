from monai.transforms import  RandRotate90d, Compose

def train_augmentations():
    """
    Returns a list of augmentations to apply to the training data
    Augmentations:
        RandRotate90d: Randomly rotate the image by 90 degrees
    """
    return Compose([
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=1),
    ])
    