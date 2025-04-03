import torch
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    EnsureType
)
import cv2
import numpy as np


post_trans = Compose(
    [EnsureType(), Activations(softmax=True),AsDiscrete(argmax=True, to_onehot=3)]
)
post_trans_v2 = Compose(
    [EnsureType(), Activations(softmax=True),AsDiscrete(argmax=True)]
)

post_softmax = Activations(softmax=True)

post_pred = AsDiscrete(argmax=True)

#TODO: test this function
def apply_cca(segmentation, min_area = 500):
    """
    Apply connected component analysis to the segmentation mask.
    Args:
        segmentation (numpy.ndarray): The segmentation mask.
        min_area (int): Minimum area of the connected components to keep.
    Returns:
        numpy.ndarray: The processed segmentation mask with small components removed.
    """
    # Find connected components
    batch_size, num_classes, depth, height, width = segmentation_tensor.shape

    # Initialize an empty mask for the processed segmentation
    processed_segmentation = np.zeros_like(segmentation)

    # Loop through each connected component
    for i in range(batch_size):
        for c in range(num_classes):
            for d in range(depth):
                segmentation_slice = segmentation[i, c, d, :, :]

                segmentation_slice = segmentation_slice.astype(np.uint8)
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(segmentation, connectivity=8)

                filtered_segmentation = np.zeros_like(segmentation_slice)
                for j in range(1, num_labels):
                    if stats[j, cv2.CC_STAT_AREA] >= min_area:
                        filtered_segmentation[labels == j] = 1
                
                processed_segmentation[i, c, d, :, :] = filtered_segmentation


    processed_segmentation = torch.tensor(processed_segmentation, dtype=torch.float32)
    return processed_segmentation

def apply_crf():
    pass