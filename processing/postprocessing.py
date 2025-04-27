import torch
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    EnsureType
)
import cv2
import numpy as np
import scipy.ndimage as ndi
from skimage import morphology, measure
from scipy.ndimage import label

post_trans = Compose(
    [EnsureType(), Activations(softmax=True),AsDiscrete(argmax=True, to_onehot=3)]
)
post_trans_v2 = Compose(
    [EnsureType(), Activations(softmax=True),AsDiscrete(argmax=True)]
)

post_trans_label = AsDiscrete(to_onehot=3)

post_softmax = Activations(softmax=True)

post_pred = AsDiscrete(argmax=True)

post_trans_stage1 = post_trans_stage2 = Compose(
    [EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold=0.5)]
)

def post_processing_stage2(logits, threshold=0.5, device=None):
    """
    Post-processing for stage 2 of the model.
    Args:
        logits (torch.Tensor): The logits output from the model, shape (1, 1, D, H, W).
        threshold (float): The threshold for binarizing the logits.
    Returns:
        torch.Tensor: The post-processed logits.
    """
    # Apply sigmoid activation function
    probs = torch.sigmoid(logits)

    # Convert to numpy array and apply threshold
    tumor_mask = probs.cpu().numpy()
    tumor_mask =(tumor_mask > threshold).astype(np.uint8)
    tumor_mask = np.squeeze(tumor_mask) # shape(1, 1, D, H, W) -> (D, H, W)

    # Morphological closing and opening
    tumor_mask = morphology.binary_closing(tumor_mask, morphology.ball(1))
    tumor_mask = morphology.binary_opening(tumor_mask, morphology.ball(1))

    # fill holes
    tumor_mask = ndi.binary_fill_holes(tumor_mask)

    tumor_mask = np.expand_dims(tumor_mask, axis=0) # shape(D, H, W) -> (1, D, H, W)
    tumor_mask = torch.from_numpy(tumor_mask).unsqueeze(0).to(device) # shape(1, D, H, W) -> (1, 1, D, H, W)

    return tumor_mask

def keep_largest_connected_component(predicted_mask):
    """
        predicted_mask: tensor, shape ( 1, D, H, W)
        Returns:
            largest_component: tensor, shape (1, 1, D, H, W)
    """
    predicted_mask = predicted_mask.squeeze().detach().cpu().numpy()  # shape (1, D, H, W) -> (D, H, W)
    labeled_mask, num_features = label(predicted_mask)

    if num_features == 0:
        return predicted_mask
    
    sizes = np.bincount(labeled_mask.ravel())
    sizes[0] = 0  # Ignore the background label

    largest_label = np.argmax(sizes)
    largest_component = np.where(labeled_mask == largest_label, 1, 0)

    return torch.from_numpy(largest_component).unsqueeze(0).unsqueeze(0)  # shape (1, 1, D, H, W)


def smooth_mask(mask, kernel_size=3):
    """
    mask: numpy array (binary 0-1), shape (D, H, W) hoặc (H, W)
    kernel_size: kích thước kernel (odd number)
    """
    mask = mask.squeeze().detach().cpu().numpy() # (1, D, H, W) -> (D, H, W)
    structure = np.ones((kernel_size, kernel_size, kernel_size))
    mask = ndi.binary_closing(mask, structure=structure).astype(np.uint8)
    mask = ndi.binary_opening(mask, structure=structure).astype(np.uint8)
    return torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)  # shape (1, 1, D, H, W)