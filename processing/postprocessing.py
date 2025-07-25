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

post_trans = Compose(
    [EnsureType(), Activations(softmax=True),AsDiscrete(argmax=True, to_onehot=3)]
)

post_trans_stage1 = post_trans_stage2 = Compose(
    [EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold=0.5)]
)

def post_processing_stage2(logits, threshold=0.5, device=None):
    """
    Post-processing for stage 2 of the model.
    Args:
        logits (torch.Tensor): The logits output from the model, shape  1, H, W).
        threshold (float): The threshold for binarizing the logits.
    Returns:
        torch.Tensor: The post-processed logits.
    """
    # Apply sigmoid activation function
    probs = torch.sigmoid(logits)

    # Convert to numpy array and apply threshold
    tumor_mask = probs.cpu().numpy()
    tumor_mask =(tumor_mask > threshold).astype(np.uint8)
    tumor_mask = np.squeeze(tumor_mask) # shape(1, H, W) -> (H, W)

    # Morphological closing and opening
    tumor_mask = morphology.binary_closing(tumor_mask, morphology.disk(3))
    tumor_mask = morphology.binary_opening(tumor_mask, morphology.disk(3))

    # fill holes
    tumor_mask = ndi.binary_fill_holes(tumor_mask)

    tumor_mask = np.expand_dims(tumor_mask, axis=0) # shape( H, W) -> (1, H, W)
    tumor_mask = torch.from_numpy(tumor_mask).to(device) # shape(1, H, W)

    return tumor_mask
