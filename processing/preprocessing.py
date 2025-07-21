import random
import numpy as np
from scipy.ndimage import zoom
import torch
import torch.nn.functional as F

def normalize(image):
    """Basic min max scaler.
    """
    min_ = np.min(image)
    max_ = np.max(image)
    scale = max_ - min_
    image = 2 * (image - min_) / scale -1
    return image

def zscore_normalise(img: np.ndarray):
    slices = (img != 0)
    if not np.any(slices):
        return img

    values = img[slices]
    mean, std = values.mean(), values.std()
    img[slices] = (values - mean) / std if std != 0 else 0
    return img


def resize_image(image=None, seg=None, mode=None, target_size=(128, 128, 128), target_size_seg = None):
    def process_tensor(tensor, mode=None, new_size=None):
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.detach().cpu().float()
        else:
            tensor = torch.tensor(tensor, dtype=torch.float32)


        original_dim = tensor.dim()

        # Determine the appropriate mode and reshape tensor based on its dimensions

        # Default mode if not provided
        if mode is None:
            if original_dim == 3:
                mode = "bilinear" if tensor.shape[0] <= 5 else "trilinear"
            elif original_dim in {4, 5}:
                mode = "trilinear"

        if original_dim == 3:  # (D, H, W) or (C=slides, H, W)
            if mode in {"bilinear", "nearest"} :
                # (D, H, W) → (1, C=slides, H, W) or (C=slides, H, W) → (1, C=slides, H, W)
                tensor = tensor.unsqueeze(0)
            else:
                tensor = tensor.unsqueeze(0).unsqueeze(0)  # (D, H, W) → (1, 1, D, H, W)
        elif original_dim == 4:  # (C, D, H, W)
            tensor = tensor.unsqueeze(0)  # (C, D, H, W) → (1, C, D, H, W)
        elif original_dim == 5:  # Already in (1, C, D, H, W)
            pass

        align = False if mode in ["bilinear", "trilinear"] else None
        out = F.interpolate(tensor, size=new_size, mode=mode, align_corners=align)
        return out.squeeze(0).numpy()

    image_resized = process_tensor(tensor= image, new_size=target_size) if image is not None else None
    seg_resized = process_tensor(tensor = seg, mode = "nearest", new_size=target_size_seg if target_size_seg is not None else target_size) if seg is not None else None

    return image_resized, seg_resized


def truncate_HU(image, hu_min=-200, hu_max=250):
    """
    clipping the HU (Hounsfield Units) values to a range.
    Args:
        image: np.ndarray, the image to truncate
        hu_min: int, the minimum HU value
        hu_max: int, the maximum HU value
    Returns:
        image: np.ndarray, the truncated image
    
    Notes:
        >1000	        Bone, calcium, metal
        100 to 600      Iodinated CT contrast
        30 to 500	    Punctate calcifications
        60 to 100	    Intracranial hemorrhage
        35	            Gray matter
        25	            White matter
        20 to 40	    Muscle, soft tissue
        0	            Water
        -30 to -70	    Fat
        <-1000	        Air
    """
    return np.clip(image, hu_min, hu_max)

def get_bbox_liver(liver_mask, margin):
    liver_voxels = np.where(liver_mask > 0)
    if len(liver_voxels[0]) == 0:
        return (0, liver_mask.shape[0], 0, liver_mask.shape[1], 0, liver_mask.shape[2])

    z_min = 0
    z_max = liver_mask.shape[0]

    y_min = max(0, np.min(liver_voxels[1]) - margin)
    y_max = min(liver_mask.shape[1], np.max(liver_voxels[1]) + margin + 1)

    x_min = max(0, np.min(liver_voxels[2]) - margin)
    x_max = min(liver_mask.shape[2], np.max(liver_voxels[2]) + margin + 1)

    if y_max <= y_min:
        y_max = y_min + 1
    if x_max <= x_min:
        x_max = x_min + 1

    bbox = (z_min, z_max, y_min, y_max, x_min, x_max)
    return bbox


def get_liver_roi(image, seg, liver_mask_bbox):
    """
    Get the liver ROI (Region of Interest) from the image and segmentation.
    Args:
        image: np.ndarray, the image to get the ROI from, shape (D, H, W)
        seg: np.ndarray, the segmentation to get the ROI from, shape (D, H, W)
    Returns:
        image: np.ndarray, the image with the liver ROI
        seg: np.ndarray, the segmentation with the liver ROI
    """
    z_min, z_max, y_min, y_max, x_min, x_max = liver_mask_bbox

    image = image[z_min:z_max, y_min:y_max, x_min:x_max]
    seg = seg[ z_min:z_max, y_min:y_max, x_min:x_max]

    return image, seg
    

def extract_liver_mask_binary(logits, threshold=0.5):
    """
    Extract the liver mask from the predicted logits.
    Args:
        logits: tensor, output of the model, shape (1, 1, D, H, W)
        threshold: float, threshold to binarize the mask
    Returns:
        liver_mask: tensor, the liver mask, shape (1, 1, D, H, W)
    """
    probs = torch.sigmoid(logits)  # shape: (1, 1, D, H, W)
    liver_mask = (probs > threshold).float()
    return liver_mask
