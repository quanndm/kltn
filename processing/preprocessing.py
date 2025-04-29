import random
import numpy as np
from scipy.ndimage import zoom
import torch
import torch.nn.functional as F

def get_crop_slice(target_size, dim):
    if dim > target_size:
        crop_extent = dim - target_size
        left = random.randint(0, crop_extent)
        right = crop_extent - left
        return slice(left, dim - right)
    elif dim <= target_size:
        return slice(0, dim)

def get_left_right_idx_should_pad(target_size, dim):
    if dim >= target_size:
        return [False]
    elif dim < target_size:
        pad_extent = target_size - dim
        left = random.randint(0, pad_extent)
        right = pad_extent - left
        return True, left, right

def pad_or_crop_image(image, seg=None, target_size=(128, 144, 144)):
    c, z, y, x = image.shape
    z_slice, y_slice, x_slice = [get_crop_slice(target, dim) for target, dim in zip(target_size, (z, y, x))]
    image = image[:, z_slice, y_slice, x_slice]
    if seg is not None:
        seg = seg[:, z_slice, y_slice, x_slice]
    todos = [get_left_right_idx_should_pad(size, dim) for size, dim in zip(target_size, [z, y, x])]
    padlist = [(0, 0)]  # channel dim
    for to_pad in todos:
        if to_pad[0]:
            padlist.append((to_pad[1], to_pad[2]))
        else:
            padlist.append((0, 0))
    image = np.pad(image, padlist)
    if seg is not None:
        seg = np.pad(seg, padlist)
        return image, seg
    return image


def normalize(image):
    """Basic min max scaler.
    """
    min_ = np.min(image)
    max_ = np.max(image)
    scale = max_ - min_
    image = (image - min_) / scale
    return image

def zscore_normalise(img: np.ndarray):
    slices = (img != 0)
    img[slices] = (img[slices] - np.mean(img[slices])) / np.std(img[slices])
    return img


def irm_min_max_preprocess(image, low_perc=1, high_perc=99):
    non_zeros = image > 0
    low, high = np.percentile(image[non_zeros], [low_perc, high_perc])
    image = np.clip(image, low, high)
    image = normalize(image)
    return image

def resize_image(image=None, seg=None, target_size=(128, 128, 128), device=None):
    device = device if device is not None else torch.device("cpu")
    def process_tensor(tensor, mode, new_size=target_size):
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.clone().detach().float().to(device)
        else:
            tensor = torch.tensor(tensor, dtype=torch.float32, device=device)

        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0).unsqueeze(0)  # [D, H, W] -> [1, 1, D, H, W]
        elif tensor.dim() == 4:
            tensor = tensor.unsqueeze(0)  # [C, D, H, W] -> [1, C, D, H, W]
        elif tensor.dim() == 5:
            pass  # Already [N, C, D, H, W]

        
        return F.interpolate(tensor, size=new_size, mode=mode, align_corners=(False if mode == "trilinear" else None)).squeeze(0).cpu().numpy()

    image_resized = process_tensor(image, "trilinear", new_size=target_size) if image is not None else None
    seg_resized = process_tensor(seg, "nearest", new_size=target_size) if seg is not None else None

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
        100 to 600   Iodinated CT contrast
        30 to 500	  Punctate calcifications
        60 to 100	  Intracranial hemorrhage
        35	                Gray matter
        25	                White matter
        20 to 40	   Muscle, soft tissue
        0	                  Water
        -30 to -70	   Fat
        <-1000	        Air
    """
    return np.clip(image, hu_min, hu_max)

def get_bbox_liver(liver_mask, margin):
    liver_voxels = np.where(liver_mask > 0)

    if len(liver_voxels[0]) == 0:
        return (0, liver_mask.shape[0], 0, liver_mask.shape[1], 0, liver_mask.shape[2])

    z_min = max(0, np.min(liver_voxels[0]) - margin)
    z_max = min(liver_mask.shape[0], np.max(liver_voxels[0]) + margin + 1)

    y_min = max(0, np.min(liver_voxels[1]) - margin)
    y_max = min(liver_mask.shape[1], np.max(liver_voxels[1]) + margin + 1)

    x_min = max(0, np.min(liver_voxels[2]) - margin)
    x_max = min(liver_mask.shape[2], np.max(liver_voxels[2]) + margin + 1)

    if z_max <= z_min:
        z_max = z_min + 1
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
    if z_max <= z_min:
        z_max = z_min + 1
    if y_max <= y_min:
        y_max = y_min + 1
    if x_max <= x_min:
        x_max = x_min + 1

    image = image[z_min:z_max, y_min:y_max, x_min:x_max]
    seg = seg[z_min:z_max, y_min:y_max, x_min:x_max]

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

def resize_crop_to_bbox_size(tensor_crop, bbox):
    """
    Resize tensor crop (e.g., predicted tumor mask) to the size of the liver ROI crop (dz, dy, dx).
    
    Args:
        tensor_crop (tensor): The tensor crop to be resized. shape: (1, D, H, W)
        bbox (tuple): The bounding box of the liver ROI crop.

    Returns:
        np.ndarray: (tensor): The resized tensor crop. shape: (1, dz, dy, dx)
    """
    # Ensure tensor is float for interpolation
    tensor_crop = tensor_crop.unsqueeze(0) # shape 1, D, H, W => 1, 1, D, H, W
    # tarrget size
    dz, dy, dx = bbox[1] - bbox[0], bbox[3] - bbox[2], bbox[5] - bbox[4]
    target_size = (dz, dy, dx)
    resized = F.interpolate(
        tensor_crop.float(),
        size=target_size,
        mode="nearest"  # because it's a binary mask
    )

    return resized.squeeze(0).to(torch.uint8)

def uncrop_to_full_image(crop_mask, bbox, full_image_shape):
    """
    Uncrop the cropped mask to the full image size.
    
    Args:
        crop_mask (tensor): The cropped mask to be uncropped.
        bbox (tuple): The bounding box of the liver ROI crop.
        full_image_shape (tuple): The shape of the full image.

    Returns:
        np.ndarray: (tensor): The uncropped mask.
    """
    # Create a tensor of zeros with the same shape as the full image
    full_mask = torch.zeros(full_image_shape, dtype=torch.uint8, device=crop_mask.device)

    # Get the bounding box coordinates
    z_min, z_max, y_min, y_max, x_min, x_max = bbox

    crop_mask = crop_mask.squeeze(0)  # shape: (1, dz, dy, dx) -> (dz, dy, dx)
    # Place the cropped mask in the correct position in the full mask
    full_mask[z_min:z_max, y_min:y_max, x_min:x_max] = crop_mask

    return full_mask.unsqueeze(0)  # shape: (dz, dy, dx) -> (1, dz, dy, dx)