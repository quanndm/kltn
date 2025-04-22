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

def resize_image(image, seg, target_size=(128, 128, 128)):
    def process_tensor(tensor, mode, new_size=target_size):
        tensor = torch.tensor(tensor, dtype=torch.float32)

        tensor = tensor.unsqueeze(0)  
        tensor_resized = F.interpolate(tensor, size=new_size, mode=mode, align_corners=False if mode == "trilinear" else None)
        return tensor_resized.squeeze(0)  

    image_resized = process_tensor(image, "trilinear")
    seg_resized = process_tensor(seg, "nearest")
    return image_resized.numpy(), seg_resized.numpy()

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

def resize_image_v2(image, seg, target_size=(128, 128, 128)):
    """
    Resize the image and segmentation to the target size.
    Args:
        image: np.ndarray, the image to resize, shape (D, H, W)
        seg: np.ndarray, the segmentation to resize, shape (D, H, W)
        target_size: tuple, the target size of the image and segmentation
    Returns:
        image: np.ndarray, the resized image
        seg: np.ndarray, the resized segmentation
    """
    zoom_factors = [target / dim for target, dim in zip(target_size, image.shape)]
    image = zoom(image, zoom_factors, order=3)
    seg = zoom(seg, zoom_factors, order=0)
    return image, seg

def get_liver_roi(image, seg, margin=5):
    """
    Get the liver ROI (Region of Interest) from the image and segmentation.
    Args:
        image: np.ndarray, the image to get the ROI from, shape (D, H, W)
        seg: np.ndarray, the segmentation to get the ROI from, shape (D, H, W)
    Returns:
        image: np.ndarray, the image with the liver ROI
        seg: np.ndarray, the segmentation with the liver ROI
    """
    # get liver ROI
    liver_voxels = np.where(seg > 0)
    
    if len(liver_voxels[0]) == 0:
        return image, seg
    z_min = max(0, np.min(liver_voxels[0]) - margin)
    z_max = min(image.shape[0], np.max(liver_voxels[0]) + margin + 1)

    y_min = max(0, np.min(liver_voxels[1]) - margin)
    y_max = min(image.shape[1], np.max(liver_voxels[1]) + margin + 1)

    x_min = max(0, np.min(liver_voxels[2]) - margin)
    x_max = min(image.shape[2], np.max(liver_voxels[2]) + margin + 1)

    image = image[z_min:z_max, y_min:y_max, x_min:x_max]
    seg = seg[z_min:z_max, y_min:y_max, x_min:x_max]

    bbox = (z_min, z_max, y_min, y_max, x_min, x_max)
    return image, seg, bbox
    
def extract_liver_mask(pred_logits):
    """
    Extract the liver mask from the predicted logits.
    Args:
        pred_logits: tensor, output of the model, shape (B, 3, D, H, W)
    Returns:
        liver_mask: tensor, the liver mask, shape (B,1, D, H, W)
    """
    out = torch.softmax(pred_logits, dim=1)
    liver_mask = (out.argmax(dim=1) == 1).float().unsqueeze(1)

    return liver_mask

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

def mask_input_with_liver(img, liver_mask):
    """
    Input: inputs (1, D, H, W), mask liver ( 1, D, H, W)
    Output: masked inputs (1, D, H, W) only liver region
    """
    return img * liver_mask


def pad_image(image, mask, pad_width):
    """
    Pad ảnh và mask nếu cần, để tránh mất dữ liệu khi crop gần biên
    Args:
        pad_width: int, số voxel cần pad mỗi phía (z, y, x)
    Returns:
        padded image, padded mask
    """
    return np.pad(image, ((pad_width, pad_width), (pad_width, pad_width), (pad_width, pad_width)), mode='constant'), np.pad(mask, ((pad_width, pad_width), (pad_width, pad_width), (pad_width, pad_width)), mode='constant')

def crop_patch_around_tumor(image, tumor_mask, patch_size=(96, 96, 96), margin=15):
    """
    image: np.ndarray, shape (D, H, W)
    tumor_mask: np.ndarray, shape (D, H, W), binary mask (0 or 1)
    patch_size: size of the output patch (D, H, W)
    margin: number of voxels to expand around tumor

    return: cropped image and mask patch
    """

    # pad the image and mask to avoid losing data when cropping near the edges
    pad_width = (patch_size[0] // 2) + margin
    image, tumor_mask = pad_image(image, tumor_mask, pad_width)

    # 1. find tumor coordinates
    coords = np.argwhere(tumor_mask > 0)
    if coords.shape[0] == 0:
        #  if no tumor, random crop
        # Randomly crop a patch from the image
        D, H, W = image.shape
        start_z = np.random.randint(0, max(1, D - patch_size[0]))
        start_y = np.random.randint(0, max(1, H - patch_size[1]))
        start_x = np.random.randint(0, max(1, W - patch_size[2]))
    else:
        min_z, min_y, min_x = coords.min(0) - margin
        max_z, max_y, max_x = coords.max(0) + margin

        # limit to image size
        min_z = max(0, min_z)
        min_y = max(0, min_y)
        min_x = max(0, min_x)
        max_z = min(image.shape[0], max_z)
        max_y = min(image.shape[1], max_y)
        max_x = min(image.shape[2], max_x)

        # get center of the tumor
        center_z = (min_z + max_z) // 2
        center_y = (min_y + max_y) // 2
        center_x = (min_x + max_x) // 2

        # solve for the start coordinates of the patch
        start_z = max(0, center_z - patch_size[0] // 2)
        start_y = max(0, center_y - patch_size[1] // 2)
        start_x = max(0, center_x - patch_size[2] // 2)

        # ensure the patch is within the image bounds
        start_z = min(start_z, image.shape[0] - patch_size[0])
        start_y = min(start_y, image.shape[1] - patch_size[1])
        start_x = min(start_x, image.shape[2] - patch_size[2])

    # 2. Crop the patch from the image and mask
    end_z = start_z + patch_size[0]
    end_y = start_y + patch_size[1]
    end_x = start_x + patch_size[2]

    img_patch = image[start_z:end_z, start_y:end_y, start_x:end_x]
    mask_patch = tumor_mask[start_z:end_z, start_y:end_y, start_x:end_x]

    return img_patch, mask_patch