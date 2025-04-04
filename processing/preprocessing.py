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

def truncate_HU(image, hu_min=-200, hu_max=200):
    """
    Truncate the HU values to a range.
    Args:
        image: np.ndarray, the image to truncate
        hu_min: int, the minimum HU value
        hu_max: int, the maximum HU value
    Returns:
        image: np.ndarray, the truncated image
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
    zoom_factors = [target / dim for target, dim in zip(target_size, image)]
    image = zoom(image, zoom_factors, order=3)
    seg = zoom(seg, zoom_factors, order=0)
    return image, seg