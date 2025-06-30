import torch
from monai.inferers import sliding_window_inference
import numpy as np
from ..processing.postprocessing import post_trans_stage1, post_processing_stage2
from ..processing.preprocessing import resize_image
from skimage.transform import resize

# Assuming the variables roi_size, sw_batch_size, and overlap are lists
roi = (128, 128, 128)
roi_2d = (3, 256, 256) 
sw_batch_size = 1
overlap = 0.5

VAL_AMP = True
device = "cuda" if torch.cuda.is_available() else "cpu"

def model_inferer(input, model):
    def _compute(input):
        return sliding_window_inference(
            inputs=input,
            roi_size= roi,
            sw_batch_size=sw_batch_size,
            predictor=model,
            overlap=overlap,
        )

    if VAL_AMP:
        with torch.autocast(device):
            return _compute(input)
    else:
        return _compute(input)


def inference(input, model):
    def _compute(input):
        return sliding_window_inference(
            inputs=input,
            roi_size=roi,
            sw_batch_size=sw_batch_size,
            predictor=model,
            overlap=overlap,
        )
    if VAL_AMP:
        with torch.autocast(device):
            return _compute(input)
    else:
        return _compute(input)

        
#####################################################
# using function below in case visualizing results
#####################################################

def preprocessing_liver(image_ct, mask_liver):
    from ..datasets.lits import Lits
    image, seg = Lits.preprocessing(image_ct, mask_liver, False, normalizations="zscores")
    seg = (seg > 0).astype(np.uint8)

    image, seg = image.astype(np.float32), seg.astype(np.uint8)
    image, seg = torch.from_numpy(image), torch.from_numpy(seg)

    return image, seg

def preprocessing_tumor(compressed_array_path):
    from ..datasets.lits import  Stage2Dataset2D
    data = Stage2Dataset2D.load_npz(compressed_array_path)
    _image, _seg, bbox = data["image"], data["seg"], data["bbox"]
    image, seg = Stage2Dataset2D.preprocessing(_image, _seg, False, normalizations="zscores")

    liver_mask = (seg > 0).astype(np.uint8)
    image, seg = image.astype(np.float32), (seg == 2).astype(np.uint8)
    image, seg = torch.from_numpy(image), torch.from_numpy(seg)

    return image, seg, liver_mask, bbox


def find_best_slice(ct_array, mask_array=None, axis=0, threshold=0.01):
    """
    Tìm slice có nội dung nhiều nhất (ít background nhất) theo 1 trục.
    Nếu có mask: chọn slice có nhiều pixel mask nhất.

    Parameters:
    - ct_array: ndarray từ sitk.GetArrayFromImage
    - mask_array: ndarray mask hoặc None
    - axis: trục (0, 1, 2)
    - threshold: nếu không có mask, số pixel > ngưỡng coi là foreground

    Returns:
    - index của slice tốt nhất
    """
    if mask_array is not None:
        num_slices = mask_array.shape[axis]
        scores = []
        for i in range(num_slices):
            if axis == 0:
                mask = mask_array[i, :, :]
            elif axis == 1:
                mask = mask_array[:, i, :]
            elif axis == 2:
                mask = mask_array[:, :, i]
            scores.append(np.sum(mask))
    else:
        # Dùng CT để đánh giá slice "có thông tin"
        num_slices = ct_array.shape[axis]
        scores = []
        for i in range(num_slices):
            if axis == 0:
                sl = ct_array[i, :, :]
            elif axis == 1:
                sl = ct_array[:, i, :]
            elif axis == 2:
                sl = ct_array[:, :, i]
            scores.append(np.mean(np.abs(sl)) > threshold)

    best_index = int(np.argmax(scores))
    return best_index

def paste_mask_to_full(mask_pred, bbox, full_shape):
    """
    Đưa mask nhỏ vào lại vị trí cũ trong CT slice full size.

    Parameters:
    - mask_pred: mask 2D đã predict, shape (h, w)
    - bbox: [z, y1, x1, y2, x2] — vị trí crop ban đầu
    - full_shape: (H, W) — kích thước lát CT gốc

    Returns:
    - full_mask: mask full size với vùng predict được dán vào đúng vị trí
    """
    y_min, y_max, x_min, x_max = bbox
    full_mask = np.zeros(full_shape, dtype=mask_pred.dtype)

    h, w = y_max - y_min, x_max - x_min

    # Resize mask nếu cần thiết
    if mask_pred.shape != (h, w):
        mask_pred = resize(
            mask_pred,
            (h, w),
            order=0,  # nearest-neighbor để giữ nhãn
            preserve_range=True,
            anti_aliasing=False
        ).astype(mask_pred.dtype)

    full_mask[y_min:y_max, x_min:x_max] = mask_pred

    return np.expand_dims(full_mask, axis=0)

def predict_and_resize_mask_stage_1(model, image, mask):
    model.eval()
    ct_prep, mask_prep = preprocessing_liver(image, mask)
    ct_prep, mask_prep = ct_prep.unsqueeze(0).to(device), mask_prep.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(ct_prep)

    mask_pred = post_trans_stage1(logits.squeeze())
    mask_pred = mask_pred.cpu().numpy()
    ct_prep = ct_prep.squeeze().cpu().numpy()

    image_liver, seg_liver = resize_image(np.expand_dims(ct_prep, axis=0), np.expand_dims(mask_pred, axis=0), target_size=(image.shape[0], image.shape[1], image.shape[2]))
    image_liver, seg_liver  = image_liver.squeeze(), seg_liver.squeeze()

    return image_liver, seg_liver

def predict_and_resize_mask_stage_2(model, compressed_array_path, full_shape):
    model.eval()

    image, mask_tumor, liver_mask, bbox = preprocessing_tumor(compressed_array_path)
    ct_prep, mask_prep = image.unsqueeze(0).to(device), mask_tumor.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(ct_prep)

    mask_pred = post_processing_stage2(logits.squeeze()).cpu().numpy()
    # ct_prep = ct_prep.squeeze().cpu().numpy()
    
    mask_full = paste_mask_to_full(mask_pred.squeeze(), bbox[2:6], full_shape=full_shape)
    return mask_full