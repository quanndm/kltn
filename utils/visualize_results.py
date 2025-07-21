import torch
import os
import numpy as np
import matplotlib.pyplot as plt

from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from ..utils.utils import inference, find_best_slice, paste_mask_to_full
from ..processing.postprocessing import post_trans, post_trans_stage1, post_processing_stage2, post_trans_stage2
from matplotlib.colors import ListedColormap
import matplotlib.colors as mcolors 
from monai.data import decollate_batch
import pathlib


def visualize_ct_slice(ct_array=None, mask_array=None, axis=0, slice_index=None,
                       alpha=0.4, cmap='gray', mask_cmap='tab10', tumor=False, ax=None):
    """
    show 1 slice from 3D CT image and overlay mask if provided.
    Parameters:
    - ct_array: ndarray, shape (D, H, W))
    - mask_array: ndarray, shape (D, H, W)
    - axis:  the axis to slice along (0, 1, or 2)
    - slice_index: index of slice to visualize
    - alpha:  the transparency of the mask overlay
    - ax: matplotlib axis to plot on, if None a new figure will be created
    - cmap: colormap for the CT image
    - mask_cmap: colormap for the mask overlay -> deprecated
    - tumor: if True, use custom colors for tumor visualization
    """
    if ct_array is None and mask_array is None:
        raise ValueError("Cần ít nhất một trong hai: ct_array hoặc mask_array")

    ref_array = ct_array if ct_array is not None else mask_array

    if slice_index is None:
        slice_index = ref_array.shape[axis] // 2

    # Cắt slice
    if axis == 0:
        ct_slice = ct_array[slice_index, :, :] if ct_array is not None else None
        mask_slice = mask_array[slice_index, :, :] if mask_array is not None else None
    elif axis == 1:
        ct_slice = ct_array[:, slice_index, :] if ct_array is not None else None
        mask_slice = mask_array[:, slice_index, :] if mask_array is not None else None
    elif axis == 2:
        ct_slice = ct_array[:, :, slice_index] if ct_array is not None else None
        mask_slice = mask_array[:, :, slice_index] if mask_array is not None else None
    else:
        raise ValueError("Axis phải là 0, 1 hoặc 2.")

    # create empty slice if ct_slice is None
    if ct_slice is None:
        ct_slice = np.zeros_like(mask_slice, dtype=np.uint8)

    # choose axis to plot on
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    ax.imshow(ct_slice, cmap=cmap)

    # Overlay mask if provided
    if mask_slice is not None:
        if np.array_equal(np.unique(mask_slice), [0, 1, 2]):
            custom_colors = [
                [0, 0, 0, 0.0],            # label 0
                [0, 1.0, 127/255, 0.5],    # label 1
                [1.0, 0, 0, 1.0],          # label 2
            ]
        else:
            if tumor:
                custom_colors = [
                    [0, 0, 0, 0.0],        # label 0
                    [1.0, 0, 0, 1.0],      # label 2
                ]
            else:
                custom_colors = [
                    [0, 0, 0, 0.0],        # label 0
                    [0, 1.0, 127/255, 0.5],# label 1
                ]
        custom_cmap = ListedColormap(custom_colors)
        ax.imshow(mask_slice, cmap=custom_cmap, alpha=alpha, interpolation='none')
    # cut off margins
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

    # ax.set_title(f"Slice {slice_index} (axis={axis})")
    ax.axis('off')

    if ax is None:
        plt.show()


def visualize_results_stage_1(model, val_loader, weight_path, num_images, device):
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    stop = 0
    for val_data in val_loader:
        stop += 1
        with torch.no_grad():
            val_input = val_data["image"].to(device)
            val_output = model(val_input)
            val_output = post_trans_stage1(val_output[0]).squeeze().cpu()  

            image = val_input.detach().cpu().numpy()[0, 0] 
            mask =  val_data["label"].detach().cpu().numpy()[0, 0] 
            best_slide = find_best_slice(val_output, mask)

            fig, axes = plt.subplots(1, 5, figsize=(20, 4))
            fig.patch.set_visible(False)

            visualize_ct_slice(image, None, slice_index=best_slide,  ax=axes[0])
            visualize_ct_slice(image, mask, slice_index=best_slide, tumor=False, alpha=0.5,  ax=axes[1])
            visualize_ct_slice(image, val_output, slice_index=best_slide, tumor=False, alpha=0.5,  ax=axes[2])
            visualize_ct_slice(None, mask, slice_index=best_slide, tumor=False, alpha=1,  ax=axes[3])
            visualize_ct_slice(None, val_output, slice_index=best_slide, tumor=False, alpha=1,  ax=axes[4])

        if stop == num_images:
            break

def visualize_results_stage_2(model, val_loader, weight_path, num_images, device, threshold=0.5):
    from ..init.install_dependencies import  load_config
    from ..datasets.lits import Lits
    config = load_config("./kltn/parameters.yaml")

    base_folder  = pathlib.Path(config["source_folder_lits"]).resolve()
    volumes =  list(base_folder.glob('volume-*.nii')) 

    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    stop = 0
    for val_data in val_loader:
        stop += 1
        with torch.no_grad():
            val_input = val_data["image"].to(device)  # (b, c, h, w)
            logits = model(val_input)  # raw logits (b, 1, h, w)
            
            patient_ids = val_data["patient_id"]
            slides = val_data["slide"]
            bboxes = val_data["bbox"]
            val_outputs_list = decollate_batch(logits) # list of tensors shape( 1, H, W)
            val_output_convert = [post_processing_stage2(val_pred_tensor, threshold=threshold).to(device).float().unsqueeze(0) for val_pred_tensor in val_outputs_list] # list of tensors shape(1, H, W)

            for i, val_output in enumerate(val_output_convert):
                if val_output.sum().item() <= 0:
                    continue

                patient_id = patient_ids[i]
                slide = slides[i]
                bbox = bboxes[i][2:6] # [y1, y2, x1, x2]

                volume_path = next((vol for vol in volumes if patient_id in vol.name), None)
                mask_path = base_folder / volume_path.name.replace("volume", "segmentation")

                volume = Lits.load_nii(volume_path) # D, H, W
                mask = Lits.load_nii(mask_path) # D, H, W

                volume = np.clip(volume, -200, 250)
                mask = (mask == 2).astype(np.uint8)

                if volume.shape[0] <= int(slide):
                    continue

                if mask[int(slide)].sum().item() <= 0:
                    continue
                    
                mask_pred_full = paste_mask_to_full(val_output.squeeze().cpu().numpy(), bbox, full_shape=volume[int(slide)].shape)


                fig, axes = plt.subplots(1, 5, figsize=(20, 4)) 
                fig.patch.set_visible(False)

                visualize_ct_slice(volume, None, slice_index=int(slide),  ax=axes[0])
                visualize_ct_slice(volume, mask, slice_index=int(slide), tumor=True, alpha=0.5,  ax=axes[1])
                visualize_ct_slice(np.expand_dims(volume[int(slide)], axis=0), mask_pred_full, tumor=True, alpha=0.5,  ax=axes[2])
                visualize_ct_slice(None, mask, slice_index=int(slide), tumor=True, alpha=1,  ax=axes[3])
                visualize_ct_slice(None, mask_pred_full, tumor=True, alpha=1,  ax=axes[4])

        if stop == num_images:
            break