import torch
import os
import numpy as np
import matplotlib.pyplot as plt

from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from ..utils.utils import inference, find_best_slice
from ..processing.postprocessing import post_trans, post_trans_stage1, post_processing_stage2, post_trans_stage2
from matplotlib.colors import ListedColormap
import matplotlib.colors as mcolors 

def visualize_results(model, val_loader, weight_path, num_images, device):
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    stop = 0
    cmap = mcolors.ListedColormap(["black", "yellow", "red"])
    for val_data in val_loader:

        stop += 1

        with torch.no_grad():

            val_input = val_data["image"].to(device)
            val_output = inference(val_input, model)
            val_output = post_trans(val_output[0]) 


            image_sample_np = val_data["image"].detach().cpu().numpy()
            label_sample_np = val_data["label"].detach().cpu().numpy()
            val_output_np = val_output.argmax(dim=0).detach().cpu().numpy()  


            z_slice = min(image_sample_np.shape[2] // 2, val_output_np.shape[0] - 1)

            image_2d = image_sample_np[0, 0, z_slice, :, :]
            label_2d = label_sample_np[0, 0, z_slice, :, :]
            pred_2d = val_output_np[z_slice, :, :]

            plt.figure("Input Image", (6, 6))
            plt.title("Input Image")
            plt.imshow(image_2d, cmap="gray")
            plt.show()

            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            ax[0].imshow(label_2d, cmap=cmap, vmin=0, vmax=2)
            ax[0].set_title("Ground Truth")
            ax[1].imshow(pred_2d, cmap=cmap, vmin=0, vmax=2)
            ax[1].set_title("Prediction")
            plt.show()
        if stop == num_images:
          break

# def visualize_results_stage_1(model, val_loader, weight_path, num_images, device):
#     model.load_state_dict(torch.load(weight_path, map_location=device))
#     model.eval()
#     stop = 0

#     cmap = mcolors.ListedColormap(["black", "yellow"])  # 0: background, 1: tumor

#     for val_data in val_loader:
#         stop += 1
#         with torch.no_grad():
#             val_input = val_data["image"].to(device)
#             val_output = inference(val_input, model)  # [B, C, D, H, W]
#             val_output = post_trans_stage1(val_output[0])    # keep [C, D, H, W] format

#             image_np = val_data["image"].detach().cpu().numpy()[0, 0]  # [D, H, W]
#             label_np = val_data["label"].detach().cpu().numpy()[0, 0]  # [D, H, W]
#             pred_np = (val_output > 0.5).int().detach().cpu().numpy()[0]  # [D, H, W]

#             # Combine all slices into one 2D image using max projection (any voxel > 0 will appear)
#             image_2d = image_np[image_np.shape[0] // 2]  # middle slice
#             label_2d = np.max(label_np, axis=0)
#             pred_2d = np.max(pred_np, axis=0)

#             # Show the image
#             plt.figure(figsize=(6, 6))
#             plt.title("Input Image")
#             plt.imshow(image_2d, cmap="gray")
#             plt.axis("off")
#             plt.show()

#             # Show label and prediction
#             fig, ax = plt.subplots(1, 2, figsize=(12, 6))
#             ax[0].imshow(label_2d, cmap=cmap, vmin=0, vmax=1)
#             ax[0].set_title("Ground Truth (liver)")
#             ax[0].axis("off")

#             ax[1].imshow(pred_2d, cmap=cmap, vmin=0, vmax=1)
#             ax[1].set_title("Prediction (liver)")
#             ax[1].axis("off")

#             plt.tight_layout()
#             plt.show()

#         if stop == num_images:
#             break

def visualize_results_stage_2(model, val_loader, weight_path, num_images, device, threshold=0.5):
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    stop = 0

    cmap = mcolors.ListedColormap(["black", "yellow"])  # 0: background, 1: tumor

    for val_data in val_loader:
        stop += 1
        with torch.no_grad():
            val_input = val_data["image"].to(device)  # input đã crop liver ROI
            val_output = model(val_input )             # raw logits

            # Apply sigmoid + thresholding + post-processing
            pred_mask = post_trans_stage2(val_output, threshold=threshold, device=device)  # [1, 1, H, W]

            # Get data to numpy
            image_np = val_input.detach().cpu().numpy()[0]            # [C, H, W]
            label_np = val_data["label"].detach().cpu().numpy()[0, 0]    # [ H, W]
            pred_np = pred_mask.detach().cpu().numpy()[0, 0]             # [H, W]

            # middle slice
            mid_slice = image_np.shape[0] // 2
            image_2d = image_np[mid_slice]  

            # Show GT and prediction
            fig, ax = plt.subplots(1, 3, figsize=(18, 6))
            # Show image
            ax[0].set_title("Input Image (Liver ROI)")
            ax[0].imshow(image_2d, cmap="gray")
            ax[0].axis("off")

            ax[1].imshow(label_np, cmap=cmap, vmin=0, vmax=1)
            ax[1].set_title("Ground Truth (Tumor)")
            ax[1].axis("off")

            ax[2].imshow(pred_np, cmap=cmap, vmin=0, vmax=1)
            ax[2].set_title("Prediction (Tumor)")
            ax[2].axis("off")

            plt.tight_layout()
            plt.show()

        if stop == num_images:
            break

################################################################
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
            val_output = inference(val_input, model)
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