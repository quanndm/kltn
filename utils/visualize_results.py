import torch
import os
import numpy as np
import matplotlib.pyplot as plt

from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from ..utils.utils import inference
from ..processing.postprocessing import post_trans, post_trans_stage1
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


def visualize_results_stage_1(model, val_loader, weight_path, num_images, device):
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    stop = 0

    cmap = mcolors.ListedColormap(["black", "yellơ"])  # 0: background, 1: tumor

    for val_data in val_loader:
        stop += 1
        with torch.no_grad():
            val_input = val_data["image"].to(device)
            val_output = inference(val_input, model)  # [B, C, D, H, W]
            val_output = post_trans_stage1(val_output[0])    # keep [C, D, H, W] format

            image_np = val_data["image"].detach().cpu().numpy()[0, 0]  # [D, H, W]
            label_np = val_data["label"].detach().cpu().numpy()[0, 0]  # [D, H, W]
            pred_np = (val_output > 0.5).int().detach().cpu().numpy()[0]  # [D, H, W]

            # Combine all slices into one 2D image using max projection (any voxel > 0 will appear)
            image_2d = np.max(image_np, axis=0)
            label_2d = np.max(label_np, axis=0)
            pred_2d = np.max(pred_np, axis=0)

            # Show the image
            plt.figure(figsize=(6, 6))
            plt.title("Input Image (max projection)")
            plt.imshow(image_2d, cmap="gray")
            plt.axis("off")
            plt.show()

            # Show label and prediction
            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            ax[0].imshow(label_2d, cmap=cmap, vmin=0, vmax=1)
            ax[0].set_title("Ground Truth (Tumor)")
            ax[0].axis("off")

            ax[1].imshow(pred_2d, cmap=cmap, vmin=0, vmax=1)
            ax[1].set_title("Prediction (Tumor)")
            ax[1].axis("off")

            plt.tight_layout()
            plt.show()

        if stop == num_images:
            break