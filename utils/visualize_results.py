import torch
import os
import numpy as np
import matplotlib.pyplot as plt

from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from ..utils.utils import inference
from ..processing.postprocessing import post_trans_v2

def visualize_results(model, val_loader, weight_path, num_images, device):
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    stop = 0
    for val_data in val_loader:
        stop += 1
        with torch.no_grad():
            val_input = val_data["image"].to(device)
            val_output = inference(val_input, model)
            val_output = post_trans_v2(val_output[0]) 

            image_sample_np = val_data["image"].detach().cpu().numpy()
            label_sample_np = val_data["label"].detach().cpu().numpy()
            label_sample_np = np.argmax(label_sample_np[0], axis=0)

            val_output_np = val_output.argmax(dim=0).detach().cpu().numpy()  


            z_slice = min(image_sample_np.shape[2] // 2, val_output_np.shape[0] - 1)
            
            image_2d = image_sample_np[0, 0, z_slice, :, :]
            label_2d = label_sample_np[z_slice, :, :]
            pred_2d = val_output_np[z_slice, :, :]
            
            colors = ["black", "red", "green"]
            cmap = ListedColormap(colors)
            bounds = [0, 1, 2, 3]
            norm = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)


            plt.figure("Input Image", (6, 6))
            plt.title("Input Image")
            plt.imshow(image_2d, cmap="gray")
            plt.show()
            

            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            ax[0].imshow(label_2d, cmap=cmap, norm=norm)
            ax[0].set_title("Ground Truth")
            ax[1].imshow(pred_2d, cmap=cmap, norm=norm)
            ax[1].set_title("Prediction")
            plt.show()  
        
        if stop == num_images:
            break