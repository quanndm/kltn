import torch
import os
import numpy as np
import matplotlib.pyplot as plt

from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from ..utils.utils import inference
from ..processing.postprocessing import post_trans

class_colors = {
    0: (0, 0, 0, 255),      # Background - Black
    1: (255, 255, 0, 255),  # Liver - Yellow
    2: (255, 0, 0, 255)     # Tumor - Red
}

def visualize_results(model, val_loader, weight_path, num_images, device):
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    stop = 0
    for val_data in val_loader:
        stop += 1
        with torch.no_grad():
            val_input = val_data["image"].to(device)
            val_output = inference(val_input, model)
            val_output = post_trans(val_output[0])  # Đầu ra có dạng (C, H, W, D)

            image_sample_np = val_data["image"].detach().cpu().numpy()
            label_sample_np = val_data["label"].detach().cpu().numpy()
            val_output_np = val_output.argmax(dim=0).detach().cpu().numpy()  # Chuyển sang dạng class index

            z_slice = image_sample_np.shape[2] // 2
            
            # Hiển thị ảnh gốc
            plt.figure("Input Image", (6, 6))
            plt.title("Input Image")
            plt.imshow(image_sample_np[0, 0, z_slice], cmap="gray")
            plt.show()
            
            # Hiển thị nhãn (ground truth)
            label_colored = np.zeros((*label_sample_np.shape[2:], 4), dtype=np.uint8)
            for class_idx, color in class_colors.items():
                label_colored[label_sample_np[0, z_slice] == class_idx] = color
            
            plt.figure("Label", (6, 6))
            plt.title("Ground Truth")
            plt.imshow(label_colored)
            plt.show()
            
            # Hiển thị dự đoán
            pred_colored = np.zeros((*val_output_np.shape[1:], 4), dtype=np.uint8)
            for class_idx, color in class_colors.items():
                pred_colored[val_output_np[z_slice] == class_idx] = color
            
            plt.figure("Prediction", (6, 6))
            plt.title("Prediction")
            plt.imshow(pred_colored)
            plt.show()

            # So sánh nhãn và dự đoán
            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            ax[0].imshow(label_colored)
            ax[0].set_title("Ground Truth")
            ax[1].imshow(pred_colored)
            ax[1].set_title("Prediction")
            plt.show()
        
        if stop == num_images:
            break
