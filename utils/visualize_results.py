import torch
import os
import numpy as np
import matplotlib.pyplot as plt

from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from ..utils.utils import inference
from ..processing.postprocessing import post_trans

label_colors = ['yellow', 'red']  # Liver: Yellow, Tumor: Red
color_values = {'yellow': (255, 255, 0, 255), 'red': (255, 0, 0, 255)}
background_color = (0, 0, 0, 255)

def visualize_results(model, val_loader, weight_path, num_images, device):
    model.load_state_dict(torch.load(weight_path), map_location=device)
    model.eval()

    stop = 1
    for val_val  in val_loader:
        stop+=1
        with torch.no_grad():
            val_input = val_val["image"].to(device)
            roi_size = (128, 128, 128)
            sw_batch_size = 4
            val_output = inference(val_input, model)
            print("val_output: ", val_output.shape)
            val_output = post_trans(val_output[0])
            print("val_output: ", val_output.shape)

            image_sample_np =  val_val["image"].numpy()
            label_sample_np = val_val["label"].numpy()
            val_output_np = val_output.detach().cpu().numpy() 
            z_slice =  image_sample_np.shape[2] // 2

            # Plot input image
            plt.figure("Input Image", (6, 6))
            plt.title("Input Image")
            plt.imshow(image_sample_np[0, 0, z_slice].detach().cpu(), cmap="gray")
            plt.show()

            # Plot label
            plt.figure("Label", (12, 6))
            for i in range(2):
                plt.subplot(1, 2, i + 1)
                plt.title(f"Label channel {i}")
                plt.imshow(label_sample_np[0, i, z_slice].detach().cpu())
            plt.show()

            # Plot output
            plt.figure("Output", (12, 6))
            for i in range(2):
                plt.subplot(1, 2, i + 1)
                plt.title(f"Output channel {i}")
                plt.imshow(val_output_np[0, i, z_slice].detach().cpu())
            plt.show()

            # combine label
            image_sample_np_label = np.full((label_sample_np.shape[3], label_sample_np.shape[4], 4), background_color, dtype=np.uint8)
            num_channels_labels = label_sample_np.shape[1]
            for channel in range(num_channels_labels -1, -1, -1):
                label_channel = label_sample_np[0, channel, z_slice]

                label_color = label_colors[channel % len(label_colors)]
                color = color_values[label_color]

                label_mask = label_channel > 0
                image_sample_np_label[label_mask] = color


            # predict combine
            image_sample_np_predict = np.full((val_output_np.shape[2], val_output_np.shape[3], 4), background_color, dtype=np.uint8)
            num_channels_predict = val_output_np.shape[0]

            for channel in range(num_channels_predict -1, -1, -1):
                label_channel = val_output_np[channel, z_slice]
                label_color = label_colors[channel % len(label_colors)]

                color_value = color_values[label_color]
                label_mask = label_channel > 0

                image_sample_np_predict[label_mask] = color_value

            plt.show()

            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            ax[0].imshow(image_sample_np_label)
            ax[0].set_title("Composite Label")
            ax[1].imshow(image_sample_np_predict)
            ax[1].set_title("Composite Prediction")
            plt.show()

        if stop == num_images:
            break