import torch
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from monai.metrics import DiceMetric
from monai.metrics import HausdorffDistanceMetric
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    EnsureType
)
from medpy.metric import binary

from ..utils.metrics import  AverageMeter
from ..utils.utils import model_inferer
from ..processing.postprocessing import post_trans

def val_epoch_hd95(model, loader, max_epochs, epoch, acc_func, device, logger):
    model.eval()
    start_time = time.time()

    run_acc = AverageMeter('Loss', ':.4e')
    run_acc1 = AverageMeter('Loss', ':.4e')

    hausdorff_metric = HausdorffDistanceMetric(include_background=False, reduction='mean_batch', get_not_nans=True)
    hd_metric = []
    hd95_metric = []

    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            val_inputs, val_labels = batch_data["image"].to(device), batch_data["label"].to(device)
            logits = model_inferer(val_inputs, model)

            val_outputs_list = decollate_batch(logits)
            val_labels_list = decollate_batch(val_labels)

            val_output_convert = [post_trans(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            acc_func.reset()
            acc_func(y_pred=val_output_convert, y=val_labels_list)

            hausdorff_metric.reset()
            hausdorff_metric(y_pred=val_output_convert, y=val_labels_list)
            acc1, not_nans1 = hausdorff_metric.aggregate()
            run_acc1.update(acc1.cpu().numpy(), n=not_nans1.cpu().numpy())
            hausdorff_liver = run_acc1.avg[0]
            hausdorff_tumor = run_acc1.avg[1]

            segs = logits
            targets = val_labels
            hd = []
            hd95 = []
            dice = []

            for l in range(segs.shape[1]):
                if targets[0,l].cpu().numpy().sum() == 0:
                    hd.append(1)
                    hd95.append(0)
                    print((segs[0,l].cpu().numpy() > 0.5).sum())

                    continue
                if (segs[0,l].cpu().numpy() > 0.5).sum() == 0:
                    hd.append(0)
                    hd95.append(0)
                    continue

                hd.append(binary.hd(segs[0,l].cpu().numpy() > 0.5, targets[0,l].cpu().numpy() > 0.5, voxelspacing=None))
                hd95.append(binary.hd95(segs[0,l].cpu().numpy() > 0.5, targets[0,l].cpu().numpy() > 0.5, voxelspacing=None))

            hd_metric.append(hd)
            hd95_metric.append(hd95)

            hd_metric_mean = [np.nanmean(l) for l in zip(*hd_metric)]
            hd95_metric_mean = [np.nanmean(l) for l in zip(*hd95_metric)]
            print("mean_hd_metric: ", hd_metric_mean)
            print("mean_hd95_metric: ", hd95_metric_mean)

            print("hausdorff_liver: ", hausdorff_liver, "hausdorff_tumor: ", hausdorff_tumor)

            acc, not_nans = acc_func.aggregate()
            run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())
            dice_liver = run_acc.avg[0]
            dice_tumor = run_acc.avg[1]

            logger.info(f"Val {epoch}/{max_epochs} {idx+1}/{len(loader)}, dice_liver: {dice_liver:.6f}, dice_tumor: {dice_tumor:.6f}, time {time.time() - start_time :.2f}s")
            start_time = time.time()

    # labels = ("Liver", "Tumor")
    #metrics = {key: value for key, value in zip(labels, metrics)}

    hd_metric = [np.nanmean(l) for l in zip(*hd_metric)]
    hd95_metric = [np.nanmean(l) for l in zip(*hd95_metric)]
    print("hd_metric_final: ", hd_metric)
    print("hd95_metric_final: ", hd95_metric)

    return run_acc.avg

def calc_hd95(model, val_loader, device, weight_path, max_epochs, logger):
    model.load_state_dict(torch.load(weight_path))
    dice_acc = DiceMetric(include_background=True, reduction='mean_batch', get_not_nans=True)
    criterian_val = EDiceLoss_Val().to(device)
    # metric = criterian_val.metric

    val_acc = val_epoch_hd95(model, val_loader, max_epochs, epoch=0, acc_func=dice_acc, device=device, logger=logger)
    dice_liver, dice_tumor = val_acc[0], val_acc[1]
    val_avg_acc = np.mean(val_acc)
    print(f"\n{'*' * 20}Epoch Summary{'*' * 20}")
    print(f"Final validation stats {1}/{max_epochs}, dice_liver: {dice_liver:.6f}, dice_tumor: {dice_tumor:.6f}, Dice_Avg: {val_avg_acc:.6f}")