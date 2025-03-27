import time
import torch

from ..processing.postprocessing import post_trans
from utils import model_inferer
from metrics import AverageMeter

from monai.metrics import DiceMetric
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    EnsureType
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_epoch(model, loader, optimizer, epoch, loss_func, batch_size, max_epochs):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter('Loss', ':.4e')

    for idx, batch_data in enumerate(loader):
        torch.cuda.empty_cache()
        data, target = batch_data["image"].float().to(device), batch_data["label"].float().to(device)
        logits = model(data)

        loss = loss_func(logits, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        run_loss.update(loss.item(), n=batch_size)
        print(
            "Epoch {}/{} {}/{}".format(epoch, max_epochs, idx+1, len(loader)),
            "loss: {:.4f}".format(run_loss.avg),
            "time {:.2f}s".format(time.time() - start_time),
        )

        start_time = time.time()
        return run_loss.avg

# TODO: Check this function again
def val_epoch(model, loader, epoch, loss_func, batch_size, max_epochs, logger):
    model.eval()
    start_time = time.time()
    run_acc = AverageMeter('Loss', ':.4e')

    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            val_inputs, val_labels = batch_data["image"].to(device), batch_data["label"].to(device)
            logits = model_inferer(val_inputs, model)

            val_outputs_list = decollate_batch(logits)
            val_labels_list = decollate_batch(val_labels)

            val_output_convert = [post_trans(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            acc_func.reset()
            acc_func(y_pred=val_output_convert, y=val_labels_list)

            acc, not_nans = acc_func.aggregate()
            run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())

            dice_liver = run_acc.avg[0]
            dice_tumor = run_acc.avg[1]

            logger.info(f"Val {epoch}/{max_epochs} {idx+1}/{len(loader)}, dice_liver: {dice_liver:.6f}, dice_tumor: {dice_tumor:.6f}, time {time.time() - start_time :.2f}s")
            start_time = time.time()