import time
import torch
import numpy as np
import os

from ..processing.postprocessing import post_trans, post_label
from ..utils.utils import model_inferer
from ..utils.metrics import AverageMeter
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
        data, target = batch_data["image"].float().to(device), batch_data["label"].to(device)
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

def val_epoch(model, loader, epoch, acc_func, max_epochs, logger):
    model.eval()
    start_time = time.time()
    run_acc = AverageMeter('Loss', ':.4e')

    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            val_inputs, val_labels = batch_data["image"].to(device), batch_data["label"].to(device)
            logits = model_inferer(val_inputs, model)

            # convert val_labels to one-hot encoding
            # val_labels = val_labels.squeeze(1) # Remove channel dimension
            # val_labels = torch.nn.functional.one_hot(val_labels.long() , num_classes=val_inputs.shape[1])
            # val_labels = val_labels.permute(0, 4, 1, 2, 3)  # Change shape to (N, C, D, H, W)

            val_outputs_list = decollate_batch(logits)
            val_labels_list = decollate_batch(val_labels)

            val_output_convert = [post_trans(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            val_output_convert = [t.float() for t in val_output_convert]

            val_labels_list = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_labels_list = [t.float() for t in val_labels_list]
            
            acc_func.reset()
            acc_func(y_pred=val_output_convert, y=val_labels_list)

            acc, not_nans = acc_func.aggregate()
            run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())

            dice_liver = run_acc.avg[0]
            dice_tumor = run_acc.avg[1]
            dice_avg = np.mean(run_acc.avg)
            logger.info(f"Val {epoch}/{max_epochs} {idx+1}/{len(loader)}, Dice_Liver: {dice_liver:.6f}, Dice_Tumor: {dice_tumor:.6f}, Dice_Avg: {dice_avg:.6f}, time {time.time() - start_time:.2f}s")

            start_time = time.time()

    return run_acc.avg

def trainer(model, train_loader, val_loader, optimizer, loss_func, acc_func, scheduler, batch_size, max_epochs, start_epoch=1, val_every = 1, logger=None, path_save_model=None):
    val_acc_max, best_epoch = 0.0, 0
    total_time = time.time()
    # dices_per_class, dices_avg, loss_epochs, trains_epoch = [], [], [], []
    dices_liver, dices_tumor,dices_avg, loss_epochs, trains_epoch = [], [], [], [], []

    for epoch in range(start_epoch, max_epochs+1):
        logger.info(f"\n{'=' * 30}Training epoch {epoch}{'=' * 30}")
        epoch_time = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, epoch, loss_func, batch_size, max_epochs)

        logger.info(f"Final training epochs: {epoch}/{max_epochs} ---[loss: {train_loss:.4f}] ---[time {time.time() - epoch_time:.2f}s]")

        if scheduler is not None:
            scheduler.step()

        if epoch % val_every == 0 or epoch == max_epochs or epoch == 1:
            loss_epochs.append(train_loss)
            trains_epoch.append(epoch)
            epoch_time = time.time()
            logger.info(f"\n{'*' * 20}Epoch {epoch} Validation{'*' * 20}")
            val_acc = val_epoch(model, val_loader, epoch, acc_func, max_epochs, logger)

            val_dice_liver = val_acc[0]
            val_dice_tumor = val_acc[1]
            val_dice_avg = np.mean(val_acc)
            logger.info(f"\n{'*' * 20}Epoch Summary{'*' * 20}")
            # logger.info(f"Final validation stats {epoch}/{max_epochs},  Dice_per_class: {dice_per_class:.6f}, Dice_Avg: {dice_avg:.6f} , time {time.time() - epoch_time:.2f}s")
            logger.info(f"Final validation stats {epoch}/{max_epochs},  Dice_Liver: {val_dice_liver:.6f}, Dice_Tumor: {val_dice_tumor:.6f}, Dice_Avg: {val_dice_avg:.6f} , time {time.time() - epoch_time:.2f}s")
            
            dices_liver.append(val_dice_liver)
            dices_tumor.append(val_dice_tumor)
            dices_avg.append(val_dice_avg)

            if val_dice_avg > val_acc_max:
                print("New best ({:.6f} --> {:.6f}). At epoch {}".format(val_acc_max, val_dice_avg, epoch))
                logger.info(f"New best ({val_acc_max:.6f} --> {val_dice_avg:.6f}). At epoch {epoch}. Time consuming: {time.time()-total_time:.2f}")
                val_acc_max = val_dice_avg
                best_epoch = epoch
                torch.save(
                    model.state_dict(),
                    os.path.join(path_save_model, f"best_metric_model_{model.__class__.__name__}.pth"),
                )

            torch.cuda.empty_cache()
        torch.save(
            model.state_dict(),
            os.path.join(path_save_model, f"model_{model.__class__.__name__}_epochs_{epoch}.pth"),
        )
    logger.info(f"Training Finished !, Best Accuracy: {val_acc_max:.6f} --At epoch: {best_epoch} --Total_time: {time.time()-total_time:.2f}")
    return val_acc_max, best_epoch, dices_liver, dices_tumor, dices_avg, loss_epochs, trains_epoch
