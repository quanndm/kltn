import time
import torch
import numpy as np

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

def val_epoch(model, loader, epoch, acc_func, criterian_val, metric, max_epochs, logger):
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

    return run_acc.avg

def trainer(model, train_loader, val_loader, optimizer, loss_func, acc_func, criterian_val, metric, scheduler, batch_size, max_epochs, start_epoch=1, val_every = 1, logger=None, path_save_model=None):
    val_acc_max, best_epoch = 0.0, 0
    total_time = time.time()
    dices_liver , dices_tumor, dices_avg, loss_epochs, trains_epoch = [], [], [], [], []

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
            logger.info(f"\t{'*' * 20}Epoch {epoch} Validation{'*' * 20}")
            val_acc = val_epoch(model, val_loader, epoch, acc_func, criterian_val, metric, max_epochs, logger)

            dice_liver, dice_tumor = val_acc[0], val_acc[1]
            val_avg_acc = np.mean(val_acc)  
            logger.info(f"\t{'*' * 20}Epoch Summary{'*' * 20}")
            logger.info(f"Final validation stats {epoch}/{max_epochs},  dice_liver: {dice_liver:.6f}, dice_tumor: {dice_tumor:.6f}, Dice_Avg: {val_avg_acc:.6f} , time {time.time() - epoch_time:.2f}s")

            dices_liver.append(dice_liver)
            dices_tumor.append(dice_tumor)
            dices_avg.append(val_avg_acc)

            if val_avg_acc > val_acc_max:
                print("New best ({:.6f} --> {:.6f}). At epoch {}".format(val_acc_max, val_avg_acc, epoch))
                logger.info(f"New best ({val_acc_max:.6f} --> {val_avg_acc:.6f}). At epoch {epoch}. Time consuming: {time.time()-total_time:.2f}")
                val_acc_max = val_avg_acc
                best_epoch = epoch
                torch.save(
                    model.state_dict(),
                    os.path.join(path_save_model, "best_metric_model.pth"),
                )
            
            torch.cuda.empty_cache()

    logger.info(f"Training Finished !, Best Accuracy: {val_acc_max:.6f} --At epoch: {best_epoch} --Total_time: {time.time()-total_time:.2f}")
    return val_avg_acc, dice_liver, dice_tumor, dices_avg, loss_epochs, trains_epoch
