import time
import torch
import numpy as np
import os

from ..processing.postprocessing import post_trans, post_trans_stage2, post_trans_stage1, post_processing_stage2
from ..utils.utils import model_inferer
from ..utils.metrics import AverageMeter, IoUMetric, PrecisionMetric, RecallMetric
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
    iou_metric = IoUMetric(num_classes=3, ignore_background=True)
    precision_metric = PrecisionMetric(num_classes=3, ignore_background=True)
    recall_metric = RecallMetric(num_classes=3, ignore_background=True)

    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            val_inputs, val_labels = batch_data["image"].to(device), batch_data["label"].to(device)
            logits = model_inferer(val_inputs, model)

            val_outputs_list = decollate_batch(logits)
            val_labels_list = decollate_batch(val_labels)

            val_output_convert = [post_trans(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            val_output_convert = [t.float() for t in val_output_convert]

            val_labels_list = [t.float() for t in val_labels_list]
            
            acc_func.reset()
            acc_func(y_pred=val_output_convert, y=val_labels_list)

            acc, not_nans = acc_func.aggregate()
            run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())

            dice_liver = run_acc.avg[0]
            dice_tumor = run_acc.avg[1]
            dice_avg = np.mean(run_acc.avg)

            ious = iou_metric(logits, val_labels)
            iou_liver = ious[0]
            iou_tumor = ious[1]
            iou_avg = np.mean(ious)

            precisions = precision_metric(logits, val_labels)
            precision_liver = precisions[0]
            precision_tumor = precisions[1]
            precision_avg = np.mean(precisions)

            recalls = recall_metric(logits, val_labels)
            recall_liver = recalls[0]
            recall_tumor = recalls[1]
            recall_avg = np.mean(recalls)

            logger.info(f"Val {epoch}/{max_epochs} {idx+1}/{len(loader)}, Dice_Liver: {dice_liver:.6f}, Dice_Tumor: {dice_tumor:.6f}, Dice_Avg: {dice_avg:.6f}, time {time.time() - start_time:.2f}s")


            start_time = time.time()

    return run_acc.avg, ious, precisions, recalls

def val_epoch_stage1(model, loader, epoch, acc_func, max_epochs, logger):
    model.eval()
    start_time = time.time()
    run_acc = AverageMeter('Loss', ':.4e')
    iou_metric = IoUMetric(num_classes=1, ignore_background=True)
    precision_metric = PrecisionMetric(num_classes=1, ignore_background=True)
    recall_metric = RecallMetric(num_classes=1, ignore_background=True)

    dice_list, iou_list, precision_list, recall_list = [], [], [], []
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            val_inputs, val_labels = batch_data["image"].to(device), batch_data["label"].to(device)
            logits = model_inferer(val_inputs, model)

            val_outputs_list = decollate_batch(logits)
            val_labels_list = decollate_batch(val_labels)

            val_output_convert = [post_trans_stage1(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            val_output_convert = [t.float() for t in val_output_convert]

            val_labels_list = [t.float() for t in val_labels_list]
            
            acc_func.reset()
            acc_func(y_pred=val_output_convert, y=val_labels_list)

            acc, not_nans = acc_func.aggregate()
            # run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())

            dice_liver = acc[0]
            dice_list.append(dice_liver.cpu().numpy())

            ious = iou_metric(logits, val_labels)
            iou_list.append(ious[0])

            precisions = precision_metric(logits, val_labels)
            precision_list.append(precisions[0])

            recalls = recall_metric(logits, val_labels)
            recall_list.append(recalls[0])

            logger.info(f"Val {epoch}/{max_epochs} {idx+1}/{len(loader)}, Dice_Liver: {dice_liver:.6f}, time {time.time() - start_time:.2f}s")

            start_time = time.time()

    acc = np.mean(dice_list)
    ious = np.mean(iou_list)
    precisions = np.mean(precision_list)
    recalls = np.mean(recall_list)
    return acc, ious, precisions, recalls

def val_epoch_stage2(model, loader, epoch, acc_func, max_epochs, logger):
    model.eval()
    start_time = time.time()
    # run_acc = AverageMeter('Loss', ':.4e')
    iou_metric = IoUMetric(num_classes=1, ignore_background=True)
    precision_metric = PrecisionMetric(num_classes=1, ignore_background=True)
    recall_metric = RecallMetric(num_classes=1, ignore_background=True)

    dice_list = []
    iou_list, precision_list, recall_list = [], [], []
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            val_inputs, val_labels = batch_data["image"].to(device), batch_data["label"].to(device)
            logits = model_inferer(val_inputs, model)
            logits = post_processing_stage2(logits, threshold=0.5, device=device)

            val_outputs_list = decollate_batch(logits)
            val_labels_list = decollate_batch(val_labels)

            val_output_convert = [t.float() for t in val_outputs_list]
            val_labels_list = [t.float() for t in val_labels_list]
            
            acc_func.reset()
            acc_func(y_pred=val_output_convert, y=val_labels_list)

            acc, not_nans = acc_func.aggregate()

            dice_tumor = acc[0]
            dice_list.append(dice_tumor.cpu().numpy())

            ious = iou_metric(logits, val_labels)
            iou_list.append(ious[0])

            precisions = precision_metric(logits, val_labels)
            precision_list.append(precisions[0])

            recalls = recall_metric(logits, val_labels)
            recall_list.append(recalls[0])

            logger.info(f"Val {epoch}/{max_epochs} {idx+1}/{len(loader)}, Dice_Tumor: {dice_tumor:.6f}, time {time.time() - start_time:.2f}s")

            start_time = time.time()

    acc = np.mean(dice_list)
    ious = np.mean(iou_list)
    precisions = np.mean(precision_list)
    recalls = np.mean(recall_list)
    return acc, ious, precisions, recalls


def trainer(model, train_loader, val_loader, optimizer, loss_func, acc_func, scheduler, batch_size, max_epochs, start_epoch=1, val_every = 1, logger=None, path_save_model=None, save_model=True, post_fix=None):
    val_acc_max, best_epoch = 0.0, 0
    total_time = time.time()
    # dices_per_class, dices_avg, loss_epochs, trains_epoch = [], [], [], []
    dices_liver, dices_tumor,dices_avg, loss_epochs, trains_epoch = [], [], [], [], []
    ious_liver, ious_tumor, ious_avg = [], [], []
    precisions_liver, precisions_tumor, precisions_avg = [], [], []
    recalls_liver, recalls_tumor, recalls_avg = [], [], []

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
            val_acc , val_ious, val_precisions, val_recalls= val_epoch(model, val_loader, epoch, acc_func, max_epochs, logger)

            val_dice_liver = val_acc[0]
            val_dice_tumor = val_acc[1]
            val_dice_avg = np.mean(val_acc)

            val_iou_liver = val_ious[0]
            val_iou_tumor = val_ious[1]
            val_iou_avg = np.mean(val_ious)

            val_precision_liver = val_precisions[0]
            val_precision_tumor = val_precisions[1]
            val_precision_avg = np.mean(val_precisions)

            val_recall_liver = val_recalls[0]
            val_recall_tumor = val_recalls[1]
            val_recall_avg = np.mean(val_recalls)
            logger.info(f"\n{'*' * 20}Epoch Summary{'*' * 20}")
            logger.info(f"Final validation stats {epoch}/{max_epochs},  Dice_Liver: {val_dice_liver:.6f}, Dice_Tumor: {val_dice_tumor:.6f}, Dice_Avg: {val_dice_avg:.6f} , time {time.time() - epoch_time:.2f}s")
            
            dices_liver.append(val_dice_liver)
            dices_tumor.append(val_dice_tumor)
            dices_avg.append(val_dice_avg)

            ious_liver.append(val_iou_liver)
            ious_tumor.append(val_iou_tumor)
            ious_avg.append(val_iou_avg)

            precisions_liver.append(val_precision_liver)
            precisions_tumor.append(val_precision_tumor)
            precisions_avg.append(val_precision_avg)

            recalls_liver.append(val_recall_liver)
            recalls_tumor.append(val_recall_tumor)
            recalls_avg.append(val_recall_avg)
            if val_dice_avg > val_acc_max:
                print("New best ({:.6f} --> {:.6f}). At epoch {}".format(val_acc_max, val_dice_avg, epoch))
                logger.info(f"New best ({val_acc_max:.6f} --> {val_dice_avg:.6f}). At epoch {epoch}. Time consuming: {time.time()-total_time:.2f}")
                val_acc_max = val_dice_avg
                best_epoch = epoch
                if save_model:
                    model_filename = f"best_metric_model_{model.__class__.__name__}"
                    if post_fix is not None:
                        model_filename += f"_{post_fix}"
                    model_filename += ".pth"
                    torch.save(
                        model.state_dict(),
                        os.path.join(path_save_model, model_filename),
                    )

            torch.cuda.empty_cache()
        if epoch % 10 == 0 or epoch == max_epochs or epoch == 1:
            logger.info(f"Epoch {epoch}/{max_epochs} ---[loss: {train_loss:.4f}] ---[val_dice: {val_dice_avg:.6f}] ---[time {time.time() - epoch_time:.2f}s]")
            # Save the model every 10 epochs
            if save_model:
                model_filename = f"model_{model.__class__.__name__}_epochs_{epoch}"
                if post_fix is not None:
                    model_filename += f"_{post_fix}"
                model_filename += ".pth"
                torch.save(
                    model.state_dict(),
                    os.path.join(path_save_model, model_filename),
                )
    logger.info(f"Training Finished !, Best Accuracy: {val_acc_max:.6f} --At epoch: {best_epoch} --Total_time: {time.time()-total_time:.2f}")
    time_tmp  = time.time()-total_time
    return val_acc_max, best_epoch, dices_liver, dices_tumor, dices_avg, loss_epochs, trains_epoch, ious_liver, ious_tumor, ious_avg, precisions_liver, precisions_tumor, precisions_avg, recalls_liver, recalls_tumor, recalls_avg, time_tmp

def trainer_stage1(model, train_loader, val_loader, optimizer, loss_func, acc_func, scheduler, batch_size, max_epochs, start_epoch=1, val_every = 1, logger=None, path_save_model=None, save_model=True, post_fix=None):
    val_acc_max, best_epoch = 0.0, 0
    total_time = time.time()
    dices_liver, loss_epochs, trains_epoch = [], [], []
    ious_liver = []
    precisions_liver = []
    recalls_liver = []

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
            val_acc , val_ious, val_precisions, val_recalls= val_epoch_stage1(model, val_loader, epoch, acc_func, max_epochs, logger)

            val_dice_liver = val_acc
            val_iou_liver = val_ious
            val_precision_liver = val_precisions
            val_recall_liver = val_recalls

            logger.info(f"\n{'*' * 20}Epoch Summary{'*' * 20}")
            logger.info(f"Final validation stats {epoch}/{max_epochs},   Dice_Liver: {val_dice_liver:.6f} , time {time.time() - epoch_time:.2f}s")
            
            dices_liver.append(val_dice_liver)
            ious_liver.append(val_iou_liver)
            precisions_liver.append(val_precision_liver)
            recalls_liver.append(val_recall_liver)

            if val_dice_liver > val_acc_max:
                print("New best ({:.6f} --> {:.6f}). At epoch {}".format(val_acc_max, val_dice_liver, epoch))
                logger.info(f"New best ({val_acc_max:.6f} --> {val_dice_liver:.6f}). At epoch {epoch}. Time consuming: {time.time()-total_time:.2f}")
                val_acc_max = val_dice_liver
                best_epoch = epoch
                if save_model:
                    model_filename = f"best_metric_model_{model.__class__.__name__}"
                    if post_fix is not None:
                        model_filename += f"_{post_fix}"
                    model_filename += ".pth"
                    torch.save(
                        model.state_dict(),
                        os.path.join(path_save_model, model_filename),
                    )

            torch.cuda.empty_cache()
        if epoch % 10 == 0 or epoch == max_epochs or epoch == 1:
            logger.info(f"Epoch {epoch}/{max_epochs} ---[loss: {train_loss:.4f}] ---[val_dice: {val_dice_liver:.6f}] ---[time {time.time() - epoch_time:.2f}s]")
            # Save the model every 10 epochs
            if save_model:
                model_filename = f"model_{model.__class__.__name__}_epochs_{epoch}"
                if post_fix is not None:
                    model_filename += f"_{post_fix}"
                model_filename += ".pth"
                torch.save(
                    model.state_dict(),
                    os.path.join(path_save_model, model_filename),
                )
    logger.info(f"Training Finished !, Best Accuracy: {val_acc_max:.6f} --At epoch: {best_epoch} --Total_time: {time.time()-total_time:.2f}")
    time_tmp  = time.time()-total_time
    return val_acc_max, best_epoch,  dices_liver,  loss_epochs, trains_epoch, ious_liver, precisions_liver, recalls_liver, time_tmp

def trainer_stage2(model, train_loader, val_loader, optimizer, loss_func, acc_func, scheduler, batch_size, max_epochs, start_epoch=1, val_every = 1, logger=None, path_save_model=None, save_model=True, post_fix=None):
    val_acc_max, best_epoch = 0.0, 0
    total_time = time.time()
    dices_tumor, loss_epochs, trains_epoch = [], [], []
    ious_tumor = []
    precisions_tumor = []
    recalls_tumor = []

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
            val_acc , val_ious, val_precisions, val_recalls= val_epoch_stage2(model, val_loader, epoch, acc_func, max_epochs, logger)

            val_dice_tumor = val_acc

            val_iou_tumor = val_ious

            val_precision_tumor = val_precisions

            val_recall_tumor = val_recalls
            logger.info(f"\n{'*' * 20}Epoch Summary{'*' * 20}")
            logger.info(f"Final validation stats {epoch}/{max_epochs},   Dice_Tumor: {val_dice_tumor:.6f} , time {time.time() - epoch_time:.2f}s")
            
            dices_tumor.append(val_dice_tumor)

            ious_tumor.append(val_iou_tumor)

            precisions_tumor.append(val_precision_tumor)

            recalls_tumor.append(val_recall_tumor)
            if val_dice_tumor > val_acc_max:
                print("New best ({:.6f} --> {:.6f}). At epoch {}".format(val_acc_max, val_dice_tumor, epoch))
                logger.info(f"New best ({val_acc_max:.6f} --> {val_dice_tumor:.6f}). At epoch {epoch}. Time consuming: {time.time()-total_time:.2f}")
                val_acc_max = val_dice_tumor
                best_epoch = epoch
                if save_model:
                    model_filename = f"best_metric_model_{model.__class__.__name__}"
                    if post_fix is not None:
                        model_filename += f"_{post_fix}"
                    model_filename += ".pth"
                    torch.save(
                        model.state_dict(),
                        os.path.join(path_save_model, model_filename),
                    )

            torch.cuda.empty_cache()
        if epoch % 10 == 0 or epoch == max_epochs or epoch == 1:
            logger.info(f"Epoch {epoch}/{max_epochs} ---[loss: {train_loss:.4f}] ---[val_dice: {val_dice_tumor:.6f}] ---[time {time.time() - epoch_time:.2f}s]")
            # Save the model every 10 epochs
            if save_model:
                model_filename = f"model_{model.__class__.__name__}_epochs_{epoch}"
                if post_fix is not None:
                    model_filename += f"_{post_fix}"
                model_filename += ".pth"
                torch.save(
                    model.state_dict(),
                    os.path.join(path_save_model, model_filename),
                )
    logger.info(f"Training Finished !, Best Accuracy: {val_acc_max:.6f} --At epoch: {best_epoch} --Total_time: {time.time()-total_time:.2f}")
    time_tmp  = time.time()-total_time
    return val_acc_max, best_epoch,  dices_tumor,  loss_epochs, trains_epoch, ious_tumor, precisions_tumor, recalls_tumor, time_tmp