import torch
from torch import optim
import torch.nn as nn
from monai.losses import  FocalLoss, TverskyLoss
from monai.metrics import DiceMetric
import torch.nn.functional as F

class DiceLossWSigmoid(nn.Module):
    def __init__(self):
        super(DiceLossWSigmoid, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def dice_coefficient(self, inputs, targets,  metric_mode=False, smooth=1e-5):
        '''
        calculate dice loss for binary segmentation
        args:
            inputs: shape (N, 1, D, H, W), predictions - logits
            targets: shape (N, 1, D, H, W), ground truth 
            metric_mode: if True, return dice score for each class
            smooth: smoothing factor to avoid division by zero
        returns:
            dice_loss: dice loss for binary segmentation
        '''

        inputs = torch.sigmoid(inputs)
        inputs = (inputs > 0.5).float()

        intersection = torch.sum(inputs * targets, dim=(2, 3, 4))
        dice_score = (2 * intersection + smooth) / (inputs.sum(dim=(2, 3, 4)) + targets.sum(dim=(2, 3, 4)) + smooth)

        if metric_mode:
            return dice_score.mean(dim=1) 
        else:
            return  1 - dice_score.mean()


    def forward(self, inputs, targets):
        '''
        calculate dice loss for binary segmentation
        '''
        dice_loss = self.dice_coefficient(inputs, targets, metric_mode=False)

        bce_loss = self.bce_loss(inputs, targets.float())

        final_loss = 0.7 * dice_loss + 0.3 * bce_loss
        return final_loss

    def metric(self, inputs, targets):
        dice_score = self.dice_coefficient(inputs, targets, metric_mode=True) 
        return dice_score

class DiceLossWSoftmax(nn.Module):
    def __init__(self):
        super(DiceLossWSoftmax, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()

    def dice_coefficient(self, inputs, targets, metric_mode=False):
        """
        calculate dice loss for multi-class segmentation
        args:
            inputs: shape (N, C, D, H, W), predictions - logits
            targets: shape (N, C, D, H, W), ground truth 
            metric_mode: if True, return dice score for each class
        """
        smooth = torch.finfo(torch.float32).eps  
        inputs = torch.softmax(inputs, dim=1)

        # Convert targets to one-hot encoding
        targets = targets.squeeze(1) # Remove channel dimension
        targets = torch.nn.functional.one_hot(targets.long() , num_classes=inputs.shape[1])
        targets = targets.permute(0, 4, 1, 2, 3)  # Change shape to (N, C, D, H, W)

        intersection = torch.sum(inputs * targets, dim=(2, 3, 4))  # (N, C)
        dice_score = (2 * intersection + smooth) / (inputs.sum(dim=(2, 3, 4)) + targets.sum(dim=(2, 3, 4)) + smooth)

        if metric_mode:
            return dice_score
        else:
            return 1 - dice_score.mean()

    def focal_loss(self, inputs, targets):
        """
        calculate focal loss for multi-class
        args:
            inputs: shape (N, C, D, H, W), predictions - logits
            targets: shape (N, C, D, H, W), ground truth

        """

        targets = targets.squeeze(1) # Remove channel dimension
        targets = torch.nn.functional.one_hot(targets.long() , num_classes=inputs.shape[1])
        targets = targets.permute(0, 4, 1, 2, 3).float()  # Change shape to (N, C, D, H, W)

        self.fc_loss = FocalLoss(gamma=2.0, alpha=0.25, use_softmax=True)
        return self.fc_loss(inputs, targets)

    def forward(self, inputs, targets):
        """
        calculate dice loss for multi-class segmentation
        """
        dice_loss = self.dice_coefficient(inputs, targets, metric_mode=False)
        # focal_loss = self.focal_loss(inputs, targets)
        
        targets = targets.argmax(dim=1)
        ce_loss = self.ce_loss(inputs, targets)
        
        # final_loss = 0.7 * dice_loss + 0.3 * ce_loss + 0.6 * focal_loss
        final_loss = 0.7 * dice_loss + 0.3 * ce_loss 
        return final_loss
    
    def metric(self, inputs, targets):
        dice_score = self.dice_coefficient(inputs, targets, metric_mode=True) 
        return dice_score

class TverskyLossWSigmoid(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma = 2.0, alpha_fc = 0.75, use_fc=False, weight_tversky=0.7, weight_bce_or_fc=0.3):
        super(TverskyLossWSigmoid, self).__init__()
        self.alpha = alpha
        self.beta = beta
        # self.bce_loss = nn.BCEWithLogitsLoss()
        self.tversky_loss = TverskyLoss( alpha=self.alpha, beta=self.beta, sigmoid=True)
        self.fc_loss = FocalLoss(gamma=gamma, alpha=alpha_fc)
        self.use_fc = use_fc    
        self.weight_tversky = weight_tversky
        self.weight_bce_or_fc = weight_bce_or_fc
    def forward(self, inputs, targets):
        tversky_loss = self.tversky_loss(inputs, targets)

        if self.use_fc:
            focal_loss = self.fc_loss(inputs, targets)
            final_loss = self.weight_tversky * tversky_loss + self.weight_bce_or_fc * focal_loss
        else:
            # bce_loss = self.bce_loss(inputs, targets.float())
            bce_loss = F.binary_cross_entropy_with_logits(inputs, targets.float())
            final_loss = self.weight_tversky * tversky_loss + self.weight_bce_or_fc * bce_loss
        return final_loss
        

class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    @staticmethod
    def _get_batch_fmtstr(num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

class IoUMetric:
    def __init__(self, num_classes=1, eps=1e-6, ignore_background=True, threshold=0.5):
        self.num_classes = num_classes
        self.eps = eps
        self.ignore_background = ignore_background
        self.threshold = threshold

    def __call__(self, y_pred, y_true):
        if y_true.ndim == 4 and y_true.shape[1] != 1:
            y_true = y_true.unsqueeze(1)

        if y_pred.shape[1] == 1:
            # Binary segmentation
            if y_pred.max() > 1 or y_pred.min() < 0:
                y_pred = torch.sigmoid(y_pred)
                # Apply threshold to get binary predictions
                y_pred = (y_pred > self.threshold).float()
            
            y_pred = y_pred.bool().squeeze()
            y_true = y_true.bool().squeeze()


            if y_pred.shape[0] == y_pred.shape[1] == y_pred.shape[2]: # 3D
                dims = tuple(range( y_pred.ndim))
            else: 
                dims = tuple(range(1, y_pred.ndim))

            true_object_sum = y_true.sum(dim=dims).float()
            pred_object_sum = y_pred.sum(dim=dims).float()

            intersection = (y_pred & y_true).sum(dim = dims).float()

            valid_mask = (true_object_sum > 0) 

            if not valid_mask.any():
                # If no valid mask, return NaN
                return [float('nan')]

            union = true_object_sum + pred_object_sum - intersection
            iou = (intersection + self.eps) / (union + self.eps)

            valid_iou = iou[valid_mask]
            mean_iou = valid_iou.mean().item()

            return [mean_iou]
        else:
            # Multi-class segmentation
            y_pred = torch.argmax(y_pred, dim=1)
            y_true = y_true.squeeze(1)

            ious = []
            for i in range(self.num_classes):
                if self.ignore_background and i == 0:
                    continue

                pred = (y_pred == i)
                true = (y_true == i)

                intersection = (pred & true).sum().float()
                union = (pred | true).sum().float()

                iou = (intersection + self.eps) / (union + self.eps)
                ious.append(iou.item())
            return ious


class PrecisionMetric:
    def __init__(self, num_classes=1, eps=1e-6, ignore_background=True, threshold=0.5):
        self.num_classes = num_classes
        self.eps = eps
        self.ignore_background = ignore_background
        self.threshold = threshold

    def __call__(self, y_pred, y_true):
        if y_true.ndim == 4 and y_true.shape[1] != 1:
            y_true = y_true.unsqueeze(1)

        if y_pred.shape[1] == 1:
            if y_pred.max() > 1 or y_pred.min() < 0:
                y_pred = torch.sigmoid(y_pred)
                y_pred = (y_pred > self.threshold).float()
            y_pred = y_pred.bool().squeeze()
            y_true = y_true.bool().squeeze()

            if y_pred.shape[0] == y_pred.shape[1] == y_pred.shape[2]: # 3D
                dims = tuple(range( y_pred.ndim))
            else: # 2D
                dims = tuple(range(1, y_pred.ndim))
            tp = (y_pred & y_true).sum(dim=dims).float()
            fp = (y_pred & (~y_true)).sum(dim=dims).float()

            pred_object_sum = tp + fp
            valid_mask = (pred_object_sum > 0)

            if not valid_mask.any():
                # If no valid mask, return NaN
                return [float('nan')]

            precision = (tp + self.eps) / ( pred_object_sum + self.eps)

            mean_precision = precision.mean().item()
            return [mean_precision]
        else:
            y_pred = torch.argmax(y_pred, dim=1)
            y_true = y_true.squeeze(1)

            precisions = []
            for i in range(self.num_classes):
                if self.ignore_background and i == 0:
                    continue

                pred = (y_pred == i)
                true = (y_true == i)

                tp = (pred & true).sum().float()
                fp = (pred & ~true).sum().float()

                precision = (tp + self.eps) / (tp + fp + self.eps)
                precisions.append(precision.item())
            return precisions


class RecallMetric:
    def __init__(self, num_classes=1, eps=1e-6, ignore_background=True, threshold=0.5):
        self.num_classes = num_classes
        self.eps = eps
        self.ignore_background = ignore_background
        self.threshold = threshold

    def __call__(self, y_pred, y_true):
        if y_true.ndim == 4 and y_true.shape[1] != 1:
            y_true = y_true.unsqueeze(1)

        if y_pred.shape[1] == 1:
            if y_pred.max() > 1 or y_pred.min() < 0:
                y_pred = torch.sigmoid(y_pred)
                y_pred = (y_pred > self.threshold).float()

            y_pred = y_pred.bool().squeeze()
            y_true = y_true.bool().squeeze()

            if y_pred.shape[0] == y_pred.shape[1] == y_pred.shape[2]: # 3D
                dims = tuple(range( y_pred.ndim))
            else: # 2D
                dims = tuple(range(1, y_pred.ndim))
            tp = (y_pred & y_true).sum(dim = dims).float()
            fn = ((~y_pred) & y_true).sum(dim = dims).float()

            true_object_sum = tp + fn
            valid_mask = (true_object_sum > 0)

            if not valid_mask.any():
                # If no valid mask, return NaN
                return [float('nan')]

            
            recall = (tp + self.eps) / (true_object_sum + self.eps)

            valid_recall = recall[valid_mask]
            mean_recall = valid_recall.mean().item()
            return [mean_recall]
        else:
            y_pred = torch.argmax(y_pred, dim=1)
            y_true = y_true.squeeze(1)

            recalls = []
            for i in range(self.num_classes):
                if self.ignore_background and i == 0:
                    continue

                pred = (y_pred == i)
                true = (y_true == i)

                tp = (pred & true).sum().float()
                fn = (~pred & true).sum().float()

                recall = (tp + self.eps) / (tp + fn + self.eps)
                recalls.append(recall.item())
            return recalls