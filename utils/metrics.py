import torch
from torch import optim
import torch.nn as nn
from monai.losses import  FocalLoss, TverskyLoss
from monai.metrics import DiceMetric

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
            return  1 - dice_score.mean(dim=1)


    def forward(self, inputs, targets):
        '''
        calculate dice loss for binary segmentation
        '''
        dice_loss = self.dice_coefficient(inputs, targets, metric_mode=False)

        bce_loss = self.bce_loss(inputs, targets)

        final_loss = 0.7 * dice_loss + 0.3 * ce_loss
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
            return 1 - dice_score

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
        dice_loss = self.dice_coefficient(inputs, targets, metric_mode=False).mean()
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
    def __init__(self, alpha=0.7, beta=0.3):
        super(TverskyLossWSigmoid, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.tversky_loss = TverskyLoss(include_background=False, alpha=self.alpha, beta=self.beta, sigmoid=True)

    def forward(self, inputs, targets):
        bce_loss = self.bce_loss(inputs, targets)
        tversky_loss = self.tversky_loss(inputs, targets)
        final_loss = 0.7 * tversky_loss + 0.3 * bce_loss
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

class IoUMetric():
    def __init__(self, num_classes = 3, eps =1e-6, ignore_background=True):
        self.num_classes = num_classes
        self.eps = eps
        self.ignore_background = ignore_background

    def __call__(self, y_pred, y_true):
        """
        Compute IoU for each class.
        Args:
            y_pred: predicted labels (B, C, D, H, W)
            y_true: ground truth labels (B, C, D, H, W)
        """
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

class PrecisionMetric():
    def __init__(self, num_classes = 3, eps =1e-6, ignore_background=True):
        self.num_classes = num_classes
        self.eps = eps
        self.ignore_background = ignore_background

    def __call__(self, y_pred, y_true):
        """
        Compute Precision for each class.
        Args:
            y_pred: predicted labels (B, C, D, H, W)
            y_true: ground truth labels (B, C, D, H, W)
        """
        y_pred = torch.argmax(y_pred, dim=1)
        y_true = y_true.squeeze(1)

        precisions = []
        for i in range(self.num_classes):
            if self.ignore_background and i == 0:
                continue

            pred = (y_pred == i)
            true = (y_true == i)

            true_positive = (pred & true).sum().float()
            false_positive = (pred & ~true).sum().float()

            precision = (true_positive + self.eps) / (true_positive + false_positive + self.eps)
            precisions.append(precision.item())

        return precisions

class RecallMetric():
    def __init__(self, num_classes = 3, eps =1e-6, ignore_background=True):
        self.num_classes = num_classes
        self.eps = eps
        self.ignore_background = ignore_background

    def __call__(self, y_pred, y_true):
        """
        Compute Recall for each class.
        Args:
            y_pred: predicted labels (B, C, D, H, W)
            y_true: ground truth labels (B, C, D, H, W)
        """
        y_pred = torch.argmax(y_pred, dim=1)
        y_true = y_true.squeeze(1)

        recalls = []
        for i in range(self.num_classes):
            if self.ignore_background and i == 0:
                continue

            pred = (y_pred == i)
            true = (y_true == i)

            true_positive = (pred & true).sum().float()
            false_negative = (~pred & true).sum().float()

            recall = (true_positive + self.eps) / (true_positive + false_negative + self.eps)
            recalls.append(recall.item())
        
        return recalls