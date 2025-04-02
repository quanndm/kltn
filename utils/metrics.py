import torch
from torch import optim
import torch.nn as nn

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

    def dice_coefficient(self, inputs, targets, metric_mode=False, smooth=1e-5):
        """
        calculate dice loss for multi-class segmentation
        args:
            inputs: shape (N, C, D, H, W), predictions - logits
            targets: shape (N, C, D, H, W), ground truth 
            metric_mode: if True, return dice score for each class
            smooth: smoothing factor to avoid division by zero
        """
        inputs = torch.softmax(inputs, dim=1)

        # Convert targets to one-hot encoding
        targets = torch.nn.functional.one_hot(targets.long() , num_classes=inputs.shape[1])
        targets = targets.permute(0, 4, 1, 2, 3)  # Change shape to (N, C, D, H, W)

        intersection = torch.sum(inputs * targets, dim=(2, 3, 4))  # (N, C)
        dice_score = (2 * intersection + smooth) / (inputs.sum(dim=(2, 3, 4)) + targets.sum(dim=(2, 3, 4)) + smooth)

        if metric_mode:
            return dice_score
        else:
            return 1 - dice_score

    def forward(self, inputs, targets):
        """
        calculate dice loss for multi-class segmentation
        """
        dice_loss = self.dice_coefficient(inputs, targets, metric_mode=False).mean()
        
        # Convert targets
        targets = targets.argmax(dim=1)
        ce_loss = self.ce_loss(inputs, targets)

        final_loss = 0.7 * dice_loss + 0.3 * ce_loss
        return final_loss
    
    def metric(self, inputs, targets):
        dice_score = self.dice_coefficient(inputs, targets, metric_mode=True) 
        return dice_score
# class EDiceLoss_Val(nn.Module):
#     def __init__(self):
#         super(EDiceLoss_Val, self).__init__()
#         self.labels = ["Liver", "Tumor"]
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     def binary_dice(self, inputs, targets, metric_mode=False):
#         smooth = 1e-5
#         inputs = torch.softmax(inputs, dim=1)

#         intersection = torch.sum(inputs * targets, dim=(2, 3, 4))  # (N, C)
#         dice_score = (2 * intersection + smooth) / (inputs.sum(dim=(2, 3, 4)) + targets.sum(dim=(2, 3, 4)) + smooth)

#         if metric_mode:
#             return dice_score  
#         return 1 - dice_score.mean()
    
#     def forward(self, inputs, targets):

#         return self.multi_class_dice(inputs, targets, metric_mode=False)

#     def metric(self, inputs, targets):
#         return self.multi_class_dice(inputs, targets, metric_mode=True)

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