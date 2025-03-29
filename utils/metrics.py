import torch
from torch import optim
import torch.nn as nn

class EDiceLoss(nn.Module):
    def __init__(self):
        super(EDiceLoss, self).__init__()
        self.labels = ["Liver", "Tumor"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def binary_dice(self, inputs, targets, label_index, metric_mode=False):
        '''
        calculate dice loss for binary segmentation
        '''
        smooth = 1e-5
        inputs = torch.sigmoid(inputs)

        if metric_mode:
            inputs = (inputs > 0.5).float()

            if targets.sum() == 0:
                print(f"No {self.labels[label_index]} for this patient")
                if inputs.sum() == 0:
                    return torch.tensor(1., device=self.device)
                else:
                    return torch.tensor(0., device=self.device)

        intersection = torch.sum(inputs * targets)
        if metric_mode:
            dice = (2 * intersection) / ((inputs.sum() + targets.sum()) * 1.0)
        else:
            dice = (2 * intersection + smooth) / (inputs.pow(2).sum() + targets.pow(2).sum() + smooth)

        return dice if metric_mode else (1 - dice)

    def forward(self, inputs, targets):
        '''
        calculate dice loss for multi-class segmentation
        '''
        dice = 0    
        ce = 0
        BCE_L = torch.nn.BCELoss()

        for i in range(target.size(1)):
            dice += self.binary_dice(inputs[:, i, ...], targets[:, i, ...], i)
            ce += BCE_L(torch.sigmoid(inputs[:, i, ...]), target[:, i, ...])
        
        final_dice = ( 0.7 * dice + 0.3 * ce) / target.size(1)
        return final_loss

    def metric(self, inputs, targets):
        dices = []
        for j in range(targets.size(0)):
            dice = []
            for i in range(targets.size(1)):
                dice.append(self.binary_dice(inputs[j, i], targets[j, i], i, True))
            dices.append(dice)
        return dices

class EDiceLoss_Val(nn.Module):
    def __init__(self):
        super(EDiceLoss_Val, self).__init__()
        self.labels = ["Liver", "Tumor"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def binary_dice(self, inputs, targets, label_index, metric_mode=False):
        smooth = 1e-5
        inputs = torch.sigmoid(inputs)

        if metric_mode:
            inputs = (inputs > 0.5).float()

            if targets.sum() == 0:
                print(f"No {self.labels[label_index]} for this patient")
                if inputs.sum() == 0:
                    return torch.tensor(1., device=self.device)
                else:
                    return torch.tensor(0., device=self.device)

        intersection = torch.sum(inputs * targets)
        if metric_mode:
            dice = (2 * intersection) / ((inputs.sum() + targets.sum()) * 1.0)
        else:
            dice = (2 * intersection + smooth) / (inputs.pow(2).sum() + targets.pow(2).sum() + smooth)

        return dice if metric_mode else (1 - dice)
    
    def forward(self, inputs, targets):
        dice = 0

        for i in range(targets.size(1)):  
            dice += self.binary_dice(inputs[:, i, ...], targets[:, i, ...], i)
        final_dice = dice / targets.size(1)  

        return final_dice

    def metric(self, inputs, targets):
        dices = []

        for j in range(targets.size(0)):
            dice = []
            for i in range(targets.size(1)):
                dice.append(self.binary_dice(inputs[j, i], targets[j, i], i, True))
            dices.append(dice)

        return dices

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