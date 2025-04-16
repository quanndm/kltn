import numpy.random as random

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """ 
    LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    reference: https://github.com/facebookresearch/ConvNeXt-V2/blob/main/models/utils.py
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None] # add None to match the shape of 3d image
            return x


class GRN(nn.Module):
    """
    GRN (Global Response Normalization) layer
    reference: https://github.com/facebookresearch/ConvNeXt-V2/blob/main/models/utils.py
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1, 1))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(2,3, 4), keepdim=True)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class DropPath(nn.Module):
    """ DropPath layer (Stochastic Depth)

    reference: https://github.com/huggingface/pytorch-image-models/blob/e44f14d7d2f557b9f3add82ee4f1ed2beefbb30d/timm/layers/drop.py#L170
    """
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob 

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1) # (batch_size, 1, 1, 1, 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        random_tensor.floor_()
        
        if self.drop_prob > 0.0:
            random_tensor.div_(keep_prob)
        return x * random_tensor