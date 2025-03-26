import torch
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    EnsureType
)

post_trans = Compose(
    [EnsureType(), Activations(sigmoid=True), AsDiscrete(argmax=False, threshold=0.5)]
)

post_sigmoid = Activations(sigmoid=True)
post_pred = AsDiscrete(argmax=False, threshold=0.5)