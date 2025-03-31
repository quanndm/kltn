import torch
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    EnsureType
)


post_trans = Compose(
    [EnsureType(), Activations(softmax=True), AsDiscrete(argmax=True)]
)

post_softmax = Activations(softmax=True)

post_pred = AsDiscrete(argmax=True)