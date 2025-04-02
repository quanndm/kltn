import torch
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    EnsureType
)


post_trans = Compose(
    [EnsureType(), Activations(softmax=True),AsDiscrete(argmax=True, to_onehot=3)]
)
post_trans_v2 = Compose(
    [EnsureType(), Activations(softmax=True),AsDiscrete(argmax=True)]
)

post_softmax = Activations(softmax=True)

post_pred = AsDiscrete(argmax=True)
# post_trans = Compose(
#     [EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold=0.5)]
# )

# post_sigmoid = Activations(sigmoid=True)

# post_pred = AsDiscrete(threshold=0.5)