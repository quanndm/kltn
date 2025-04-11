import torch
import torch.nn as nn
from monai.networks.nets import DenseNet121

from .unet3d  import UNet3D
from .unet3d_cot import UNet3DWCoT
from .unet3d_cot_da import UNet3DWCoTDA
from .unet3d_resnextcot_da import UNet3DWResNeXtCoTDA
from .unet3d_convnextv2cot_da import UNet3DWConvNeXtV2CoTDA

class CombinedPretrainedModel(nn.Module):
    def __init__(self, in_channels, n_classes, n_channels, model):
        super(CombinedPretrainedModel, self).__init__()
        self.pretrained = DenseNet121(spatial_dims=3, in_channels=in_channels, out_channels=2, pretrained=True)
        self.model = model(in_channels, n_classes, n_channels)
        self.feature_extractor = self.pretrained.features

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.projector = nn.Conv3d(1024, in_channels, kernel_size=1)

    def forward(self, x):
        with torch.no_grad():
            features = self.feature_extractor(x)
        features = self.projector(features)

        return self.model(features)

class ModelFactory:
    _model = {
        "unet3d": UNet3D,
        "unet3d_cot": UNet3DWCoT,
        "unet3d_cot_da": UNet3DWCoTDA,
        "unet3d_resnextcot_da": UNet3DWResNeXtCoTDA,
        "unet3d_convnextv2cot_da": UNet3DWConvNeXtV2CoTDA,
    }

    @staticmethod
    def get_model(model_name, in_channels, n_classes, n_channels, pretrained=False):
        """
        Get the model class based on the model name.
        :param model_name: Name of the model to retrieve.
        :param in_channels: Number of input channels.
        :param n_classes: Number of output classes.
        :param n_channels: Number of channels in the model.
        :return: An instance of the specified model.
        """
        if model_name not in ModelFactory._model:
            raise ValueError(f"Model {model_name} not found!")
        
        if pretrained:
            return CombinedPretrainedModel(in_channels, n_classes, n_channels, ModelFactory._model[model_name])
        return ModelFactory._model[model_name](in_channels, n_classes, n_channels)