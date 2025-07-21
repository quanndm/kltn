from .unet3d  import UNet3D, UNet3DPretrained
from .unet3d_resnextcot import CoTNeXtUNet, UNet3DWResNeXtCoTPretrained
from .unet2d_resnextcot import UNet2DWResNeXtCoT
from .unet3d_resnextcot_mcb import MSC_CoTNeXtUNet
from .unet2d_resnextcot_mcb import MSC_CoTNeXtUNet2D
from .unet3d_resnext_ms_cot import MSCoTNeXtUNet
from .unet2d_resnext_ms_cot import MSCoTNeXtUNet2D

class ModelFactory:
    _model = {
        "unet3d": UNet3D,
        "unet3d_resnextcot": CoTNeXtUNet,
        "unet2d_resnextcot": UNet2DWResNeXtCoT,
        "unet3d_resnext_ms_cot": MSCoTNeXtUNet,
        "unet2d_resnext_ms_cot": MSCoTNeXtUNet2D,
        "unet3d_resnextcot_mcb":MSC_CoTNeXtUNet,
        "unet2d_resnextcot_mcb": MSC_CoTNeXtUNet2D,
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
        
        # if pretrained:
        #     return ModelFactory._model_pretrained[model_name](in_channels, n_classes, n_channels)
            
        return ModelFactory._model[model_name](in_channels, n_classes, n_channels)