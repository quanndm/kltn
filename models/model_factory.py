from .unet3d  import UNet3D
from .unet3d_cot import UNet3DWCoT
from .unet3d_cot_da import UNet3DWCoTDA
from .unet3d_resnextcot_da import UNet3DWResNeXtCoTDA
from .unet3d_convnextv2cot_da import UNet3DWConvNeXtV2CoTDA

class ModelFactory:
    _model = {
        "unet3d": UNet3D,
        "unet3d_cot": UNet3DWCoT,
        "unet3d_cot_da": UNet3DWCoTDA,
        "unet3d_resnextcot_da": UNet3DWResNeXtCoTDA,
        "unet3d_convnextv2cot_da": UNet3DWConvNeXtV2CoTDA,
    }

    @staticmethod
    def get_model(model_name, in_channels, n_classes, n_channels):
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
        return ModelFactory._model[model_name](in_channels, n_classes, n_channels)