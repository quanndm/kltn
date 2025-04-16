from .unet3d  import UNet3D, UNet3DPretrained
from .unet3d_cot import UNet3DWCoT, UNet3DWCoTPretrained
from .unet3d_cot_da import UNet3DWCoTDA, UNet3DWCoTDAPretrained
from .unet3d_resnextcot import UNet3DWResNeXtCoT, UNet3DWResNeXtCoTPretrained
from .unet3d_convnextv2cot import UNet3DWConvNeXtV2CoT, UNet3DWConvNeXtV2CoTPretrained
from .unet3d_resnextcot_da import UNet3DWResNeXtCoTDA, UNet3DWResNeXtCoTDAPretrained
from .unet3d_convnextv2cot_da import UNet3DWConvNeXtV2CoTDA, UNet3DWConvNeXtV2CoTDAPretrained

class ModelFactory:
    _model = {
        "unet3d": UNet3D,
        "unet3d_cot": UNet3DWCoT,
        "unet3d_resnextcot": UNet3DWResNeXtCoT,
        "unet3d_cot_da": UNet3DWCoTDA,
        "unet3d_convnextv2_cot": UNet3DWConvNeXtV2CoT,
        "unet3d_resnextcot_da": UNet3DWResNeXtCoTDA,
        "unet3d_convnextv2cot_da": UNet3DWConvNeXtV2CoTDA,
    }

    _model_pretrained ={
        "unet3d": UNet3DPretrained,
        "unet3d_cot": UNet3DWCoTPretrained,
        "unet3d_resnextcot": UNet3DWResNeXtCoTPretrained,
        "unet3d_convnextv2_cot": UNet3DWConvNeXtV2CoTPretrained,
        "unet3d_cot_da": UNet3DWCoTDAPretrained,
        "unet3d_resnextcot_da": UNet3DWResNeXtCoTDAPretrained,
        "unet3d_convnextv2cot_da": UNet3DWConvNeXtV2CoTDAPretrained,
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
            return ModelFactory._model_pretrained[model_name](in_channels, n_classes, n_channels)
            
        return ModelFactory._model[model_name](in_channels, n_classes, n_channels)