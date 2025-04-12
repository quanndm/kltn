import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.layers import DoubleConv, DoubleAttention, OutConv, ResNeXtCoTBlock
from monai.networks.nets import resnet50

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            ResNeXtCoTBlock(in_channels, out_channels)
        )

    def forward(self, x):
        return self.encoder(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, trilinear=True):
        super().__init__()
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = ResNeXtCoTBlock(in_channels, out_channels)
        self.da = DoubleAttention(in_channels // 2, in_channels // 4, in_channels // 4)
    def forward(self, inputs, skips):
        inputs = self.up(inputs)

        diffZ = skips.size()[2] - inputs.size()[2]
        diffY = skips.size()[3] - inputs.size()[3]
        diffX = skips.size()[4] - inputs.size()[4]
        inputs = F.pad(inputs, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2, diffZ // 2, diffZ - diffZ // 2])

        skips = self.da(skips)
        x = torch.cat([skips, inputs], dim=1)
        return self.conv(x)

class UNet3DWResNeXtCoTDA(nn.Module):
    def __init__(self, in_channels, n_classes, n_channels):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.n_channels = n_channels

        self.conv = DoubleConv(in_channels, n_channels)
        self.enc1 = Down(n_channels, 2 * n_channels)
        self.enc2 = Down(2 * n_channels, 4 * n_channels)
        self.enc3 = Down(4 * n_channels, 8 * n_channels)
        self.enc4 = Down(8 * n_channels, 8 * n_channels)

        self.dec1 = Up(16 * n_channels, 4 * n_channels)
        self.dec2 = Up(8 * n_channels, 2 * n_channels)
        self.dec3 = Up(4 * n_channels, n_channels)
        self.dec4 = Up(2 * n_channels, n_channels)
        self.out = OutConv(n_channels, n_classes)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.enc1(x1)
        x3 = self.enc2(x2)
        x4 = self.enc3(x3)
        x5 = self.enc4(x4)

        mask = self.dec1(x5, x4)
        mask = self.dec2(mask, x3)
        mask = self.dec3(mask, x2)
        mask = self.dec4(mask, x1)
        mask = self.out(mask)
        return mask

class UNet3DWResNeXtCoTDAPretrained(nn.Module):
    def __init__(self, in_channels, n_classes, n_channels):
        super(UNet3DPretrained, self).__init__()
        self.pretrained_model =  resnet50(spatial_dims=3, n_input_channels=1, pretrained=True, feed_forward=False , shortcut_type="B", bias_downsample=False)

        for name, param in self.pretrained_model.named_parameters():
            if 'layer3' in name or 'layer4' in name:
                param.requires_grad = True  
            else:
                param.requires_grad = False  

        self.conv1 = nn.Sequential(
            self.pretrained_model.conv1,
            self.pretrained_model.bn1,
            self.pretrained_model.act,
            self.pretrained_model.maxpool
        )
        self.layer1 = self.pretrained_model.layer1
        self.layer2 = self.pretrained_model.layer2
        self.layer3 = self.pretrained_model.layer3
        self.layer4 = self.pretrained_model.layer4

        self.reduce_channel1 = nn.Conv3d(64, n_channels, kernel_size=1, stride=1, padding=0)
        self.reduce_channel2 = nn.Conv3d(256, 2 * n_channels, kernel_size=1, stride=1, padding=0)
        self.reduce_channel3 = nn.Conv3d(512, 4 * n_channels, kernel_size=1, stride=1, padding=0)
        self.reduce_channel4 = nn.Conv3d(1024, 8 * n_channels, kernel_size=1, stride=1, padding=0)
        self.reduce_channel5 = nn.Conv3d(2048, 8 * n_channels, kernel_size=1, stride=1, padding=0)

        self.unet = UNet3DWResNeXtCoTDA(in_channels, n_classes, n_channels)

    def forward(self, x):
        x1 = self.reduce_channel1(self.conv1(x))
        x2 = self.reduce_channel2(self.layer1(x1))
        x3 = self.reduce_channel3(self.layer2(x2))
        x4 = self.reduce_channel4(self.layer3(x3))
        x5 = self.reduce_channel5(self.layer4(x4))

        mask = self.unet.dec1(x5, x4)
        mask = self.unet.dec2(mask, x3)
        mask = self.unet.dec3(mask, x2)
        mask = self.unet.dec4(mask, x1)
        mask = self.unet.out(mask)

        return mask