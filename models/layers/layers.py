import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels,  num_groups=8):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            # nn.ReLU(inplace=True),
            nn.SiLU(inplace=True),

            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            # nn.ReLU(inplace=True),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class DoubleConv2D(nn.Module):
    def __init__(self, in_channels, out_channels,  num_groups=8):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.SiLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class CoTAttention(nn.Module):
    def __init__(self, dim=512,kernel_size=3):
        super().__init__()
        self.dim=dim
        self.kernel_size=kernel_size

        self.key_embed=nn.Sequential(
            nn.Conv3d(dim,dim,kernel_size=kernel_size,padding=kernel_size//2,groups=4,bias=False),
            nn.BatchNorm3d(dim),
            # nn.ReLU()
            nn.SiLU(),
        )
        self.value_embed=nn.Sequential(
            nn.Conv3d(dim,dim,1,bias=False),
            nn.BatchNorm3d(dim)
        )

        factor=4
        self.attention_embed=nn.Sequential(
            nn.Conv3d(2*dim,2*dim//factor,1,bias=False),
            nn.BatchNorm3d(2*dim//factor),
            # nn.ReLU(),
            nn.SiLU(),
            nn.Conv3d(2*dim//factor,kernel_size*kernel_size*dim,1)
        )


    def forward(self, x):
        bs,c,h,w,d=x.shape
        k1=self.key_embed(x) #bs,c,h,w
        v=self.value_embed(x).view(bs,c,-1) #bs,c,h,w

        y=torch.cat([k1,x],dim=1) #bs,2c,h,w
        att=self.attention_embed(y) #bs,c*k*k,h,w
        att=att.reshape(bs,c,self.kernel_size*self.kernel_size,h,w,d)
        att=att.mean(2,keepdim=False).view(bs,c,-1) #bs,c,h*w
        k2=F.softmax(att,dim=-1)*v
        k2=k2.view(bs,c,h,w,d)
        out = k1 + k2
        return out

class CoTAttention2D(nn.Module):
    def __init__(self, dim=512,kernel_size=3):
        super().__init__()
        self.dim=dim
        self.kernel_size=kernel_size

        self.key_embed=nn.Sequential(
            nn.Conv2d(dim,dim,kernel_size=kernel_size,padding=kernel_size//2,groups=4,bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(),
        )
        self.value_embed=nn.Sequential(
            nn.Conv2d(dim,dim,1,bias=False),
            nn.BatchNorm2d(dim)
        )

        factor=4
        self.attention_embed=nn.Sequential(
            nn.Conv2d(2*dim,2*dim//factor,1,bias=False),
            nn.BatchNorm2d(2*dim//factor),
            nn.SiLU(),
            nn.Conv2d(2*dim//factor,kernel_size*kernel_size*dim,1)
        )


    def forward(self, x):
        bs,c,h,w=x.shape
        k1=self.key_embed(x) #bs,c,h,w
        v=self.value_embed(x).view(bs,c,-1) #bs,c,h,w

        y=torch.cat([k1,x],dim=1) #bs,2c,h,w
        att=self.attention_embed(y) #bs,c*k*k,h,w
        att=att.reshape(bs,c,self.kernel_size*self.kernel_size,h,w)
        att=att.mean(2,keepdim=False).view(bs,c,-1) #bs,c,h*w
        k2=F.softmax(att,dim=-1)*v
        k2=k2.view(bs,c,h,w)
        out = k1 + k2
        return out

class DoubleConvDownWCoT(nn.Module):
    def __init__(self, in_channels, out_channels,  num_groups=8):
        super().__init__()
        self.double_conv = nn.Sequential(
            CoTAttention(in_channels, 3),
            nn.GroupNorm(num_groups=num_groups, num_channels=in_channels),
            # nn.ReLU(inplace=True),
            nn.SiLU(inplace=True),

            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            # nn.ReLU(inplace=True)
            nn.SiLU(inplace=True),
        )
        
    def forward(self, x):
        return self.double_conv(x)


class DoubleConvUpWCoT(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups=8):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            # nn.ReLU(inplace=True),
            nn.SiLU(inplace=True),

            CoTAttention(out_channels, 3),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            # nn.ReLU(inplace=True)
            nn.SiLU(inplace=True),
          )

    def forward(self,x):
        return self.double_conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size = 1)

    def forward(self, x):
        return self.conv(x)

class OutConv2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 1)

    def forward(self, x):
        return self.conv(x)

class DoubleAttention(nn.Module):
    def __init__(self, in_channels, c_m, c_n):
        super(DoubleAttention, self).__init__()
        self.in_channels = in_channels
        self.c_m = c_m  
        self.c_n = c_n  

        # step 1: Spatial Attention
        self.convA = nn.Conv3d(in_channels, c_m, kernel_size=1) 
        self.convB = nn.Conv3d(in_channels, c_n, kernel_size=1)  
        self.convV = nn.Conv3d(in_channels, c_n, kernel_size=1)  

        # step 2: Channel Attention
        self.convC = nn.Conv3d(c_m, in_channels, kernel_size=1)  # Adjust output channel

        # self.norm = nn.LayerNorm([in_channels, 1, 1, 1])
    def forward(self, x):
        batch_size, _, d, h, w = x.shape


        A = self.convA(x)  # (B, c_m, D, H, W)
        B = self.convB(x)  # (B, c_n, D, H, W)
        V = self.convV(x)  # (B, c_n, D, H, W)

        A = A.view(batch_size, self.c_m,  d * h * w)  # (B, c_m, D*H*W)
        # A = F.softmax(A, dim=-1)

        B = B.view(batch_size, self.c_n,  d * h * w)  # (B, c_n, D*H*W)
        B = F.softmax(B, dim=-1)

        V = V.view(batch_size, self.c_n,  d * h * w)  # (B, c_n, D*H*W)
        V = F.softmax(V, dim=-1)

        attn_map = torch.bmm(A, B.permute(0, 2, 1))  # (B, c_n, c_m)
        attn_out = attn_map.matmul(V)  # (B, c_n, D*H*W)

        attn_out = attn_out.view(batch_size, self.c_m, d, h, w)  # (B, c_m, D, H, W)
        attn_out = self.convC(attn_out)  # (B, in_channels, D, H, W)

        # out = self.norm(x + attn_out) 
        out = x + attn_out  # Residual connection
        return out


class ResNeXtCoTBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNeXtCoTBlock, self).__init__()
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.SiLU(inplace=True)
        inner_channels = out_channels // 2

        self.conv1 = nn.Sequential(

            nn.Conv3d(in_channels, inner_channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=4, num_channels=inner_channels),
            # nn.ReLU(inplace=True)
            nn.SiLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(inner_channels, inner_channels, kernel_size=3, stride=1, padding=1, groups=4, bias=False),
            nn.GroupNorm(num_groups=4, num_channels=inner_channels),
            # nn.ReLU(inplace=True)
            nn.SiLU(inplace=True),
        )

        self.conv3 = nn.Sequential(
            CoTAttention(inner_channels, 3),
            nn.GroupNorm(num_groups=4, num_channels=inner_channels),
            # nn.ReLU(inplace=True)
            nn.SiLU(inplace=True),
        )

        self.conv4 = nn.Sequential(
            nn.Conv3d(inner_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=4, num_channels=out_channels)
        )

        self.residual = None
        if in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.GroupNorm(num_groups=4, num_channels=out_channels)
            )

    def forward(self, x):
        identity = x    
        if self.residual is not None:
            identity = self.residual(x)

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)

        out += identity
        out = self.relu(out)

        return out


class ResNeXtCoTBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNeXtCoTBlock2D, self).__init__()
        self.relu = nn.SiLU(inplace=True)
        inner_channels = out_channels // 2

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=4, num_channels=inner_channels),
            nn.SiLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels, kernel_size=3, stride=1, padding=1, groups=4, bias=False),
            nn.GroupNorm(num_groups=4, num_channels=inner_channels),
            nn.SiLU(inplace=True),
        )

        self.conv3 = nn.Sequential(
            CoTAttention2D(inner_channels, 3),
            nn.GroupNorm(num_groups=4, num_channels=inner_channels),
            nn.SiLU(inplace=True),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(inner_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=4, num_channels=out_channels)
        )

        self.residual = None
        if in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.GroupNorm(num_groups=4, num_channels=out_channels)
            )
        
    def forward(self, x):
        identity = x    
        if self.residual is not None:
            identity = self.residual(x)

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)

        out += identity
        out = self.relu(out)

        return out

class ChannelAttention3D(nn.Module):
    def __init__(self, in_planes, reduction=16):
        super(ChannelAttention3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(in_planes, in_planes // reduction, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv3d(in_planes // reduction, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        return self.sigmoid(avg_out)

class SpatialAttention3D(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention3D, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x_cat))

class BottleneckAttentionBlock3D(nn.Module):
    def __init__(self, channels):
        super(BottleneckAttentionBlock3D, self).__init__()
        self.residual_conv = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=channels)
        )
        self.relu = nn.SiLU(inplace=True)

        self.ca = ChannelAttention3D(channels)
        self.sa = SpatialAttention3D()

    def forward(self, x):
        res = self.residual_conv(x)
        res = self.ca(res) * res
        res = self.sa(res) * res
        out = x + res
        return self.relu(out)


# Simplified ScaleDotProduct (no einsum, no generality)
class ScaleDotProduct(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, scale):
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, N, N]
        attn = self.softmax(attn)
        out = torch.matmul(attn, v)  # [B, H, N, C]
        return out

# Simplified DepthwiseProjection for (B, N, C) tensors using Conv3D
class DepthwiseProjection(nn.Module):
    def __init__(self, in_features, out_features, groups):
        super().__init__()
        self.proj = nn.Conv3d(in_features, out_features, kernel_size=3, padding=1, groups=groups)

    def forward(self, x):
        B, N, C = x.shape
        D = int(N ** (1/3))
        x = x.transpose(1, 2).view(B, C, D, D, D)  # [B, C, D, H, W]
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # [B, N, C]
        return x

# Cross-Channel Attention for 3D
class CrossChannelAttention3D(nn.Module):
    def __init__(self, in_features, out_features, n_heads=1):
        super().__init__()
        self.n_heads = n_heads
        self.q_map = DepthwiseProjection(out_features, out_features, groups=out_features)
        self.k_map = DepthwiseProjection(in_features, in_features, groups=in_features)
        self.v_map = DepthwiseProjection(in_features, in_features, groups=in_features)
        self.projection = DepthwiseProjection(out_features, out_features, groups=out_features)
        self.sdp = ScaleDotProduct()

    def forward(self, x_dec, x_enc):
        q = self.q_map(x_dec)
        k = self.k_map(x_enc)
        v = self.v_map(x_enc)
        B, N, Cq = q.shape
        C = k.shape[2]
        scale = C ** -0.5
        q = q.reshape(B, N, self.n_heads, Cq // self.n_heads).permute(0, 2, 1, 3)
        k = k.reshape(B, N, self.n_heads, C // self.n_heads).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.n_heads, C // self.n_heads).permute(0, 2, 1, 3)
        att = self.sdp(q, k, v, scale).permute(0, 2, 1, 3).flatten(2)
        att = self.projection(att)
        D = int(att.shape[1] ** (1/3))
        return att.transpose(1, 2).view(B, -1, D, D, D)  # [B, C, D, H, W]

# Cross-Spatial Attention for 3D
class CrossSpatialAttention3D(nn.Module):
    def __init__(self, in_features, out_features, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.q_map = DepthwiseProjection(in_features, in_features, groups=in_features)
        self.k_map = DepthwiseProjection(in_features, in_features, groups=in_features)
        self.v_map = DepthwiseProjection(out_features, out_features, groups=out_features)
        self.projection = DepthwiseProjection(out_features, out_features, groups=out_features)
        self.sdp = ScaleDotProduct()

    def forward(self, x_dec, x_enc):
        q = self.q_map(x_dec)
        k = self.k_map(x_enc)
        v = self.v_map(x_enc)
        B, N, C = q.shape
        Cv = v.shape[2]
        scale = (C // self.n_heads) ** -0.5
        q = q.reshape(B, N, self.n_heads, C // self.n_heads).permute(0, 2, 1, 3)
        k = k.reshape(B, N, self.n_heads, C // self.n_heads).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.n_heads, Cv // self.n_heads).permute(0, 2, 1, 3)
        att = self.sdp(q, k, v, scale).transpose(1, 2).flatten(2)
        att = self.projection(att)
        D = int(att.shape[1] ** (1/3))
        return att.transpose(1, 2).view(B, -1, D, D, D)  # [B, C, D, H, W]

class DualCrossAttention3D(nn.Module):
    def __init__(self, 
                 in_features, 
                 out_features, 
                 channel_heads=4, 
                 spatial_heads=4, 
                 use_spatial=True, 
                 use_channel=True):
        super().__init__()
        self.use_channel = use_channel
        self.use_spatial = use_spatial

        if self.use_channel:
            self.channel_norm = nn.LayerNorm(in_features, eps=1e-6)
            self.channel_attn = CrossChannelAttention3D(
                in_features=in_features,
                out_features=out_features,
                n_heads=channel_heads
            )

        if self.use_spatial:
            self.spatial_norm = nn.LayerNorm(in_features, eps=1e-6)
            self.spatial_attn = CrossSpatialAttention3D(
                in_features=in_features,
                out_features=out_features,
                n_heads=spatial_heads
            )

    def forward(self, x):  # x: [B, C, D, H, W]
        B, C, D, H, W = x.shape
        N = D * H * W
        x_flat = x.view(B, C, N).transpose(1, 2)  # [B, N, C]

        if self.use_channel:
            x_norm = self.channel_norm(x_flat)
            x_flat = x_flat + self.channel_attn(x_norm, x_norm)
        if self.use_spatial:
            x_norm = self.spatial_norm(x_flat)
            x_flat = x_flat + self.spatial_attn(x_norm, x_norm)
        x_out = x_flat.transpose(1, 2).view(B, C, D, H, W)  # [B, C, D, H, W]
        return x_out