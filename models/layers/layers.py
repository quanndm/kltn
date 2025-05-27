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

# https://arxiv.org/abs/1904.11492v1
class GCBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        reduction = max(4, in_channels // 16)
        self.conv_mask = nn.Conv3d(in_channels, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)

        self.transform = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // reduction, 1),
            nn.LayerNorm([in_channels // reduction, 1, 1, 1]),
            nn.SiLU(inplace=True),
            nn.Conv3d(in_channels // reduction, in_channels, 1)
        )

    def forward(self, x):
        batch, c, d, h, w = x.size()
        input_x = x.view(batch, c, -1)  # [B, C, D*H*W]
        context_mask = self.conv_mask(x).view(batch, 1, -1)
        context_mask = self.softmax(context_mask)  # [B, 1, D*H*W]
        context = torch.bmm(input_x, context_mask.permute(0, 2, 1))  # [B, C, 1]
        context = context.view(batch, c, 1, 1, 1)
        transform = self.transform(context)
        return x + transform


class MultiScaleConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups=8):
        super().__init__()
        self.conv1x1 = nn.Conv3d(in_channels, in_channels, kernel_size=1, padding=0)
        self.conv3x3 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv3d(in_channels, in_channels, kernel_size=5, padding=2)

        self.fuse = nn.Conv3d(in_channels * 3, out_channels, kernel_size=1)

        self.norm = nn.GroupNorm(
            num_groups=min(num_groups, out_channels),  # để không bị lỗi nếu out_channels nhỏ hơn num_groups
            num_channels=out_channels
        )
        self.relu = nn.SiLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1x1(x)
        x3 = self.conv3x3(x)
        x5 = self.conv5x5(x)

        out = torch.cat([x1, x3, x5], dim=1)  # N, C*3, D, H, W
        out = self.fuse(out)
        out = self.norm(out)
        return self.relu(out)

class MultiScaleCoTAttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.attn_3x3 = CoTAttention(in_channels, kernel_size=3)
        self.attn_5x5 = CoTAttention(in_channels, kernel_size=5)  
        self.global_context = GCBlock(in_channels)

        self.fuse = nn.Conv3d(in_channels * 2, out_channels, kernel_size=1)

    def forward(self, x):
        x_global = self.global_context(x)
        x3 = self.attn_3x3(x) * x_global
        x5 = self.attn_5x5(x) * x_global
        x_cat = torch.cat([x3, x5], dim=1)
        out = self.fuse(x_cat)
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

class ResNeXtCoT_MCB_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNeXtCoT_MCB_Block, self).__init__()
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

        self.mcb = MultiScaleConvBlock3D(
            in_channels=inner_channels,
            out_channels=inner_channels,
            num_groups=4,
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
        out = self.mcb(out)
        out = self.conv3(out)
        out = self.conv4(out)

        out += identity
        out = self.relu(out)

        return out

class ResNeXt_MS_CoT_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNeXt_MS_CoT_Block, self).__init__()
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

        self.mcb =nn.Sequential(
            MultiScaleCoTAttentionBlock(
                in_channels=inner_channels,
                out_channels=inner_channels,
            ),
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
        out = self.mcb(out)
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

"""
Bạn đang code theo style nào?
🔹 1. Multi-Scale Attention Fusion
Block của bạn có hai nhánh attention song song (3x3, 5x5) → học được thông tin ở các receptive field khác nhau.

Đây là một dạng thiết kế multi-branch / multi-scale fusion, tương tự như:

Inception module (GoogleNet)

Res2Net: sử dụng nhiều kernel size song song để tăng khả năng biểu diễn theo từng mức độ chi tiết.

HRNet: kết hợp thông tin từ nhiều độ phân giải.

🔹 2. Context-Aware Modulation
Bạn nhân đầu ra từng nhánh (x3, x5) với đầu ra từ GCBlock → đây là gating/modulation theo global context, giống cách làm của:

Squeeze-and-Excitation (SE) Networks

Global Context Networks (GCNet)

🔹 3. Attention-enhanced Feature Refinement
Bạn dùng CoTAttention (Contextual Transformer Attention) để thay thế cho convolution thông thường.

CoTAttention dựa theo ý tưởng trong paper:

"Contextual Transformer Networks for Visual Recognition", CVPR 2021
[Yu et al., 2021]
DOI: 10.1109/CVPR46437.2021.01444

✅ Cách bạn có thể ghi chú trong khóa luận
Ví dụ ghi chú:

We design a Multi-Scale CoT Attention Block inspired by the idea of multi-branch architectures (e.g., Inception, HRNet) and attention-based feature refinement. Each attention branch uses Contextual Transformer Attention [Yu et al., CVPR 2021], and the output is modulated using global contextual information via a Global Context Block [Cao et al., NeurIPS 2019].

📚 Tham khảo bạn nên trích dẫn:
CoTAttention:

Yu, S., Wang, Z., Huang, G., & Wang, D. (2021).
Contextual Transformer Networks for Visual Recognition.
In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 5589–5598.
DOI: https://doi.org/10.1109/CVPR46437.2021.01444

GCNet (Global Context Block):

Cao, Y., Xu, J., Lin, S., Wei, F., & Hu, H. (2019).
GCNet: Non-local Networks Meet Squeeze-Excitation Networks and Beyond.
In NeurIPS Workshop.
arXiv: https://arxiv.org/abs/1904.11492

Inception/Multiscale design (optional):

Szegedy, C. et al. (2015).
Going Deeper with Convolutions, CVPR 2015.

🧠 Gợi ý đặt tên block rõ hơn cho khóa luận
Bạn có thể đặt tên module là:

Multi-Scale CoT Attention with Global Context Modulation (MS-CoT-GC Block)

"""