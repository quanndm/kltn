import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels,  num_groups=8):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.ReLU(inplace=True),

            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.ReLU(inplace=True),
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
            nn.ReLU()
        )
        self.value_embed=nn.Sequential(
            nn.Conv3d(dim,dim,1,bias=False),
            nn.BatchNorm3d(dim)
        )

        factor=4
        self.attention_embed=nn.Sequential(
            nn.Conv3d(2*dim,2*dim//factor,1,bias=False),
            nn.BatchNorm3d(2*dim//factor),
            nn.ReLU(),
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


class DoubleConvDownWCoT(nn.Module):
    def __init__(self, in_channels, out_channels,  num_groups=8):
        super().__init__()
        self.double_conv = nn.Sequential(
            CoTAttention(in_channels, 3),
            nn.GroupNorm(num_groups=num_groups, num_channels=in_channels),
            nn.ReLU(inplace=True),

            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.double_conv(x)


class DoubleConvUpWCoT(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups=8):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.ReLU(inplace=True),

            CoTAttention(out_channels, 3),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.ReLU(inplace=True)
          )

    def forward(self,x):
        return self.double_conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size = 1)

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

        return attn_out + x

class ResCoTAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResCoTAttention, self).__init__()
        self.hidden_channels = out_channels // 2
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, self.hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=self.hidden_channels),
            self.relu
        )

        self.conv2 = nn.Sequential(
            CoTAttention(self.hidden_channels, 3),
            nn.GroupNorm(num_groups=8, num_channels=self.hidden_channels),
            self.relu
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(self.hidden_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
        )

        self.residual = None
        if in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1),
                nn.GroupNorm(num_groups=8, num_channels=out_channels)
            )

    def forward(self, x):
        identity = x    
        if self.residual is not None:
            identity = self.residual(x)

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        out += identity
        out = self.relu(out)

        return out