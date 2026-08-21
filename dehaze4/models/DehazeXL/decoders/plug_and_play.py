import torch
import torch.nn as nn
import torch.nn.functional as F

class SFTLayer(nn.Module):
    """
    Spatial Feature Transform (SFT) Layer
    Used to modulate features (x) based on a condition (cond, e.g., depth map).
    """
    def __init__(self, in_channels, cond_channels):
        super(SFTLayer, self).__init__()
        self.sft_scale = nn.Sequential(
            nn.Conv2d(cond_channels, 32, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, in_channels, 1)
        )
        self.sft_shift = nn.Sequential(
            nn.Conv2d(cond_channels, 32, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, in_channels, 1)
        )

    def forward(self, x, cond):
        # Resize condition to match x if necessary
        if cond.shape[2:] != x.shape[2:]:
            cond = F.interpolate(cond, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        scale = self.sft_scale(cond)
        shift = self.sft_shift(cond)
        
        return x * (scale + 1) + shift

class SKFusion(nn.Module):
    """
    Selective Kernel Fusion
    Dynamically fuses features from multiple branches (e.g., RGB, Depth, DWT).
    """
    def __init__(self, in_channels, height=32, reduction=8, group_size=3):
        super(SKFusion, self).__init__()
        self.height = height
        self.d = max(int(in_channels / reduction), 4)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, self.d, 1, bias=False),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(self.d, in_channels * group_size, 1, bias=False)
        )
        self.softmax = nn.Softmax(dim=1)
        self.in_channels = in_channels
        self.group_size = group_size

    def forward(self, *feats):
        # feats: list of tensors [B, C, H, W]
        assert len(feats) == self.group_size
        batch_size = feats[0].size(0)
        
        # Stack features: [B, G, C, H, W]
        feats_stack = torch.stack(feats, dim=1)
        
        # Sum over groups: [B, C, H, W] -> U
        U = torch.sum(feats_stack, dim=1)
        
        # Squeeze: [B, C, 1, 1]
        S = self.avg_pool(U)
        
        # Excitation: [B, G*C, 1, 1]
        Z = self.mlp(S)
        
        # Reshape: [B, G, C, 1, 1]
        Z = Z.view(batch_size, self.group_size, self.in_channels, 1, 1)
        
        # Softmax over groups
        weights = self.softmax(Z)
        
        # Fuse
        V = (feats_stack * weights).sum(dim=1)
        return V

class CBAM(nn.Module):
    """
    Convolutional Block Attention Module
    """
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        x_out = self.SpatialGate(x_out)
        return x_out

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        self.spatial = BasicConv(2, 1, kernel_size=7, stride=1, padding=3, relu=False)

    def forward(self, x):
        x_compress = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        scale = F.sigmoid(self.spatial(x_compress))
        return x * scale

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

