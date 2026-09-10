import torch
import torch.nn as nn
import torch.nn.functional as F

from models2 import DepthNet
from model1.model_convnext import CP_Attention_block1, dwt_ffc_UNet2


class DepthConditionedMixer(nn.Module):
    """Lightweight depth-conditioned token mixer for Mamba-style context modeling."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.local_mixer = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels, bias=False
        )
        self.channel_mixer = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.depth_gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, channels * 2, kernel_size=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=True),
        )

    def forward(self, x, depth_feat):
        h = self.norm(x)
        h = self.channel_mixer(self.local_mixer(h))
        h = h * (1.0 + self.depth_gate(depth_feat))
        x = x + h
        x = x + self.ffn(self.norm(x))
        return x


class VisionMambaDepthContext(nn.Module):
    """Mamba-oriented context branch with explicit depth conditioning."""

    def __init__(self, out_channels=28, hidden_dim=64, blocks=4):
        super().__init__()
        self.image_stem = nn.Sequential(
            nn.Conv2d(3, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.depth_stem = nn.Sequential(
            nn.Conv2d(1, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.mixers = nn.ModuleList(
            [DepthConditionedMixer(hidden_dim) for _ in range(blocks)]
        )
        self.out_proj = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=True),
        )

    def forward(self, image, depth):
        image_feat = self.image_stem(image)
        depth_feat = self.depth_stem(depth)
        feat = image_feat
        for block in self.mixers:
            feat = block(feat, depth_feat)
        return self.out_proj(feat)


class DepthGeometryHead(nn.Module):
    """Produce geometry features and confidence from predicted depth."""

    def __init__(self, geometry_channels=32):
        super().__init__()
        self.geometry = nn.Sequential(
            nn.Conv2d(1, geometry_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, geometry_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(geometry_channels, geometry_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
        )
        self.confidence = nn.Sequential(
            nn.Conv2d(geometry_channels + 1, geometry_channels // 2, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(geometry_channels // 2, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, depth):
        geometry = self.geometry(depth)
        confidence = self.confidence(torch.cat((geometry, depth), dim=1))
        return geometry, confidence


class fusion_net_depth_best_mamba(nn.Module):
    """Depth-confidence-gated Mamba dehazing model."""

    def __init__(self, crop_size=256, backbone="visionmamba", embed_dim=64):
        super().__init__()
        self.crop_size = crop_size
        self.backbone = backbone

        self.dwt_branch = dwt_ffc_UNet2()
        self.depth_branch = DepthNet.DN()
        self.knowledge_adaptation_branch = VisionMambaDepthContext(out_channels=28)
        self.depth_geometry_head = DepthGeometryHead(geometry_channels=32)

        self.proj_dwt = nn.Sequential(
            nn.Conv2d(3, embed_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(inplace=True),
        )
        self.proj_ka = nn.Sequential(
            nn.Conv2d(28, embed_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(inplace=True),
        )
        self.base_fusion = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(inplace=True),
        )
        self.proj_depth = nn.Sequential(
            nn.Conv2d(33, embed_dim, 3, padding=1, bias=False),
            nn.GroupNorm(8, embed_dim),
            nn.SiLU(inplace=True),
        )
        self.depth_adapter = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim, 3, padding=1, bias=False),
            nn.GroupNorm(8, embed_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
        )
        self.depth_scale = nn.Parameter(torch.zeros(1))

        self.refine = nn.Sequential(
            CP_Attention_block1(nn.Conv2d, embed_dim, kernel_size=3),
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
            nn.SiLU(inplace=True),
        )
        self.tail = nn.Sequential(
            nn.Conv2d(embed_dim, 32, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Tanh(),
        )

    @staticmethod
    def _resize(x, size):
        if x.shape[-2:] != size:
            x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        return x

    def forward(self, image, return_depth=False, return_aux=False):
        target_size = image.shape[-2:]
        dwt = self._resize(self.dwt_branch(image), target_size)
        depth = self._resize(self.depth_branch(image), target_size)
        semantic = self._resize(self.knowledge_adaptation_branch(image, depth), target_size)

        geometry, confidence = self.depth_geometry_head(depth)
        geometry = self._resize(geometry, target_size)
        confidence = self._resize(confidence, target_size)

        base = self.base_fusion(torch.cat((self.proj_dwt(dwt), self.proj_ka(semantic)), dim=1))
        depth_feat = self.proj_depth(torch.cat((geometry, depth), dim=1))
        correction = self.depth_adapter(torch.cat((base, depth_feat), dim=1))
        fused = base + torch.tanh(self.depth_scale) * confidence * correction
        residual = 0.25 * self.tail(self.refine(fused))
        restored = torch.clamp(image + residual, 0.0, 1.0)

        if return_aux:
            return restored, {
                "depth": depth,
                "confidence": confidence,
                "depth_scale": torch.tanh(self.depth_scale),
            }
        if return_depth:
            return restored, depth
        return restored
