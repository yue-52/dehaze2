import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from models2 import DepthNet, DepthNet2
# import Convnext as PreConv
from model1.myFFCResblock0 import myFFCResblock
from models.DehazeXL.decoders.decoder1 import DehazeXL
from models.DehazeXL.decoders.decoder2 import DehazeXL2
from models.DehazeXL.decoders.decoder3 import DehazeXL3
from models.DehazeXL.decoders.decoder4 import DehazeXLWithMaskedAttention
from torchvision.ops import DeformConv2d
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights

try:
    from classification.models.intern_image import InternImage
except (ImportError, OSError):
    # InternImage depends on a platform-specific DCNv3 extension. Most models in
    # this file, including depth_geometry_v1, do not require that optional branch.
    InternImage = None

class InternImageEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        if InternImage is None:
            raise ImportError("InternImage requires a DCNv3 extension built for this platform")


        self.backbone = InternImage(

            core_op='DCNv3',

            channels=64,

            depths=[4,4,18,4],

            groups=[4,8,16,32],

            mlp_ratio=4.0,

            drop_path_rate=0.2,

            num_classes=0
        )


    def forward(self,x):

        return self.backbone(x)


class InternImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        if InternImage is None:
            raise ImportError("InternImage requires a DCNv3 extension built for this platform")

        self.backbone = InternImage(
            core_op='DCNv3',
            channels=64,
            depths=[4,4,18,4],
            groups=[4,8,16,32],
            mlp_ratio=4.0,
            drop_path_rate=0.2,
            num_classes=0
        )

        checkpoint = torch.load(
            "/newhome/zhangbaoguo/project1/DehazeXL/weight/internimage_t_1k_224.pth",
            map_location="cpu"
        )

        
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint   

        
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        
        msg = self.backbone.load_state_dict(state_dict, strict=False)
        print(msg)
    def forward(self, x):
    
        features = self.backbone.forward_features_seq_out(x)  # list of 4 tensors, each shape [B, H_i, W_i, C_i]
    
        f1, f2, f3, f4 = [f.permute(0, 3, 1, 2) for f in features]
        return f1, f2, f3, f4


class knowledge_adaptation_internimage(nn.Module):


    def __init__(self):

        super().__init__()


        self.encoder=InternImageEncoder()



        self.decoder=nn.Sequential(

            nn.Conv2d(
                64+128+256+512,
                128,
                3,
                padding=1
            ),

            nn.BatchNorm2d(128),

            nn.GELU(),


            nn.Conv2d(
                128,
                28,
                3,
                padding=1
            )

        )



    def forward(self,x):


        H,W=x.shape[2:]


        f1,f2,f3,f4=self.encoder(x)



        f2=F.interpolate(
            f2,
            size=f1.shape[2:]
        )


        f3=F.interpolate(
            f3,
            size=f1.shape[2:]
        )


        f4=F.interpolate(
            f4,
            size=f1.shape[2:]
        )


        feat=torch.cat(
            [
            f1,
            f2,
            f3,
            f4
            ],
            dim=1
        )


        feat=self.decoder(feat)


        feat=F.interpolate(
            feat,
            size=(H,W)
        )


        return feat




class DSConvBlock(nn.Module):
    """
    Lightweight Depthwise Separable Convolution Block
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.dwconv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(in_channels)

        self.act = nn.SiLU(inplace=True)

        self.pwconv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):

        x = self.dwconv(x)
        x = self.bn1(x)
        x = self.act(x)

        x = self.pwconv(x)
        x = self.bn2(x)
        x = self.act(x)

        return x


class LightweightContextBlock(nn.Module):

    def __init__(self, channels):

        super().__init__()

        self.local_branch = nn.Sequential(

            nn.Conv2d(
                channels,
                channels,
                kernel_size=5,
                padding=2,
                groups=channels,
                bias=False
            ),

            nn.BatchNorm2d(channels),

            nn.SiLU(inplace=True)
        )

        self.context_branch = nn.Sequential(

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=3,
                dilation=3,
                groups=channels,
                bias=False
            ),

            nn.BatchNorm2d(channels),

            nn.SiLU(inplace=True)
        )

        self.fusion = nn.Conv2d(
            channels * 2,
            channels,
            kernel_size=1,
            bias=False
        )

        self.attention = nn.Sequential(

            nn.AdaptiveAvgPool2d(1),

            nn.Conv2d(
                channels,
                channels // 4,
                kernel_size=1
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                channels // 4,
                channels,
                kernel_size=1
            ),

            nn.Sigmoid()
        )

    def forward(self, x):

        identity = x

        local = self.local_branch(x)

        context = self.context_branch(x)

        feat = torch.cat(
            [local, context],
            dim=1
        )

        feat = self.fusion(feat)

        attn = self.attention(feat)

        feat = feat * attn

        return feat + identity


class LightweightSemanticBranch(nn.Module):

    def __init__(self, out_channels=28):

        super().__init__()

        # Stem
        self.stem = nn.Sequential(

            nn.Conv2d(
                3,
                32,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(32),

            nn.SiLU(inplace=True)
        )

        # Stage 1
        self.stage1 = nn.Sequential(

            DSConvBlock(
                32,
                48,
                stride=2
            ),

            LightweightContextBlock(48)
        )

        # Stage 2
        self.stage2 = nn.Sequential(

            DSConvBlock(
                48,
                96,
                stride=2
            ),

            LightweightContextBlock(96),

            LightweightContextBlock(96)
        )

        # Stage 3
        self.stage3 = nn.Sequential(

            DSConvBlock(
                96,
                160,
                stride=2
            ),

            LightweightContextBlock(160),

            LightweightContextBlock(160)
        )

        # Semantic projection
        self.semantic_proj = nn.Sequential(

            nn.Conv2d(
                160,
                96,
                kernel_size=1,
                bias=False
            ),

            nn.BatchNorm2d(96),

            nn.SiLU(inplace=True)
        )

        # Lightweight decoder
        self.decoder1 = nn.Sequential(

            nn.Conv2d(
                96,
                64,
                kernel_size=3,
                padding=1,
                groups=1,
                bias=False
            ),

            nn.BatchNorm2d(64),

            nn.SiLU(inplace=True)
        )

        self.decoder2 = nn.Sequential(

            nn.Conv2d(
                64,
                32,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(32),

            nn.SiLU(inplace=True)
        )

        # Output 28 channels
        self.output_proj = nn.Conv2d(
            32,
            out_channels,
            kernel_size=3,
            padding=1
        )

    def forward(self, x):

        B, C, H, W = x.shape

        # Padding
        pad_h = (32 - H % 32) % 32
        pad_w = (32 - W % 32) % 32

        if pad_h > 0 or pad_w > 0:

            x = F.pad(
                x,
                (0, pad_w, 0, pad_h),
                mode='reflect'
            )

        # Encoder
        x = self.stem(x)

        x = self.stage1(x)

        x = self.stage2(x)

        x = self.stage3(x)

        # Semantic projection
        x = self.semantic_proj(x)

        # Decoder
        x = F.interpolate(
            x,
            scale_factor=2,
            mode='bilinear',
            align_corners=False
        )

        x = self.decoder1(x)

        x = F.interpolate(
            x,
            scale_factor=2,
            mode='bilinear',
            align_corners=False
        )

        x = self.decoder2(x)

        x = F.interpolate(
            x,
            size=(H, W),
            mode='bilinear',
            align_corners=False
        )

        x = self.output_proj(x)

        # Crop
        if pad_h > 0 or pad_w > 0:

            x = x[:, :, :H, :W]

        return x


class MultiScaleSemanticFusion(nn.Module):
    """
    Multi-scale semantic feature fusion.

    输入：
        f1: 1/4 resolution
        f2: 1/8 resolution
        f3: 1/16 resolution
        f4: 1/32 resolution

    输出：
        1/4 resolution semantic feature
    """

    def __init__(
        self,
        channels_list,
        out_channels=64
    ):
        super(
            MultiScaleSemanticFusion,
            self
        ).__init__()

        self.proj1 = nn.Sequential(

            nn.Conv2d(
                channels_list[0],
                out_channels,
                kernel_size=1,
                bias=False
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.SiLU(
                inplace=True
            )
        )

        self.proj2 = nn.Sequential(

            nn.Conv2d(
                channels_list[1],
                out_channels,
                kernel_size=1,
                bias=False
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.SiLU(
                inplace=True
            )
        )

        self.proj3 = nn.Sequential(

            nn.Conv2d(
                channels_list[2],
                out_channels,
                kernel_size=1,
                bias=False
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.SiLU(
                inplace=True
            )
        )

        self.proj4 = nn.Sequential(

            nn.Conv2d(
                channels_list[3],
                out_channels,
                kernel_size=1,
                bias=False
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.SiLU(
                inplace=True
            )
        )

        self.fusion = nn.Sequential(

            nn.Conv2d(
                out_channels * 4,
                out_channels,
                kernel_size=1,
                bias=False
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.SiLU(
                inplace=True
            ),

            LightweightContextBlock(
                out_channels
            )
        )

    def forward(
        self,
        f1,
        f2,
        f3,
        f4
    ):

        target_size = f1.shape[2:]

        p1 = self.proj1(f1)

        p2 = self.proj2(f2)

        p3 = self.proj3(f3)

        p4 = self.proj4(f4)

        p2 = F.interpolate(
            p2,
            size=target_size,
            mode='bilinear',
            align_corners=False
        )

        p3 = F.interpolate(
            p3,
            size=target_size,
            mode='bilinear',
            align_corners=False
        )

        p4 = F.interpolate(
            p4,
            size=target_size,
            mode='bilinear',
            align_corners=False
        )

        feat = torch.cat(
            [
                p1,
                p2,
                p3,
                p4
            ],
            dim=1
        )

        feat = self.fusion(
            feat
        )

        return feat


class DepthGuidedSemanticAttention(nn.Module):
    """
    Depth-Guided Semantic Attention

    使用深度特征生成空间注意力，
    在语义分支内部提前增强不同深度区域的语义表示。

    核心：
        Semantic Feature
              ×
        Depth Attention

    输出：
        Depth-guided semantic feature
    """

    def __init__(
        self,
        semantic_channels,
        depth_channels=64,
        reduction=4
    ):
        super(
            DepthGuidedSemanticAttention,
            self
        ).__init__()

        hidden_channels = max(
            semantic_channels // reduction,
            8
        )

        # Depth encoder
        self.depth_encoder = nn.Sequential(

            nn.Conv2d(
                depth_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                hidden_channels
            ),

            nn.SiLU(
                inplace=True
            )
        )

        # Semantic encoder
        self.semantic_encoder = nn.Sequential(

            nn.Conv2d(
                semantic_channels,
                hidden_channels,
                kernel_size=1,
                bias=False
            ),

            nn.BatchNorm2d(
                hidden_channels
            ),

            nn.SiLU(
                inplace=True
            )
        )

        # Spatial attention
        self.attention = nn.Sequential(

            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                hidden_channels
            ),

            nn.SiLU(
                inplace=True
            ),

            nn.Conv2d(
                hidden_channels,
                1,
                kernel_size=1,
                bias=True
            ),

            nn.Sigmoid()
        )

        # Learnable guidance strength
        self.alpha = nn.Parameter(
            torch.tensor(0.5)
        )

    def forward(
        self,
        semantic_feat,
        depth_feat
    ):

        # 尺寸对齐
        if (
            semantic_feat.shape[2:]
            != depth_feat.shape[2:]
        ):

            depth_feat = F.interpolate(
                depth_feat,
                size=semantic_feat.shape[2:],
                mode='bilinear',
                align_corners=False
            )

        semantic_embed = \
            self.semantic_encoder(
                semantic_feat
            )

        depth_embed = \
            self.depth_encoder(
                depth_feat
            )

        # Depth + Semantic interaction
        interaction = \
            semantic_embed + depth_embed

        # Generate spatial attention
        attention = \
            self.attention(
                interaction
            )

        # Depth-guided enhancement
        enhanced = \
            semantic_feat * (
                1.0
                + self.alpha
                * attention
            )

        return enhanced



class LightweightSemanticBranchV2(nn.Module):
    """
    Lightweight Multi-Scale Semantic Branch

    特点：
        1. Depthwise Separable Conv
        2. Multi-scale feature extraction
        3. Large-kernel contextual modeling
        4. Multi-scale semantic fusion
        5. Depth-guided semantic attention

    最终输出：
        [B, 28, H, W]

    与原 fusion_net_depth_best
    的 knowledge_adaptation_branch 接口保持一致。
    """

    def __init__(
        self,
        out_channels=28
    ):
        super(
            LightweightSemanticBranchV2,
            self
        ).__init__()

        # =====================================
        # Stem
        # =====================================

        self.stem = nn.Sequential(

            nn.Conv2d(
                3,
                32,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                32
            ),

            nn.SiLU(
                inplace=True
            )
        )

        # =====================================
        # Stage 1
        # 1/4
        # =====================================

        self.stage1 = nn.Sequential(

            DSConvBlock(
                32,
                48,
                stride=2
            ),

            LightweightContextBlock(
                48
            )
        )

        # =====================================
        # Stage 2
        # 1/8
        # =====================================

        self.stage2 = nn.Sequential(

            DSConvBlock(
                48,
                96,
                stride=2
            ),

            LightweightContextBlock(
                96
            ),

            LightweightContextBlock(
                96
            )
        )

        # =====================================
        # Stage 3
        # 1/16
        # =====================================

        self.stage3 = nn.Sequential(

            DSConvBlock(
                96,
                160,
                stride=2
            ),

            LightweightContextBlock(
                160
            ),

            LightweightContextBlock(
                160
            )
        )

        # =====================================
        # Stage 4
        # 1/32
        # =====================================

        self.stage4 = nn.Sequential(

            DSConvBlock(
                160,
                192,
                stride=2
            ),

            LightweightContextBlock(
                192
            )
        )

        # =====================================
        # Multi-scale fusion
        # =====================================

        self.multiscale_fusion = \
            MultiScaleSemanticFusion(

                channels_list=[
                    48,
                    96,
                    160,
                    192
                ],

                out_channels=64
            )

        # =====================================
        # Semantic context enhancement
        # =====================================

        self.semantic_context = nn.Sequential(

            LightweightContextBlock(
                64
            ),

            LightweightContextBlock(
                64
            )
        )

        # =====================================
        # Output
        # =====================================

        self.output_proj = nn.Sequential(

            nn.Conv2d(
                64,
                48,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                48
            ),

            nn.SiLU(
                inplace=True
            ),

            nn.Conv2d(
                48,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=True
            )
        )

    def forward(
        self,
        x,
        return_features=False
    ):

        B, C, H, W = x.shape

        # =====================================
        # Padding
        # =====================================

        pad_h = (
            32 - H % 32
        ) % 32

        pad_w = (
            32 - W % 32
        ) % 32

        if (
            pad_h > 0
            or pad_w > 0
        ):

            x = F.pad(
                x,
                (
                    0,
                    pad_w,
                    0,
                    pad_h
                ),
                mode='reflect'
            )

        # =====================================
        # Encoder
        # =====================================

        x = self.stem(x)

        f1 = self.stage1(x)

        f2 = self.stage2(f1)

        f3 = self.stage3(f2)

        f4 = self.stage4(f3)

        # =====================================
        # Multi-scale fusion
        # =====================================

        semantic_feat = \
            self.multiscale_fusion(
                f1,
                f2,
                f3,
                f4
            )

        # =====================================
        # Semantic context
        # =====================================

        semantic_feat = \
            self.semantic_context(
                semantic_feat
            )

        # =====================================
        # Upsample to original size
        # =====================================

        semantic_feat = F.interpolate(
            semantic_feat,
            size=(H, W),
            mode='bilinear',
            align_corners=False
        )

        # =====================================
        # Output projection
        # =====================================

        out = self.output_proj(
            semantic_feat
        )

        # =====================================
        # Crop
        # =====================================

        if (
            pad_h > 0
            or pad_w > 0
        ):

            out = out[
                :,
                :,
                :H,
                :W
            ]

        if return_features:

            return (
                out,
                {
                    'f1': f1,
                    'f2': f2,
                    'f3': f3,
                    'f4': f4,
                    'semantic': semantic_feat
                }
            )

        return out







class ResNet34Encoder(nn.Module):
    """
    使用 ResNet34 作为编码器，提取三个尺度的特征：
        - out1: 原始输入的 1/4 分辨率，通道数 256  (匹配 ConvNeXt 的 x_layer1)
        - out2: 原始输入的 1/8 分辨率，通道数 512  (匹配 ConvNeXt 的 x_layer2)
        - out3: 原始输入的 1/16 分辨率，通道数 1024 (匹配 ConvNeXt 的 x_output)
    """
    def __init__(self, pretrained=True):
        super().__init__()
        # 加载预训练的 ResNet34
        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = resnet34(weights=weights)
        
        # 定义各阶段（不包含最后的全连接层和全局池化）
        self.conv1 = nn.Sequential(
            resnet.conv1,   # 7x7, stride=2, out_channels=64
            resnet.bn1,
            resnet.relu,
            resnet.maxpool  # stride=2
        )  # 输出 1/4 分辨率，64 通道
        
        self.layer1 = resnet.layer1  # 64 -> 64, 输出分辨率 1/4
        self.layer2 = resnet.layer2  # 64 -> 128, 输出分辨率 1/8
        self.layer3 = resnet.layer3  # 128 -> 256, 输出分辨率 1/16
        self.layer4 = resnet.layer4  # 256 -> 512, 输出分辨率 1/32
        
        # 适配通道数到目标维度 (256, 512, 1024)
        self.adjust1 = nn.Conv2d(64, 256, kernel_size=1)   # layer1 输出 64 -> 256
        self.adjust2 = nn.Conv2d(128, 512, kernel_size=1)  # layer2 输出 128 -> 512
        self.adjust3 = nn.Conv2d(512, 1024, kernel_size=1) # layer4 输出 512 -> 1024
        
    def forward(self, x):
        # 输入 x: [B, 3, H, W]
        x = self.conv1(x)          # 1/4, 64ch
        f1 = self.layer1(x)        # 1/4, 64ch
        f2 = self.layer2(f1)       # 1/8, 128ch
        f3 = self.layer3(f2)       # 1/16, 256ch
        f4 = self.layer4(f3)       # 1/32, 512ch
        
        # 调整通道数
        out1 = self.adjust1(f1)    # 1/4, 256ch
        out2 = self.adjust2(f2)    # 1/8, 512ch
        out3 = self.adjust3(f4)    # 1/32, 1024ch
        
        return out1, out2, out3

class knowledge_adaptation_resnet34(nn.Module):
    """基于 ResNet34 的知识适应分支，输出 28 通道特征图"""
    def __init__(self):
        super().__init__()
        # 编码器
        self.encoder = ResNet34Encoder(pretrained=True)
        
        # 后续模块（与原 ConvNeXt 版本完全一致）
        self.up_block = nn.PixelShuffle(2)
        self.attention0 = CP_Attention_block(default_conv, 1024, 3)
        self.attention1 = CP_Attention_block(default_conv, 256, 3)
        self.attention2 = CP_Attention_block(default_conv, 192, 3)
        self.attention3 = CP_Attention_block(default_conv, 112, 3)
        self.attention4 = CP_Attention_block(default_conv, 28, 3)
        self.conv_process_1 = nn.Conv2d(28, 28, kernel_size=3, padding=1)
        self.conv_process_2 = nn.Conv2d(28, 28, kernel_size=3, padding=1)
        
    def forward(self, input):
        # 动态填充以确保编码器输出尺寸为整数
        B, C, H, W = input.shape
        pad_h = (32 - H % 32) % 32 if H % 32 != 0 else 0
        pad_w = (32 - W % 32) % 32 if W % 32 != 0 else 0
        if pad_h > 0 or pad_w > 0:
            input = F.pad(input, (0, pad_w, 0, pad_h), mode='reflect')
        
        # 编码器提取三个尺度的特征
        x_layer1, x_layer2, x_output = self.encoder(input)
        
        # 解码/上采样路径（与原实现完全一致）
        x_mid = self.attention0(x_output)          # [1024, H/32, W/32]
        x = self.up_block(x_mid)                   # [256, H/16, W/16]
        x = self.attention1(x)                     # [256, H/16, W/16]
        
        # 拼接 x_layer2 (分辨率 H/8) 前需上采样对齐
        if x.shape[2:] != x_layer2.shape[2:]:
            x = F.interpolate(x, size=x_layer2.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat((x, x_layer2), 1)            # [256+512=768, H/8, W/8]
        
        x = self.up_block(x)                       # [192, H/4, W/4]
        x = self.attention2(x)                     # [192, H/4, W/4]
        
        if x.shape[2:] != x_layer1.shape[2:]:
            x = F.interpolate(x, size=x_layer1.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat((x, x_layer1), 1)            # [192+256=448, H/4, W/4]
        x = self.up_block(x)                       # [112, H/2, W/2]
        x = self.attention3(x)                     # [112, H/2, W/2]
        
        x = self.up_block(x)                       # [28, H, W]
        x = self.attention4(x)                     # [28, H, W]
        
        x = self.conv_process_1(x)
        out = self.conv_process_2(x)               # [28, H, W]
        
        # 裁剪回原始尺寸
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :H, :W]
        return out

class ConvNeXt0(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(self, block, in_chans=3, num_classes=1000,
                 depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], drop_path_rate=0.,
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[block(dim=dims[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2  # x01.shape=[4,3,128,256]   从0开始，每隔两个取出，#像素值还要除以2    0,2,4,6...254-->0,1,2,...127
    x02 = x[:, :, 1::2, :] / 2  # x02.shape=[4,3,128,256]   从1开始，每隔两个取出，#像素值还要除以2    1,3,5,7...255-->0,1,2,...127
    x1 = x01[:, :, :, 0::2]  # x1.shape=[4,3,128,128]   从0取出      0,2,4,6...254-->0,1,2,...127
    x2 = x02[:, :, :, 0::2]  # x2.shape=[4,3,128,128]   从0取出
    x3 = x01[:, :, :, 1::2]  # x3.shape=[4,3,128,128]   从1取出     1,3,5,7...255-->0,1,2,...127
    x4 = x02[:, :, :, 1::2]  # x4.shape=[4,3,128,128]   从1取出
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4
    return x_LL, torch.cat((x_HL, x_LH, x_HH), 1)


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)


class DWT_transform(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dwt = DWT()
        self.conv1x1_low = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.conv1x1_high = nn.Conv2d(in_channels * 3, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        dwt_low_frequency, dwt_high_frequency = self.dwt(x)
        dwt_low_frequency = self.conv1x1_low(dwt_low_frequency)
        dwt_high_frequency = self.conv1x1_high(dwt_high_frequency)
        return dwt_low_frequency, dwt_high_frequency


def blockUNet(in_c, out_c, name, transposed=False, bn=False, relu=True, dropout=False):
    block = nn.Sequential()
    if relu:
        block.add_module('%s_relu' % name, nn.ReLU(inplace=True))
    else:
        block.add_module('%s_leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
    if not transposed:
        block.add_module('%s_conv' % name, nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False))
    else:
        block.add_module('%s_tconv' % name, nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False))
    if bn:
        block.add_module('%s_bn' % name, nn.BatchNorm2d(out_c))
    if dropout:
        block.add_module('%s_dropout' % name, nn.Dropout2d(0.5, inplace=True))
    return block


class dwt_ffc_UNet2(nn.Module):
    def __init__(self, output_nc=3, nf=16):
        super(dwt_ffc_UNet2, self).__init__()
        layer_idx = 1
        name = 'layer%d' % layer_idx
        layer1 = nn.Sequential()
        layer1.add_module(name, nn.Conv2d(16, nf - 1, 4, 2, 1, bias=False))
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer2 = blockUNet(nf, nf * 2 - 2, name, transposed=False, bn=True, relu=False, dropout=False)
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer3 = blockUNet(nf * 2, nf * 4 - 4, name, transposed=False, bn=True, relu=False, dropout=False)
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer4 = blockUNet(nf * 4, nf * 8 - 8, name, transposed=False, bn=True, relu=False, dropout=False)
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer5 = blockUNet(nf * 8, nf * 8 - 16, name, transposed=False, bn=True, relu=False, dropout=False)
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer6 = blockUNet(nf * 4, nf * 4, name, transposed=False, bn=False, relu=False, dropout=False)  # 有改动

        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        dlayer6 = blockUNet(nf * 4, nf * 2, name, transposed=True, bn=True, relu=True, dropout=False)
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        dlayer5 = blockUNet(nf * 16 + 16, nf * 8, name, transposed=True, bn=True, relu=True, dropout=False)
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        dlayer4 = blockUNet(nf * 16 + 8, nf * 4, name, transposed=True, bn=True, relu=True, dropout=False)
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        dlayer3 = blockUNet(nf * 8 + 4, nf * 2, name, transposed=True, bn=True, relu=True, dropout=False)
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        dlayer2 = blockUNet(nf * 4 + 2, nf, name, transposed=True, bn=True, relu=True, dropout=False)
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        dlayer1 = blockUNet(nf * 2 + 1, nf * 2, name, transposed=True, bn=True, relu=True, dropout=False)

        self.initial_conv = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = layer1
        self.DWT_down_0 = DWT_transform(3, 1)
        self.layer2 = layer2
        self.DWT_down_1 = DWT_transform(16, 2)
        self.layer3 = layer3
        self.DWT_down_2 = DWT_transform(32, 4)
        self.layer4 = layer4
        self.DWT_down_3 = DWT_transform(64, 8)
        self.layer5 = layer5
        self.DWT_down_4 = DWT_transform(128, 16)
        self.layer6 = layer6
        self.dlayer6 = dlayer6
        self.dlayer5 = dlayer5
        self.dlayer4 = dlayer4
        self.dlayer3 = dlayer3
        self.dlayer2 = dlayer2
        self.dlayer1 = dlayer1
        self.tail_conv1 = nn.Conv2d(48, 32, 3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(32)
        self.tail_conv2 = nn.Conv2d(nf * 2, output_nc, 3, padding=1, bias=True)

        self.FFCResNet = myFFCResblock(input_nc=64, output_nc=64)

    def forward(self, x):
        # 动态填充以确保尺寸合适
        B, C, H, W = x.shape
        pad_h = (64 - H % 64) % 64 if H % 64 != 0 else 0
        pad_w = (64 - W % 64) % 64 if W % 64 != 0 else 0

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        conv_start = self.initial_conv(x)
        conv_start = self.bn1(conv_start)
        conv_out1 = self.layer1(conv_start)
        dwt_low_0, dwt_high_0 = self.DWT_down_0(x)
        out1 = torch.cat([conv_out1, dwt_low_0], 1)
        conv_out2 = self.layer2(out1)
        dwt_low_1, dwt_high_1 = self.DWT_down_1(out1)
        out2 = torch.cat([conv_out2, dwt_low_1], 1)
        conv_out3 = self.layer3(out2)

        dwt_low_2, dwt_high_2 = self.DWT_down_2(out2)
        out3 = torch.cat([conv_out3, dwt_low_2], 1)

        # conv_out4 = self.layer4(out3)
        # dwt_low_3,dwt_high_3 = self.DWT_down_3(out3)
        # out4 = torch.cat([conv_out4, dwt_low_3], 1)

        # conv_out5 = self.layer5(out4)
        # dwt_low_4,dwt_high_4 = self.DWT_down_4(out4)
        # out5 = torch.cat([conv_out5, dwt_low_4], 1)

        # out6 = self.layer6(out5)

        out3_ffc = self.FFCResNet(out3)

        dout3 = self.dlayer6(out3_ffc)

        # Tout6_out5 = torch.cat([dout6, out5, dwt_high_4], 1)

        # Tout5 = self.dlayer5(Tout6_out5)
        # Tout5_out4 = torch.cat([Tout5, out4,dwt_high_3], 1)
        # Tout4 = self.dlayer4(Tout5_out4)
        # Tout4_out3 = torch.cat([Tout4, out3,dwt_high_2], 1)        # Tout3 = self.dlayer3(Tout4_out3)
        Tout3_out2 = torch.cat([dout3, out2, dwt_high_1], 1)
        Tout2 = self.dlayer2(Tout3_out2)
        Tout2_out1 = torch.cat([Tout2, out1, dwt_high_0], 1)
        Tout1 = self.dlayer1(Tout2_out1)

        Tout1_outinit = torch.cat([Tout1, conv_start], 1)
        tail1 = self.tail_conv1(Tout1_outinit)
        tail2 = self.bn2(tail1)
        dout1 = self.tail_conv2(tail2)
        # 如果进行了填充，在输出时裁剪回原始尺寸
        if pad_h > 0 or pad_w > 0:
            dout1 = dout1[:, :, :H, :W]

        return dout1


class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class ConvNeXt(nn.Module):
    def __init__(self, block, in_chans=3, num_classes=1000,
                 depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], drop_path_rate=0.,
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[block(dim=dims[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def forward(self, x):
        x_layer1 = self.downsample_layers[0](x)
        x_layer1 = self.stages[0](x_layer1)

        x_layer2 = self.downsample_layers[1](x_layer1)
        x_layer2 = self.stages[1](x_layer2)

        x_layer3 = self.downsample_layers[2](x_layer2)
        out = self.stages[2](x_layer3)

        return x_layer1, x_layer2, out


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y


class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y


class CP_Attention_block(nn.Module):
    def __init__(self, conv, dim, kernel_size):
        super(CP_Attention_block, self).__init__()
        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.calayer = CALayer(dim)  # 通道注意力
        self.palayer = PALayer(dim)  # 相素注意力

    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = res + x
        res = self.conv2(res)
        res = self.calayer(res)
        res = self.palayer(res)
        res += x
        return res



def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class knowledge_adaptation_convnext(nn.Module):  # 知识适应分支参数量: 371,614,743 (371.61M)
    def __init__(self):
        super(knowledge_adaptation_convnext, self).__init__()
        self.encoder = ConvNeXt(Block, in_chans=3, num_classes=1000, depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048],
                                drop_path_rate=0., layer_scale_init_value=1e-6,
                                head_init_scale=1.)  # 最新的2024年编码器使用了最新的第四代可变形卷积
        # pretrained_model=nn.DataParallel(pretrained_model)
        checkpoint_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "weights",
            "convnext_xlarge_22k_1k_384_ema.pth",
        )
        # for k,v in checkpoint["model"].items():
        # print(k)
        # url="https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_384.pth"

        # checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cuda:0")
        if os.path.isfile(checkpoint_path):
            pretrained_model = ConvNeXt0(
                Block, in_chans=3, num_classes=1000, depths=[3, 3, 27, 3],
                dims=[256, 512, 1024, 2048], drop_path_rate=0.,
                layer_scale_init_value=1e-6, head_init_scale=1.,
            )
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            pretrained_model.load_state_dict(checkpoint["model"])
            pretrained_dict = pretrained_model.state_dict()
            model_dict = self.encoder.state_dict()
            key_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(key_dict)
            self.encoder.load_state_dict(model_dict)
        else:
            print(f"Warning: optional ConvNeXt initialization weight not found: {checkpoint_path}")

        self.up_block = nn.PixelShuffle(2)
        self.attention0 = CP_Attention_block(default_conv, 1024, 3)
        self.attention1 = CP_Attention_block(default_conv, 256, 3)
        self.attention2 = CP_Attention_block(default_conv, 192, 3)
        self.attention3 = CP_Attention_block(default_conv, 112, 3)
        self.attention4 = CP_Attention_block(default_conv, 28, 3)
        self.conv_process_1 = nn.Conv2d(28, 28, kernel_size=3, padding=1)
        self.conv_process_2 = nn.Conv2d(28, 28, kernel_size=3, padding=1)
        self.tail = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(28, 3, kernel_size=7, padding=0), nn.Tanh())

    # def forward(self, input):
    #     x_layer1, x_layer2, x_output = self.encoder(input)
    #
    #     x_mid = self.attention0(x_output)  #[1024,24,24]
    #
    #     x = self.up_block(x_mid)      #[256,48,48]
    #     x = self.attention1(x)
    #
    #     x = torch.cat((x, x_layer2), 1)  #[768,48,48]
    #
    #     x = self.up_block(x)            #[192,96,96]
    #     x = self.attention2(x)
    #     x = torch.cat((x, x_layer1), 1)   #[448,96,96]
    #     x = self.up_block(x)            #[112,192,192]
    #     x = self.attention3(x)
    #
    #     x = self.up_block(x)        #[28,384,384]
    #     x = self.attention4(x)
    #
    #     x=self.conv_process_1(x)
    #     out=self.conv_process_2(x)
    #     return out
    def forward(self, input):
        # 动态填充确保输入尺寸合适
        B, C, H, W = input.shape
        # ConvNeXt需要输入是32的倍数
        pad_h = (32 - H % 32) % 32 if H % 32 != 0 else 0
        pad_w = (32 - W % 32) % 32 if W % 32 != 0 else 0

        if pad_h > 0 or pad_w > 0:
            input = F.pad(input, (0, pad_w, 0, pad_h), mode='reflect')

        x_layer1, x_layer2, x_output = self.encoder(input)  # 编码器参数量: 350,196,968 (350.20M)

        x_mid = self.attention0(x_output)

        x = self.up_block(x_mid)
        x = self.attention1(x)

        # 确保尺寸匹配
        if x.shape[2:] != x_layer2.shape[2:]:
            x = F.interpolate(x, size=x_layer2.shape[2:], mode='bilinear', align_corners=False)

        x = torch.cat((x, x_layer2), 1)

        x = self.up_block(x)
        x = self.attention2(x)

        # 确保尺寸匹配
        if x.shape[2:] != x_layer1.shape[2:]:
            x = F.interpolate(x, size=x_layer1.shape[2:], mode='bilinear', align_corners=False)

        x = torch.cat((x, x_layer1), 1)
        x = self.up_block(x)
        x = self.attention3(x)

        x = self.up_block(x)
        x = self.attention4(x)

        x = self.conv_process_1(x)
        out = self.conv_process_2(x)

        # 如果进行了填充，裁剪回原始尺寸
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :H, :W]

        return out
class fusion_net_1(nn.Module):
    def __init__(self):
        super(fusion_net_1, self).__init__()
        self.dwt_branch=dwt_ffc_UNet2()
        #self.knowledge_adaptation_branch = knowledge_adaptation_internimage()
        self.knowledge_adaptation_branch = knowledge_adaptation_convnext()
        
        self.fusion = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(31, 3, kernel_size=7, padding=0), nn.Tanh())
    def forward(self, input):
        dwt_branch=self.dwt_branch(input)
        knowledge_adaptation_branch=self.knowledge_adaptation_branch(input)
        x = torch.cat([dwt_branch, knowledge_adaptation_branch], 1)
        x = self.fusion(x)
        return x

class fusion_net(nn.Module):
    def __init__(self, crop_size: int = 256,
                 mlp_ratio: int = 4):
        super(fusion_net, self).__init__()
        self.crop_size = crop_size
        self.mlp_ratio = mlp_ratio
        self.dwt_branch = dwt_ffc_UNet2()  # 三个通道
        self.knowledge_adaptation_branch = DehazeXL()  # 28个通道
        self.fusion = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(31, 3, kernel_size=7, padding=0), nn.Tanh())

    def forward(self, input):
        dwt_branch = self.dwt_branch(input)
        knowledge_adaptation_branch = self.knowledge_adaptation_branch(input)
        x = torch.cat([dwt_branch, knowledge_adaptation_branch], 1)
        x = self.fusion(x)
        print("XL-28channel+DWT")
        return x


class DFF_2D(nn.Module):
    def __init__(self, dim):
        super().__init__()

        # 使用2D平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # 使用2D卷积
        self.conv_atten = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 2, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.conv_redu = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False)

        self.conv1 = nn.Conv2d(dim, 1, kernel_size=1, stride=1, bias=True)
        self.conv2 = nn.Conv2d(dim, 1, kernel_size=1, stride=1, bias=True)
        self.nonlin = nn.Sigmoid()

    def forward(self, x, skip):
        output = torch.cat([x, skip], dim=1)

        att = self.conv_atten(self.avg_pool(output))
        output = output * att
        output = self.conv_redu(output)

        att = self.conv1(x) + self.conv2(skip)
        att = self.nonlin(att)
        output = output * att
        return output


class fusion_net2(nn.Module):
    def __init__(self, crop_size: int = 256, mlp_ratio: int = 4):
        super(fusion_net2, self).__init__()
        self.crop_size = crop_size
        self.mlp_ratio = mlp_ratio
        self.dwt_branch = dwt_ffc_UNet2()  # 输出3个通道
        self.knowledge_adaptation_branch = DehazeXL()  # 注意：这里应该是28个通道

        # 修改为DFF_2D
        self.fusion = DFF_2D(3)

    def forward(self, input):
        dwt_branch = self.dwt_branch(input)  # [B, 3, H, W]
        knowledge_adaptation_branch = self.knowledge_adaptation_branch(input)  # [B, 28, H, W]
        x = self.fusion(knowledge_adaptation_branch, dwt_branch)
        print('XL+DWT+DFF')
        return x


class fusion_net3(nn.Module):
    def __init__(self, crop_size: int = 256,
                 mlp_ratio: int = 4):
        super(fusion_net3, self).__init__()
        self.crop_size = crop_size
        self.mlp_ratio = mlp_ratio
        self.dwt_branch = dwt_ffc_UNet2()  # 三个通道
        self.knowledge_adaptation_branch = DehazeXL2()  # 28个通道
        self.fusion = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(6, 3, kernel_size=7, padding=0), nn.Tanh())

    def forward(self, input):
        dwt_branch = self.dwt_branch(input)
        knowledge_adaptation_branch = self.knowledge_adaptation_branch(input)
        x = torch.cat([dwt_branch, knowledge_adaptation_branch], 1)
        x = self.fusion(x)
        print("XL+DWT+FreqFusion")
        return x


class fusion_net4(nn.Module):
    def __init__(self, crop_size: int = 256,
                 mlp_ratio: int = 4):
        super(fusion_net4, self).__init__()
        self.crop_size = crop_size
        self.mlp_ratio = mlp_ratio
        self.dwt_branch = dwt_ffc_UNet2()  # 三个通道
        self.knowledge_adaptation_branch = DehazeXL3()  # 28个通道
        self.fusion = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(6, 3, kernel_size=7, padding=0), nn.Tanh())

    def forward(self, input):
        dwt_branch = self.dwt_branch(input)
        knowledge_adaptation_branch = self.knowledge_adaptation_branch(input)
        x = torch.cat([dwt_branch, knowledge_adaptation_branch], 1)
        print("XL+DWT+反卷积")
        x = self.fusion(x)
        return x


class fusion_net5(nn.Module):
    def __init__(self, crop_size: int = 256,
                 mlp_ratio: int = 4):
        super(fusion_net5, self).__init__()
        self.crop_size = crop_size
        self.mlp_ratio = mlp_ratio
        self.dwt_branch = dwt_ffc_UNet2()  # 三个通道
        self.knowledge_adaptation_branch = DehazeXLWithMaskedAttention()  # 28个通道
        self.fusion = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(6, 3, kernel_size=7, padding=0), nn.Tanh())

    def forward(self, input):
        dwt_branch = self.dwt_branch(input)
        knowledge_adaptation_branch = self.knowledge_adaptation_branch(input)
        x = torch.cat([dwt_branch, knowledge_adaptation_branch], 1)
        x = self.fusion(x)
        print("XL+DWT+掩码注意力")
        return x


class fusion_net6(nn.Module):
    def __init__(self, crop_size: int = 256, mlp_ratio: int = 4):
        super(fusion_net6, self).__init__()
        self.crop_size = crop_size
        self.mlp_ratio = mlp_ratio
        self.dwt_branch = dwt_ffc_UNet2()
        self.knowledge_adaptation_branch = DehazeXL()
        self.depth_branch = DepthNet.DN()
        self.fusion = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(7, 3, kernel_size=7, padding=0), nn.Tanh())

    def forward(self, input):
        dwt_branch = self.dwt_branch(input)
        knowledge_adaptation_branch = self.knowledge_adaptation_branch(input)
        depth_branch = self.depth_branch(input)
        x = torch.cat([dwt_branch, knowledge_adaptation_branch, depth_branch], 1)
        x = self.fusion(x)
        print('XL+DWT+Depth')
        return x


class fusion_net7(nn.Module):
    def __init__(self, crop_size: int = 256, mlp_ratio: int = 4):
        super(fusion_net7, self).__init__()
        self.crop_size = crop_size
        self.mlp_ratio = mlp_ratio
        self.dwt_branch = dwt_ffc_UNet2()
        self.knowledge_adaptation_branch = DehazeXL()
        self.depth_branch = DepthNet2.DNWithPatches()
        self.fusion = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(7, 3, kernel_size=7, padding=0), nn.Tanh())

    def forward(self, input):
        dwt_branch = self.dwt_branch(input)
        knowledge_adaptation_branch = self.knowledge_adaptation_branch(input)
        depth_branch = self.depth_branch(input)
        x = torch.cat([dwt_branch, knowledge_adaptation_branch, depth_branch], 1)
        x = self.fusion(x)
        print('XL+DWT+Depth Patches')
        return x


# 创建改进的深度分支
depth_net = DepthNet2.EfficientDepthNet(
    in_channels=3,
    base_channels=32,
    num_scales=4,
    crop_size=256,
    use_patches=True,
    patch_size=256,
    batch_size=8
)


# 在fusion_net7中使用
class fusion_net8(nn.Module):
    def __init__(self, crop_size=256, mlp_ratio=4):
        super().__init__()
        self.crop_size = crop_size
        self.mlp_ratio = mlp_ratio

        # 各分支
        self.dwt_branch = dwt_ffc_UNet2()
        self.knowledge_adaptation_branch = DehazeXL()
        self.depth_branch = DepthNet2.EfficientDepthNet(
            crop_size=crop_size,
            use_patches=True,
            patch_size=crop_size,
            batch_size=8
        )

        # 融合层
        self.fusion = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(7, 3, kernel_size=7, padding=0), nn.Tanh())

    def forward(self, input):
        # 各分支并行处理
        dwt_out = self.dwt_branch(input)
        ka_out = self.knowledge_adaptation_branch(input)
        depth_out = self.depth_branch(input)

        # 融合
        x = torch.cat([dwt_out, ka_out, depth_out], dim=1)
        output = self.fusion(x)

        print("XL+DWT+优化深度分支")
        return output
class DWConv2d_BN(nn.Module):#自定义的深度可分离卷积

    def __init__(
            self,
            in_ch,
            out_ch,
            kernel_size=1,
            stride=1,
            norm_layer=nn.BatchNorm2d,
            act_layer=nn.Hardswish,
            bn_weight_init=1,
            offset_clamp=(-1,1)
    ):
        super().__init__()
        # dw
        # self.conv=torch.nn.Conv2d(in_ch,out_ch,kernel_size,stride,(kernel_size - 1) // 2,bias=False,)
       # self.mask_generator = nn.Sequential(nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=3,
       #                                                 stride=1, padding=1, bias=False, groups=in_ch),
       #                                       nn.Conv2d(in_channels=in_ch, out_channels=9,
       #                                                 kernel_size=1,
       #                                                 stride=1, padding=0, bias=False)
        #                                      )
        self.offset_clamp=offset_clamp
        self.offset_generator=nn.Sequential(nn.Conv2d(in_channels=in_ch,out_channels=in_ch,kernel_size=3,
                                                      stride= 1,padding= 1,bias= False,groups=in_ch),
                                            nn.Conv2d(in_channels=in_ch, out_channels=18,
                                                      kernel_size=1,
                                                      stride=1, padding=0, bias=False)

                                            )
        self.dcn=DeformConv2d(
                    in_channels=in_ch,
                    out_channels=in_ch,
                    kernel_size=3,
                    stride= 1,
                    padding= 1,
                    bias= False,
                    groups=in_ch
                    )#.cuda(7)
        self.pwconv = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)


        #self.bn = norm_layer(out_ch)
        self.act = act_layer() if act_layer is not None else nn.Identity()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
                # print(m)

         #   elif isinstance(m, nn.BatchNorm2d):
           #     m.weight.data.fill_(bn_weight_init)
          #      m.bias.data.zero_()

    def forward(self, x):

        # x=self.conv(x)
        #x = self.bn(x)
        #x = self.act(x)
        #mask= torch.sigmoid(self.mask_generator(x))
        #print('1')
        offset = self.offset_generator(x)
        #print('2')
        if self.offset_clamp:
            offset=torch.clamp(offset, min=self.offset_clamp[0], max=self.offset_clamp[1])#.cuda(7)1
        #print(offset)
        #print('3')
        #x=x.cuda(7)
        x = self.dcn(x,offset)
        #x=x.cpu()
        #print('4')
        x=self.pwconv(x)
       # print('5')
        #x = self.bn(x)
        x = self.act(x)
        return x
class fusion_net9(nn.Module):
    def __init__(self, crop_size: int = 256,
                 mlp_ratio: int = 4):
        super(fusion_net9, self).__init__()
        self.crop_size = crop_size
        self.mlp_ratio = mlp_ratio
        self.dwt_branch = dwt_ffc_UNet2()  # 三个通道
        self.knowledge_adaptation_branch = DehazeXL()  # 28个通道
        # self.fusion = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(6, 3, kernel_size=7, padding=0), nn.Tanh())
        self.fusion=DWConv2d_BN(6,3)


    def forward(self, input):
        dwt_branch = self.dwt_branch(input)
        knowledge_adaptation_branch = self.knowledge_adaptation_branch(input)
        x = torch.cat([dwt_branch, knowledge_adaptation_branch], 1)
        x = self.fusion(x)
        print("XL+DWT+DW卷积")
        return x
class CP_Attention_block1(nn.Module):
    def __init__(self, conv, dim, kernel_size):
        super(CP_Attention_block1, self).__init__()
        padding = kernel_size // 2  # 确保尺寸不变
        self.conv1 = conv(dim, dim, kernel_size, padding=padding, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, padding=padding, bias=True)
        self.calayer = CALayer(dim)
        self.palayer = PALayer(dim)

    def forward(self, x):
        identity = x  # 保存原始输入
        
        # 第一个卷积 + ReLU
        res = self.act1(self.conv1(x))
        res = res + identity  # 第一个残差连接
        
        # 第二个卷积
        res = self.conv2(res)
        
        # 注意力机制
        res = self.calayer(res)
        res = self.palayer(res)
        
        # 第二个残差连接
        res += identity
        
        return res


class fusion_net10(nn.Module):
    def __init__(self, crop_size: int = 256,
                 mlp_ratio: int = 4):
        super(fusion_net10, self).__init__()
        self.crop_size = crop_size
        self.mlp_ratio = mlp_ratio
        self.dwt_branch = dwt_ffc_UNet2()
        self.knowledge_adaptation_branch = DehazeXL()
        
        # 计算各层输出尺寸，确保一致性
        self.fusion = nn.Sequential(
            # 7x7 卷积，padding=3 保持尺寸
            nn.Conv2d(31, 64, kernel_size=7, padding=3, bias=True),
            nn.ReLU(inplace=True),
            
            # 多个 CP_Attention_block，保持尺寸
            CP_Attention_block1(nn.Conv2d, 64, kernel_size=3),
            CP_Attention_block1(nn.Conv2d, 64, kernel_size=3),
            
            # 输出层，3x3 卷积，padding=1 保持尺寸
            nn.Conv2d(64, 3, kernel_size=3, padding=1, bias=True),
            nn.Tanh()
        )

    def forward(self, input):
        dwt_branch = self.dwt_branch(input)
        knowledge_adaptation_branch = self.knowledge_adaptation_branch(input)
        x = torch.cat([dwt_branch, knowledge_adaptation_branch], 1)
        x = self.fusion(x)
        print("XL-28channel+DWT + CP_Attention")
        return x

class AdaptiveTriBranchFusion(nn.Module):
    """Plug-and-play tri-branch fusion with static+dynamic branch weighting."""

    def __init__(self, channels):
        super(AdaptiveTriBranchFusion, self).__init__()
        self.eps = 1e-4
        # BiFPN-style learnable positive branch weights
        self.branch_logits = nn.Parameter(torch.ones(3, dtype=torch.float32))
        # Dynamic sample-aware branch weights
        self.dynamic_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 3, channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 3, kernel_size=1, bias=True)
        )
        # Dynamic spatial-aware branch routing
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 3, kernel_size=1, bias=True)
        )
        self.global_spatial_blend = 0.5

    def forward(self, feat_dwt, feat_ka, feat_depth):
        feat_cat = torch.cat([feat_dwt, feat_ka, feat_depth], dim=1)

        static_w = F.relu(self.branch_logits)
        static_w = static_w / (static_w.sum() + self.eps)

        dynamic_w = self.dynamic_gate(feat_cat).view(feat_cat.size(0), 3)
        dynamic_w = F.softmax(dynamic_w, dim=1)
        spatial_w = F.softmax(self.spatial_gate(feat_cat), dim=1)

        # Blend static global priors and dynamic sample-aware routing.
        w1 = 0.5 * static_w[0] + 0.5 * dynamic_w[:, 0]
        w2 = 0.5 * static_w[1] + 0.5 * dynamic_w[:, 1]
        w3 = 0.5 * static_w[2] + 0.5 * dynamic_w[:, 2]

        w1 = w1.view(-1, 1, 1, 1)
        w2 = w2.view(-1, 1, 1, 1)
        w3 = w3.view(-1, 1, 1, 1)

        w1 = self.global_spatial_blend * w1 + (1.0 - self.global_spatial_blend) * spatial_w[:, 0:1]
        w2 = self.global_spatial_blend * w2 + (1.0 - self.global_spatial_blend) * spatial_w[:, 1:2]
        w3 = self.global_spatial_blend * w3 + (1.0 - self.global_spatial_blend) * spatial_w[:, 2:3]

        fused = w1 * feat_dwt + w2 * feat_ka + w3 * feat_depth
        return fused


class DepthGuidedSpatialModulation(nn.Module):
    """Use the depth branch to modulate fused features spatially."""

    def __init__(self, channels):
        super(DepthGuidedSpatialModulation, self).__init__()
        self.depth_to_attn = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, 1, kernel_size=1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.reliability_scale = 3.0

    def forward(self, fused_feat, depth_feat):
        depth_attn = self.depth_to_attn(depth_feat)

        depth_grad_x = torch.abs(depth_feat[:, :, :, 1:] - depth_feat[:, :, :, :-1])
        depth_grad_y = torch.abs(depth_feat[:, :, 1:, :] - depth_feat[:, :, :-1, :])
        depth_grad_x = F.pad(depth_grad_x, (0, 1, 0, 0), mode='replicate')
        depth_grad_y = F.pad(depth_grad_y, (0, 0, 0, 1), mode='replicate')
        depth_grad = (depth_grad_x + depth_grad_y).mean(dim=1, keepdim=True)
        depth_grad = depth_grad / (depth_grad.mean(dim=(2, 3), keepdim=True) + 1e-6)
        reliability = torch.exp(-self.reliability_scale * depth_grad)

        gated_attn = depth_attn * reliability
        return fused_feat * (1.0 + gated_attn)


class fusion_net11(nn.Module):
    def __init__(self, crop_size: int = 256,
                 mlp_ratio: int = 4):
        super(fusion_net11, self).__init__()
        self.crop_size = crop_size
        self.mlp_ratio = mlp_ratio

        # Three branches: frequency (3ch), Swin/DehazeXL (28ch), distilled depth (1ch).
        self.dwt_branch = dwt_ffc_UNet2()
        self.knowledge_adaptation_branch = DehazeXL()
        self.depth_branch = DepthNet.DN()

        # Branch projection (plug-and-play alignment to same channel width)
        embed_dim = 64
        self.proj_dwt = nn.Sequential(
            nn.Conv2d(3, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(inplace=True)
        )
        self.proj_ka = nn.Sequential(
            nn.Conv2d(28, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(inplace=True)
        )
        self.proj_depth = nn.Sequential(
            nn.Conv2d(1, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(inplace=True)
        )

        self.fusion_router = AdaptiveTriBranchFusion(embed_dim)
        self.depth_guided_modulation = DepthGuidedSpatialModulation(embed_dim)

        self.refine = nn.Sequential(
            CP_Attention_block1(nn.Conv2d, embed_dim, kernel_size=3),
            CP_Attention_block1(nn.Conv2d, embed_dim, kernel_size=3),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )

        self.to_rgb = nn.Sequential(
            nn.Conv2d(embed_dim, 32, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=3, padding=1, bias=True),
            nn.Tanh()
        )

    @staticmethod
    def _align_to_ref(x, ref):
        if x.shape[2:] != ref.shape[2:]:
            x = F.interpolate(x, size=ref.shape[2:], mode='bilinear', align_corners=False)
        return x

    def forward(self, input):
        dwt_branch = self.dwt_branch(input)
        knowledge_adaptation_branch = self.knowledge_adaptation_branch(input)
        depth_branch = self.depth_branch(input)

        knowledge_adaptation_branch = self._align_to_ref(knowledge_adaptation_branch, dwt_branch)
        depth_branch = self._align_to_ref(depth_branch, dwt_branch)

        feat_dwt = self.proj_dwt(dwt_branch)
        feat_ka = self.proj_ka(knowledge_adaptation_branch)
        feat_depth = self.proj_depth(depth_branch)

        fused = self.fusion_router(feat_dwt, feat_ka, feat_depth)
        fused = self.depth_guided_modulation(fused, feat_depth)
        fused = self.refine(fused)

        out = self.to_rgb(fused)
        print("XL+DWT+Depth+AdaptiveFusion(Net11)")
        return out
class ColorCorrectionModule(nn.Module):
    """Color Correction & Fidelity Module (CCFM)"""
    def __init__(self, with_local=True):
        super().__init__()
        self.W = nn.Parameter(torch.eye(3, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(3, dtype=torch.float32))
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.with_local = with_local
        if with_local:
            self.local_conv = nn.Sequential(
                nn.Conv2d(1, 8, 3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(8, 3, 1, bias=True),
            )

    def forward(self, x, hazy):
        # x, hazy are in [0,1], shape (B,3,H,W)
        B, C, H, W = x.shape
        x_flat = x.view(B, 3, -1).transpose(1, 2)          # (B, N, 3)
        x_affine = torch.matmul(x_flat, self.W) + self.b   # (B, N, 3)
        x_affine = x_affine.transpose(1, 2).view(B, 3, H, W).clamp(0, 1)

        if self.with_local and self.training:
            with torch.no_grad():
                gray = 0.299 * hazy[:,0] + 0.587 * hazy[:,1] + 0.114 * hazy[:,2]
                grad_x = torch.abs(gray[:,1:,:] - gray[:,:-1,:])
                grad_y = torch.abs(gray[:,:,1:] - gray[:,:,:-1])
                grad = F.pad(grad_x, (0,0,0,1), mode='replicate') + F.pad(grad_y, (0,1,0,0), mode='replicate')
                grad = grad / (grad.max() + 1e-6)
            delta = self.local_conv(grad.unsqueeze(1))
            x_local = x_affine + torch.tanh(delta) * 0.05
        else:
            x_local = x_affine

        out = x + self.alpha * (x_local - x)
        out = torch.clamp(out, 0, 1)
        return out
class Discriminator(nn.Module):  # 判别器参数量: 5,215,425 (5.22M)
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        return torch.sigmoid(self.net(x).view(batch_size))


import torch
import torch.nn as nn


def calculate_branch_flops(model, input_size=(1, 3, 256, 256), device='cuda'):
    try:
        from thop import profile, clever_format
    except ImportError as exc:
        raise ImportError("calculate_branch_flops requires the optional 'thop' package") from exc
    """
    计算网络各分支的计算量和参数
    """
    # 创建输入
    dummy_input = torch.randn(input_size).to(device)
    model = model.to(device)
    model.eval()

    results = {}

    # 计算各分支的计算量
    with torch.no_grad():
        print("=" * 80)
        print("计算各分支FLOPs和参数数量:")
        print("=" * 80)

        # 1. 计算整个网络
        total_macs, total_params = profile(model, inputs=(dummy_input,), verbose=False)
        total_flops = total_macs * 2  # MACs转换为FLOPs

        # 2. 分别计算各分支
        # DWT分支
        dwt_macs, dwt_params = profile(model.dwt_branch, inputs=(dummy_input,), verbose=False)
        dwt_flops = dwt_macs * 2

        # 知识适应分支
        ka_macs, ka_params = profile(model.knowledge_adaptation_branch, inputs=(dummy_input,), verbose=False)
        ka_flops = ka_macs * 2

        # 深度分支
        depth_macs, depth_params = profile(model.depth_branch, inputs=(dummy_input,), verbose=False)
        depth_flops = depth_macs * 2

        # 3. 计算融合层
        # 创建融合层的输入（三个分支输出的拼接）
        dwt_output = model.dwt_branch(dummy_input)
        ka_output = model.knowledge_adaptation_branch(dummy_input)
        depth_output = model.depth_branch(dummy_input)
        fusion_input = torch.cat([dwt_output, ka_output, depth_output], 1)

        fusion_macs, fusion_params = profile(model.fusion, inputs=(fusion_input,), verbose=False)
        fusion_flops = fusion_macs * 2

        # 格式化输出
        total_flops_g, total_params_m = clever_format([total_flops, total_params], "%.3f")
        dwt_flops_g, dwt_params_m = clever_format([dwt_flops, dwt_params], "%.3f")
        ka_flops_g, ka_params_m = clever_format([ka_flops, ka_params], "%.3f")
        depth_flops_g, depth_params_m = clever_format([depth_flops, depth_params], "%.3f")
        fusion_flops_g, fusion_params_m = clever_format([fusion_flops, fusion_params], "%.3f")

        # 计算占比
        total_flops_val = total_flops
        total_params_val = total_params

        print(f"\n{'网络组件':<30} {'FLOPs':<20} {'参数数量':<20} {'FLOPs占比':<15} {'参数占比':<15}")
        print("-" * 100)

        branches = [
            ("DWT分支", dwt_flops, dwt_params),
            ("知识适应分支", ka_flops, ka_params),
            ("深度分支", depth_flops, depth_params),
            ("融合层", fusion_flops, fusion_params)
        ]

        for name, flops, params in branches:
            flops_percent = (flops / total_flops_val) * 100
            params_percent = (params / total_params_val) * 100
            flops_fmt, params_fmt = clever_format([flops, params], "%.3f")
            print(f"{name:<30} {flops_fmt:<20} {params_fmt:<20} {flops_percent:.2f}%{'':<10} {params_percent:.2f}%")

        print("-" * 100)
        print(f"{'总计':<30} {total_flops_g:<20} {total_params_m:<20} 100.00%{'':<10} 100.00%")

        # 返回详细结果
        results = {
            'total': {'flops': total_flops, 'params': total_params},
            'dwt_branch': {'flops': dwt_flops, 'params': dwt_params},
            'ka_branch': {'flops': ka_flops, 'params': ka_params},
            'depth_branch': {'flops': depth_flops, 'params': depth_params},
            'fusion': {'flops': fusion_flops, 'params': fusion_params}
        }

        return results


# 使用示例
if __name__ == "__main__":
    # 假设这些类已经定义
    # from your_module import dwt_ffc_UNet2, DehazeXL, DepthNet

    # 创建模型
    model = fusion_net8()

    # 设置输入大小 (batch_size, channels, height, width)
    # 这里使用模型的crop_size
    input_size = (1, 3, model.crop_size, model.crop_size)

    # 计算计算量
    results = calculate_branch_flops(model, input_size=input_size, device='cuda:0')


class fusion_net_depth_best(nn.Module):
    """
    Final optimized model for Depth Innovation.
    Combines:
    1. Strong Backbone (ConvNeXt-XL) from fusion_net_1 (for performance)
    2. DWT Frequency Branch (for detail)
    3. Depth Estimation Branch (for geometric layout / innovation)
    4. Adaptive Fusion (for smart combination)
    """
    def __init__(
            self,
            crop_size: int = 256,
            mlp_ratio: int = 4,
            semantic_backbone: str = "lightweight_v2",
    ):
        super(fusion_net_depth_best, self).__init__()
        self.crop_size = crop_size
        self.semantic_backbone = semantic_backbone
        
        # 1. DWT Branch (Frequency) - Output: 3 channels
        self.dwt_branch = dwt_ffc_UNet2()
        
        # 2. Knowledge Adaptation (Semantic) - Output: 28 channels
        if semantic_backbone == "convnext":
            self.knowledge_adaptation_branch = knowledge_adaptation_convnext()
        elif semantic_backbone == "lightweight":
            self.knowledge_adaptation_branch = LightweightSemanticBranch(out_channels=28)
        elif semantic_backbone == "lightweight_v2":
            self.knowledge_adaptation_branch = LightweightSemanticBranchV2(out_channels=28)
        else:
            raise ValueError(f"Unsupported semantic_backbone: {semantic_backbone}")
        
        # 3. Depth Branch (Geometric) - Output: 1 channel
        self.depth_branch = DepthNet.DN()
        
        # Feature Projection to common dimension for fusion
        embed_dim = 64
        self.proj_dwt = nn.Sequential(
            nn.Conv2d(3, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )
        self.proj_ka = nn.Sequential(
            nn.Conv2d(28, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )
        self.proj_depth = nn.Sequential(
            nn.Conv2d(1, embed_dim, kernel_size=3, padding=1, bias=False), # Depth is 1 channel
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )
        self.depth_confidence_head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 2, 1, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        
        # Adaptive Fusion Module (similar to fusion_net11)
        self.fusion_router = AdaptiveTriBranchFusion(embed_dim)
        
        # Depth-Guided Modulation (Spatial Attention based on depth)
        self.depth_guided_modulation = DepthGuidedSpatialModulation(embed_dim)
        
        # Refinement
        self.refine = nn.Sequential(
            CP_Attention_block1(nn.Conv2d, embed_dim, kernel_size=3),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        
        # Reconstruction to RGB
        self.tail = nn.Sequential(
            nn.Conv2d(embed_dim, 32, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=3, padding=1, bias=True),
            nn.Tanh() # Output -1 to 1
        )

    def forward(self, input, return_depth=False, return_aux=False):
        # 1. Forward Pass Branches
        dwt_out = self.dwt_branch(input)                    # [B, 3, H, W]
        ka_out = self.knowledge_adaptation_branch(input)    # [B, 28, H, W]
        depth_out = self.depth_branch(input)                # [B, 1, H, W]
        
        # 2. Align dimensions (if needed)
        if ka_out.shape[2:] != dwt_out.shape[2:]:
            ka_out = F.interpolate(ka_out, size=dwt_out.shape[2:], mode='bilinear', align_corners=False)
        if depth_out.shape[2:] != dwt_out.shape[2:]:
            depth_out = F.interpolate(depth_out, size=dwt_out.shape[2:], mode='bilinear', align_corners=False)
            
        # 3. Project to common embedding
        feat_dwt = self.proj_dwt(dwt_out)
        feat_ka = self.proj_ka(ka_out)
        feat_depth = self.proj_depth(depth_out)
        depth_confidence = self.depth_confidence_head(feat_depth)
        
        # 4. Adaptive Fusion of the 3 branches
        fused = self.fusion_router(feat_dwt, feat_ka, feat_depth)
        
        # 5. Spatially Modulate features using Depth (Innovation Highlight)
        # Allows the model to treat far/hazy regions differently from close regions
        fused = self.depth_guided_modulation(fused, feat_depth)
        
        # 6. Refine and Reconstruct
        fused = self.refine(fused)
        out = self.tail(fused)
        
        if return_aux:
            return out, {"depth": depth_out, "confidence": depth_confidence}
        if return_depth:
            return out, depth_out
        return out

# ================== 带颜色校正的模型 ==================
class fusion_net_depth_best_color(nn.Module):
    def __init__(self, crop_size: int = 256, mlp_ratio: int = 4, use_color_correction: bool = True):
        super().__init__()
        self.crop_size = crop_size
        self.use_color_correction = use_color_correction

        # 三个分支
        self.dwt_branch = dwt_ffc_UNet2()
        self.knowledge_adaptation_branch = knowledge_adaptation_convnext()
        self.depth_branch = DepthNet.DN()   # 请确保 DepthNet.DN 已导入

        embed_dim = 64
        self.proj_dwt = nn.Sequential(
            nn.Conv2d(3, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )
        self.proj_ka = nn.Sequential(
            nn.Conv2d(28, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )
        self.proj_depth = nn.Sequential(
            nn.Conv2d(1, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )
        self.fusion_router = AdaptiveTriBranchFusion(embed_dim)
        self.depth_guided_modulation = DepthGuidedSpatialModulation(embed_dim)
        self.refine = nn.Sequential(
            CP_Attention_block1(nn.Conv2d, embed_dim, kernel_size=3),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        self.tail = nn.Sequential(
            nn.Conv2d(embed_dim, 32, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=3, padding=1, bias=True),
            nn.Tanh()
        )

        if self.use_color_correction:
            self.color_corr = ColorCorrectionModule(with_local=True)
        else:
            self.color_corr = None

    def forward(self, input, return_depth=False):
        # 输入假定在 [0,1] 范围内（未进行均值归一化）
        dwt_out = self.dwt_branch(input)
        ka_out = self.knowledge_adaptation_branch(input)
        depth_out = self.depth_branch(input)

        # 对齐尺寸
        if ka_out.shape[2:] != dwt_out.shape[2:]:
            ka_out = F.interpolate(ka_out, size=dwt_out.shape[2:], mode='bilinear', align_corners=False)
        if depth_out.shape[2:] != dwt_out.shape[2:]:
            depth_out = F.interpolate(depth_out, size=dwt_out.shape[2:], mode='bilinear', align_corners=False)

        feat_dwt = self.proj_dwt(dwt_out)
        feat_ka = self.proj_ka(ka_out)
        feat_depth = self.proj_depth(depth_out)

        fused = self.fusion_router(feat_dwt, feat_ka, feat_depth)
        fused = self.depth_guided_modulation(fused, feat_depth)
        fused = self.refine(fused)
        out = self.tail(fused)          # [-1, 1]
        out = (out + 1.0) / 2.0         # [0, 1]

        if self.use_color_correction and self.color_corr is not None:
            out = self.color_corr(out, input)

        if return_depth:
            return out, depth_out
        return out
class fusion_net_depth_best_res(nn.Module):
    def __init__(self, crop_size: int = 256, backbone: str = 'resnet34'):
        """
        backbone: 选择 'resnet34' 或 'convnext_xl'
        """
        super().__init__()
        self.crop_size = crop_size
        
        # DWT 分支（保持不变）
        self.dwt_branch = dwt_ffc_UNet2()
        
        # 知识适应分支：根据参数选择
        if backbone == 'resnet34':
            self.knowledge_adaptation_branch = knowledge_adaptation_resnet34()
        elif backbone == 'convnext_xl':
            self.knowledge_adaptation_branch = knowledge_adaptation_convnext()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # 深度分支（可选，此处先保留但可通过参数关闭）
        self.depth_branch = DepthNet.DN()
        
        # 特征投影层（与原有代码相同）
        embed_dim = 64
        self.proj_dwt = nn.Sequential(
            nn.Conv2d(3, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )
        self.proj_ka = nn.Sequential(
            nn.Conv2d(28, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )
        self.proj_depth = nn.Sequential(
            nn.Conv2d(1, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )
        
        # 自适应融合和后续模块（与原有代码相同）
        self.fusion_router = AdaptiveTriBranchFusion(embed_dim)
        self.depth_guided_modulation = DepthGuidedSpatialModulation(embed_dim)
        self.refine = nn.Sequential(
            CP_Attention_block1(nn.Conv2d, embed_dim, kernel_size=3),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        self.tail = nn.Sequential(
            nn.Conv2d(embed_dim, 32, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=3, padding=1, bias=True),
            nn.Tanh()
        )
        
    def forward(self, input, return_depth=False):
        dwt_out = self.dwt_branch(input)
        ka_out = self.knowledge_adaptation_branch(input)
        depth_out = self.depth_branch(input)
        
        # 对齐尺寸
        if ka_out.shape[2:] != dwt_out.shape[2:]:
            ka_out = F.interpolate(ka_out, size=dwt_out.shape[2:], mode='bilinear', align_corners=False)
        if depth_out.shape[2:] != dwt_out.shape[2:]:
            depth_out = F.interpolate(depth_out, size=dwt_out.shape[2:], mode='bilinear', align_corners=False)
        
        feat_dwt = self.proj_dwt(dwt_out)
        feat_ka = self.proj_ka(ka_out)
        feat_depth = self.proj_depth(depth_out)
        
        fused = self.fusion_router(feat_dwt, feat_ka, feat_depth)
        fused = self.depth_guided_modulation(fused, feat_depth)
        fused = self.refine(fused)
        out = self.tail(fused)          # [-1, 1]
        out = (out + 1.0) / 2.0         # 转换到 [0, 1]
        
        if return_depth:
            return out, depth_out
        return out


class fusion_net_depth_best_light(nn.Module):

    def __init__(
        self,
        crop_size=256,
        mlp_ratio=4
    ):

        super().__init__()

        self.crop_size = crop_size

        # =====================================
        # 1. Frequency Branch
        # =====================================
        self.dwt_branch = dwt_ffc_UNet2()

        # =====================================
        # 2. Lightweight Semantic Branch
        # =====================================
        self.knowledge_adaptation_branch = \
            LightweightSemanticBranch(
                out_channels=28
            )

        # =====================================
        # 3. Depth Branch
        # =====================================
        self.depth_branch = DepthNet.DN()

        # =====================================
        # 4. Feature Projection
        # =====================================

        embed_dim = 64

        self.proj_dwt = nn.Sequential(

            nn.Conv2d(
                3,
                embed_dim,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(embed_dim),

            nn.ReLU(inplace=True)
        )

        self.proj_ka = nn.Sequential(

            nn.Conv2d(
                28,
                embed_dim,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(embed_dim),

            nn.ReLU(inplace=True)
        )

        self.proj_depth = nn.Sequential(

            nn.Conv2d(
                1,
                embed_dim,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(embed_dim),

            nn.ReLU(inplace=True)
        )

        # =====================================
        # 5. Adaptive Fusion
        # =====================================

        self.fusion_router = \
            AdaptiveTriBranchFusion(
                embed_dim
            )

        # =====================================
        # 6. Depth Guided Modulation
        # =====================================

        self.depth_guided_modulation = \
            DepthGuidedSpatialModulation(
                embed_dim
            )

        # =====================================
        # 7. Refinement
        # =====================================

        self.refine = nn.Sequential(

            CP_Attention_block1(
                nn.Conv2d,
                embed_dim,
                kernel_size=3
            ),

            nn.Conv2d(
                embed_dim,
                embed_dim,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True)
        )

        # =====================================
        # 8. Reconstruction
        # =====================================

        self.tail = nn.Sequential(

            nn.Conv2d(
                embed_dim,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                32,
                3,
                kernel_size=3,
                padding=1
            ),

            nn.Tanh()
        )

    def forward(
        self,
        input,
        return_depth=False
    ):

        # Frequency
        dwt_out = self.dwt_branch(input)

        # Semantic
        ka_out = \
            self.knowledge_adaptation_branch(input)

        # Depth
        depth_out = \
            self.depth_branch(input)

        # Align
        if ka_out.shape[2:] != dwt_out.shape[2:]:

            ka_out = F.interpolate(
                ka_out,
                size=dwt_out.shape[2:],
                mode='bilinear',
                align_corners=False
            )

        if depth_out.shape[2:] != dwt_out.shape[2:]:

            depth_out = F.interpolate(
                depth_out,
                size=dwt_out.shape[2:],
                mode='bilinear',
                align_corners=False
            )

        # Projection
        feat_dwt = \
            self.proj_dwt(dwt_out)

        feat_ka = \
            self.proj_ka(ka_out)

        feat_depth = \
            self.proj_depth(depth_out)

        # Adaptive fusion
        fused = self.fusion_router(
            feat_dwt,
            feat_ka,
            feat_depth
        )

        # Depth-guided semantic modulation
        fused = \
            self.depth_guided_modulation(
                fused,
                feat_depth
            )

        # Refinement
        fused = self.refine(fused)

        # Reconstruction
        out = self.tail(fused)

        if return_depth:

            return out, depth_out

        return out


class fusion_net_depth_best_v2(nn.Module):
    """
    Lightweight Depth-Guided Multi-Scale Fusion Network

    Branch 1:
        DWT Frequency Branch

    Branch 2:
        Lightweight Multi-Scale Semantic Branch

    Branch 3:
        Depth Estimation Branch

    Main innovations:
        1. Lightweight semantic backbone
        2. Multi-scale semantic fusion
        3. Depth-guided semantic attention
        4. Adaptive tri-branch fusion
        5. Depth-guided spatial modulation
    """

    def __init__(
        self,
        crop_size: int = 256,
        mlp_ratio: int = 4,
        embed_dim: int = 64,
        semantic_out_channels: int = 28
    ):

        super(
            fusion_net_depth_best_v2,
            self
        ).__init__()

        self.crop_size = crop_size

        self.mlp_ratio = mlp_ratio

        self.embed_dim = embed_dim

        # =====================================
        # 1. DWT Frequency Branch
        # =====================================

        self.dwt_branch = \
            dwt_ffc_UNet2()

        # =====================================
        # 2. Lightweight Semantic Branch
        # =====================================

        self.knowledge_adaptation_branch = \
            LightweightSemanticBranchV2(
                out_channels=
                semantic_out_channels
            )

        # =====================================
        # 3. Depth Branch
        # =====================================

        self.depth_branch = \
            DepthNet.DN()

        # =====================================
        # 4. Branch Projection
        # =====================================

        self.proj_dwt = nn.Sequential(

            nn.Conv2d(
                3,
                embed_dim,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                embed_dim
            ),

            nn.SiLU(
                inplace=True
            )
        )

        self.proj_ka = nn.Sequential(

            nn.Conv2d(
                semantic_out_channels,
                embed_dim,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                embed_dim
            ),

            nn.SiLU(
                inplace=True
            )
        )

        self.proj_depth = nn.Sequential(

            nn.Conv2d(
                1,
                embed_dim,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                embed_dim
            ),

            nn.SiLU(
                inplace=True
            )
        )

        # =====================================
        # 5. Depth-Guided Semantic Attention
        # =====================================

        self.depth_guided_semantic = \
            DepthGuidedSemanticAttention(

                semantic_channels=
                semantic_out_channels,

                depth_channels=
                embed_dim
            )

        # =====================================
        # 6. Adaptive Tri-Branch Fusion
        # =====================================

        self.fusion_router = \
            AdaptiveTriBranchFusion(
                embed_dim
            )

        # =====================================
        # 7. Depth-Guided Spatial Modulation
        # =====================================

        self.depth_guided_modulation = \
            DepthGuidedSpatialModulation(
                embed_dim
            )

        # =====================================
        # 8. Refinement
        # =====================================

        self.refine = nn.Sequential(

            CP_Attention_block1(
                nn.Conv2d,
                embed_dim,
                kernel_size=3
            ),

            nn.Conv2d(
                embed_dim,
                embed_dim,
                kernel_size=3,
                padding=1,
                bias=True
            ),

            nn.SiLU(
                inplace=True
            )
        )

        # =====================================
        # 9. Reconstruction
        # =====================================

        self.tail = nn.Sequential(

            nn.Conv2d(
                embed_dim,
                32,
                kernel_size=3,
                padding=1,
                bias=True
            ),

            nn.SiLU(
                inplace=True
            ),

            nn.Conv2d(
                32,
                3,
                kernel_size=3,
                padding=1,
                bias=True
            ),

            nn.Tanh()
        )

    @staticmethod
    def _align_to_ref(
        x,
        ref
    ):

        if (
            x.shape[2:]
            != ref.shape[2:]
        ):

            x = F.interpolate(
                x,
                size=ref.shape[2:],
                mode='bilinear',
                align_corners=False
            )

        return x

    def forward(
        self,
        input,
        return_depth=False
    ):

        # =====================================
        # 1. Frequency Branch
        # =====================================

        dwt_out = \
            self.dwt_branch(
                input
            )

        # =====================================
        # 2. Semantic Branch
        # =====================================

        ka_out = \
            self.knowledge_adaptation_branch(
                input
            )

        # =====================================
        # 3. Depth Branch
        # =====================================

        depth_out = \
            self.depth_branch(
                input
            )

        # =====================================
        # 4. Spatial Alignment
        # =====================================

        ka_out = \
            self._align_to_ref(
                ka_out,
                dwt_out
            )

        depth_out = \
            self._align_to_ref(
                depth_out,
                dwt_out
            )

        # =====================================
        # 5. Project to Common Feature Space
        # =====================================

        feat_dwt = \
            self.proj_dwt(
                dwt_out
            )

        feat_ka = \
            self.proj_ka(
                ka_out
            )

        feat_depth = \
            self.proj_depth(
                depth_out
            )

        # =====================================
        # 6. Depth-Guided Semantic Attention
        #
        # Depth information enters semantic
        # branch before tri-branch fusion.
        # =====================================

        ka_out_guided = \
            self.depth_guided_semantic(
                ka_out,
                feat_depth
            )

        # Re-project enhanced semantic feature
        feat_ka = \
            self.proj_ka(
                ka_out_guided
            )

        # =====================================
        # 7. Adaptive Tri-Branch Fusion
        # =====================================

        fused = \
            self.fusion_router(

                feat_dwt,

                feat_ka,

                feat_depth
            )

        # =====================================
        # 8. Global Depth-Guided Modulation
        # =====================================

        fused = \
            self.depth_guided_modulation(

                fused,

                feat_depth
            )

        # =====================================
        # 9. Refinement
        # =====================================

        fused = \
            self.refine(
                fused
            )

        # =====================================
        # 10. Reconstruction
        # =====================================

        out = \
            self.tail(
                fused
            )

        # =====================================
        # 11. Output Range
        # =====================================

        out = (
            out + 1.0
        ) / 2.0

        # =====================================
        # 12. Return
        # =====================================

        if return_depth:

            return (
                out,
                depth_out
            )

        return out


import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. Lightweight Depthwise Separable Convolution
# ============================================================

class DSConvBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1
    ):
        super().__init__()

        self.dwconv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(
            in_channels
        )

        self.act1 = nn.SiLU(
            inplace=True
        )

        self.pwconv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(
            out_channels
        )

        self.act2 = nn.SiLU(
            inplace=True
        )

    def forward(self, x):

        x = self.dwconv(x)

        x = self.bn1(x)

        x = self.act1(x)

        x = self.pwconv(x)

        x = self.bn2(x)

        x = self.act2(x)

        return x


# ============================================================
# 2. Lightweight Context Block
# ============================================================

class LightweightContextBlock(nn.Module):

    def __init__(
        self,
        channels
    ):
        super().__init__()

        # Local branch
        self.local_branch = nn.Sequential(

            nn.Conv2d(
                channels,
                channels,
                kernel_size=5,
                padding=2,
                groups=channels,
                bias=False
            ),

            nn.BatchNorm2d(
                channels
            ),

            nn.SiLU(
                inplace=True
            )
        )

        # Large receptive field branch
        self.context_branch = nn.Sequential(

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=3,
                dilation=3,
                groups=channels,
                bias=False
            ),

            nn.BatchNorm2d(
                channels
            ),

            nn.SiLU(
                inplace=True
            )
        )

        self.fusion = nn.Sequential(

            nn.Conv2d(
                channels * 2,
                channels,
                kernel_size=1,
                bias=False
            ),

            nn.BatchNorm2d(
                channels
            ),

            nn.SiLU(
                inplace=True
            )
        )

        hidden = max(
            channels // 4,
            8
        )

        self.attention = nn.Sequential(

            nn.AdaptiveAvgPool2d(1),

            nn.Conv2d(
                channels,
                hidden,
                kernel_size=1
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                hidden,
                channels,
                kernel_size=1
            ),

            nn.Sigmoid()
        )

    def forward(self, x):

        identity = x

        local = self.local_branch(x)

        context = self.context_branch(x)

        feat = torch.cat(
            [
                local,
                context
            ],
            dim=1
        )

        feat = self.fusion(feat)

        attn = self.attention(feat)

        feat = feat * attn

        return feat + identity


# ============================================================
# 3. Multi-scale Lightweight Semantic Encoder
# ============================================================

class LightweightSemanticEncoderV3(nn.Module):
    """
    Multi-scale lightweight semantic encoder.

    Output:

        F1 : 1/4
        F2 : 1/8
        F3 : 1/16
        F4 : 1/32

    Channels:

        F1 : 48
        F2 : 96
        F3 : 160
        F4 : 192
    """

    def __init__(self):

        super().__init__()

        # ------------------------------------------------
        # Stem
        # ------------------------------------------------

        self.stem = nn.Sequential(

            nn.Conv2d(
                3,
                32,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                32
            ),

            nn.SiLU(
                inplace=True
            )
        )

        # ------------------------------------------------
        # Stage 1
        # 1/4
        # ------------------------------------------------

        self.stage1 = nn.Sequential(

            DSConvBlock(
                32,
                48,
                stride=2
            ),

            LightweightContextBlock(
                48
            )
        )

        # ------------------------------------------------
        # Stage 2
        # 1/8
        # ------------------------------------------------

        self.stage2 = nn.Sequential(

            DSConvBlock(
                48,
                96,
                stride=2
            ),

            LightweightContextBlock(
                96
            ),

            LightweightContextBlock(
                96
            )
        )

        # ------------------------------------------------
        # Stage 3
        # 1/16
        # ------------------------------------------------

        self.stage3 = nn.Sequential(

            DSConvBlock(
                96,
                160,
                stride=2
            ),

            LightweightContextBlock(
                160
            ),

            LightweightContextBlock(
                160
            )
        )

        # ------------------------------------------------
        # Stage 4
        # 1/32
        # ------------------------------------------------

        self.stage4 = nn.Sequential(

            DSConvBlock(
                160,
                192,
                stride=2
            ),

            LightweightContextBlock(
                192
            ),

            LightweightContextBlock(
                192
            )
        )

    def forward(self, x):

        x = self.stem(x)

        f1 = self.stage1(x)

        f2 = self.stage2(f1)

        f3 = self.stage3(f2)

        f4 = self.stage4(f3)

        return f1, f2, f3, f4


# ============================================================
# 4. Multi-scale Depth Feature Adapter
# ============================================================

class MultiScaleDepthAdapterV3(nn.Module):
    """
    从原始 DN 的 Encoder 中间特征中提取
    与 Semantic Encoder 对应的多尺度 Depth Features.

    原始 DN:

        conv2 -> 1/4  : 48 channels
        conv3 -> 1/8  : 96 channels
        conv4 -> 1/16 : 128 channels
        DRDB5 -> 1/16 : 128 channels

    为了不破坏原 DN 的结构，
    不增加新的下采样层。
    """

    def __init__(self):

        super().__init__()

        # 1/4
        self.proj1 = nn.Sequential(

            nn.Conv2d(
                48,
                48,
                kernel_size=1,
                bias=False
            ),

            nn.BatchNorm2d(
                48
            ),

            nn.SiLU(
                inplace=True
            )
        )

        # 1/8
        self.proj2 = nn.Sequential(

            nn.Conv2d(
                96,
                96,
                kernel_size=1,
                bias=False
            ),

            nn.BatchNorm2d(
                96
            ),

            nn.SiLU(
                inplace=True
            )
        )

        # 1/16
        self.proj3 = nn.Sequential(

            nn.Conv2d(
                128,
                160,
                kernel_size=1,
                bias=False
            ),

            nn.BatchNorm2d(
                160
            ),

            nn.SiLU(
                inplace=True
            )
        )

        # 1/16 deep feature
        self.proj4 = nn.Sequential(

            nn.Conv2d(
                128,
                192,
                kernel_size=1,
                bias=False
            ),

            nn.BatchNorm2d(
                192
            ),

            nn.SiLU(
                inplace=True
            )
        )

    def forward(
        self,
        d2,
        d3,
        d4,
        d5
    ):

        depth1 = self.proj1(
            d2
        )

        depth2 = self.proj2(
            d3
        )

        depth3 = self.proj3(
            d4
        )

        depth4 = self.proj4(
            d5
        )

        return (
            depth1,
            depth2,
            depth3,
            depth4
        )


# ============================================================
# 5. Multi-scale Depth-Guided Semantic Attention
# ============================================================

class MSDGSA(nn.Module):
    """
    Multi-Scale Depth-Guided Semantic Attention.

    Semantic:
        F_i

    Depth:
        D_i

    Interaction:
        Z_i = phi_s(F_i) + phi_d(D_i)

    Attention:
        A_i = sigmoid(psi(Z_i))

    Output:
        F'_i = F_i * (1 + alpha * A_i)
    """

    def __init__(
        self,
        semantic_channels,
        depth_channels,
        reduction=4
    ):

        super().__init__()

        hidden_channels = max(
            semantic_channels // reduction,
            8
        )

        # Semantic projection
        self.semantic_proj = nn.Sequential(

            nn.Conv2d(
                semantic_channels,
                hidden_channels,
                kernel_size=1,
                bias=False
            ),

            nn.BatchNorm2d(
                hidden_channels
            ),

            nn.SiLU(
                inplace=True
            )
        )

        # Depth projection
        self.depth_proj = nn.Sequential(

            nn.Conv2d(
                depth_channels,
                hidden_channels,
                kernel_size=1,
                bias=False
            ),

            nn.BatchNorm2d(
                hidden_channels
            ),

            nn.SiLU(
                inplace=True
            )
        )

        # Cross-modal interaction
        self.interaction = nn.Sequential(

            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                groups=hidden_channels,
                bias=False
            ),

            nn.BatchNorm2d(
                hidden_channels
            ),

            nn.SiLU(
                inplace=True
            ),

            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=1,
                bias=False
            ),

            nn.BatchNorm2d(
                hidden_channels
            ),

            nn.SiLU(
                inplace=True
            )
        )

        # Spatial attention
        self.attention = nn.Sequential(

            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                groups=hidden_channels,
                bias=False
            ),

            nn.BatchNorm2d(
                hidden_channels
            ),

            nn.SiLU(
                inplace=True
            ),

            nn.Conv2d(
                hidden_channels,
                1,
                kernel_size=1,
                bias=True
            ),

            nn.Sigmoid()
        )

        # Learnable guidance strength
        self.alpha = nn.Parameter(
            torch.tensor(0.5)
        )

    def forward(
        self,
        semantic_feat,
        depth_feat
    ):

        if (
            semantic_feat.shape[2:]
            != depth_feat.shape[2:]
        ):

            depth_feat = F.interpolate(
                depth_feat,
                size=semantic_feat.shape[2:],
                mode='bilinear',
                align_corners=False
            )

        semantic_embed = \
            self.semantic_proj(
                semantic_feat
            )

        depth_embed = \
            self.depth_proj(
                depth_feat
            )

        interaction = (
            semantic_embed
            +
            depth_embed
        )

        interaction = \
            self.interaction(
                interaction
            )

        attention = \
            self.attention(
                interaction
            )

        enhanced = semantic_feat * (
            1.0
            +
            self.alpha
            *
            attention
        )

        return enhanced


# ============================================================
# 6. Cross-scale Semantic Fusion
# ============================================================

class CrossScaleSemanticFusionV3(nn.Module):

    def __init__(
        self,
        channels_list=(
            48,
            96,
            160,
            192
        ),
        out_channels=64
    ):

        super().__init__()

        self.proj1 = nn.Sequential(

            nn.Conv2d(
                channels_list[0],
                out_channels,
                kernel_size=1,
                bias=False
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.SiLU(
                inplace=True
            )
        )

        self.proj2 = nn.Sequential(

            nn.Conv2d(
                channels_list[1],
                out_channels,
                kernel_size=1,
                bias=False
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.SiLU(
                inplace=True
            )
        )

        self.proj3 = nn.Sequential(

            nn.Conv2d(
                channels_list[2],
                out_channels,
                kernel_size=1,
                bias=False
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.SiLU(
                inplace=True
            )
        )

        self.proj4 = nn.Sequential(

            nn.Conv2d(
                channels_list[3],
                out_channels,
                kernel_size=1,
                bias=False
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.SiLU(
                inplace=True
            )
        )

        self.fusion = nn.Sequential(

            nn.Conv2d(
                out_channels * 4,
                out_channels,
                kernel_size=1,
                bias=False
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.SiLU(
                inplace=True
            ),

            LightweightContextBlock(
                out_channels
            )
        )

    def forward(
        self,
        f1,
        f2,
        f3,
        f4
    ):

        target_size = f1.shape[2:]

        p1 = self.proj1(
            f1
        )

        p2 = self.proj2(
            f2
        )

        p3 = self.proj3(
            f3
        )

        p4 = self.proj4(
            f4
        )

        p2 = F.interpolate(
            p2,
            size=target_size,
            mode='bilinear',
            align_corners=False
        )

        p3 = F.interpolate(
            p3,
            size=target_size,
            mode='bilinear',
            align_corners=False
        )

        p4 = F.interpolate(
            p4,
            size=target_size,
            mode='bilinear',
            align_corners=False
        )

        feat = torch.cat(
            [
                p1,
                p2,
                p3,
                p4
            ],
            dim=1
        )

        feat = self.fusion(
            feat
        )

        return feat


# ============================================================
# 7. Dynamic Semantic Gating
# ============================================================

class DynamicSemanticGatingV3(nn.Module):
    """
    动态决定：

        Local Texture
        Global Semantic
        Depth Geometry

    三者权重。
    """

    def __init__(
        self,
        channels
    ):

        super().__init__()

        hidden = max(
            channels // 4,
            16
        )

        self.pool = \
            nn.AdaptiveAvgPool2d(1)

        self.mlp = nn.Sequential(

            nn.Conv2d(
                channels * 2,
                hidden,
                kernel_size=1
            ),

            nn.SiLU(
                inplace=True
            ),

            nn.Conv2d(
                hidden,
                3,
                kernel_size=1
            )
        )

    def forward(
        self,
        semantic_feat,
        depth_feat
    ):

        semantic_global = \
            self.pool(
                semantic_feat
            )

        depth_global = \
            self.pool(
                depth_feat
            )

        x = torch.cat(
            [
                semantic_global,
                depth_global
            ],
            dim=1
        )

        weights = self.mlp(
            x
        )

        weights = F.softmax(
            weights,
            dim=1
        )

        return weights


# ============================================================
# 8. Dynamic Semantic Fusion
# ============================================================

class DynamicSemanticFusionV3(nn.Module):

    def __init__(
        self,
        channels=64
    ):

        super().__init__()

        # Local texture
        self.local_branch = nn.Sequential(

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False
            ),

            nn.BatchNorm2d(
                channels
            ),

            nn.SiLU(
                inplace=True
            ),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=1
            )
        )

        # Global context
        self.global_branch = nn.Sequential(

            nn.Conv2d(
                channels,
                channels,
                kernel_size=5,
                padding=2,
                groups=channels,
                bias=False
            ),

            nn.BatchNorm2d(
                channels
            ),

            nn.SiLU(
                inplace=True
            ),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=1
            )
        )

        # Depth geometry
        self.depth_branch = nn.Sequential(

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=2,
                dilation=2,
                groups=channels,
                bias=False
            ),

            nn.BatchNorm2d(
                channels
            ),

            nn.SiLU(
                inplace=True
            ),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=1
            )
        )

        self.gate = \
            DynamicSemanticGatingV3(
                channels
            )

    def forward(
        self,
        semantic,
        depth
    ):

        local_feat = \
            self.local_branch(
                semantic
            )

        global_feat = \
            self.global_branch(
                semantic
            )

        depth_feat = \
            self.depth_branch(
                depth
            )

        weights = self.gate(
            semantic,
            depth
        )

        w_local = weights[
            :,
            0:1
        ]

        w_global = weights[
            :,
            1:2
        ]

        w_depth = weights[
            :,
            2:3
        ]

        output = (

            w_local
            *
            local_feat

            +

            w_global
            *
            global_feat

            +

            w_depth
            *
            depth_feat
        )

        return (
            output,
            weights
        )


# ============================================================
# 9. V3 Semantic Branch
# ============================================================

class LightweightSemanticBranchV3(nn.Module):
    """
    V3 Semantic Branch

    Lightweight Encoder
            ↓
    Multi-scale Semantic Features
            ↓
    MS-DGSA × 4
            ↓
    Cross-scale Fusion
            ↓
    Dynamic Semantic Gating
            ↓
    28-channel output
    """

    def __init__(
        self,
        out_channels=28
    ):

        super().__init__()

        self.encoder = \
            LightweightSemanticEncoderV3()

        # ==========================================
        # MS-DGSA
        # ==========================================

        self.dgsa1 = DepthGuidedSemanticAttention(
    semantic_channels=48,
    depth_channels=48
)

        self.dgsa2 = DepthGuidedSemanticAttention(
    semantic_channels=96,
    depth_channels=96
)

        self.dgsa3 = DepthGuidedSemanticAttention(
    semantic_channels=160,
    depth_channels=160
)

        self.dgsa4 = DepthGuidedSemanticAttention(
    semantic_channels=192,
    depth_channels=192
)

        # ==========================================
        # Cross-scale Fusion
        # ==========================================

        self.cross_scale_fusion = \
            CrossScaleSemanticFusionV3(
                channels_list=[
                    48,
                    96,
                    160,
                    192
                ],
                out_channels=64
            )

        # ==========================================
        # Dynamic Semantic Fusion
        # ==========================================

        self.dynamic_fusion = \
            DynamicSemanticFusionV3(
                channels=64
            )

        # ==========================================
        # Output
        # ==========================================

        self.output_proj = nn.Sequential(

            nn.Conv2d(
                64,
                48,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                48
            ),

            nn.SiLU(
                inplace=True
            ),

            nn.Conv2d(
                48,
                out_channels,
                kernel_size=3,
                padding=1
            )
        )

    def forward(
        self,
        x,
        depth_features,
        return_aux=False
    ):

        # ==========================================
        # Semantic Encoder
        # ==========================================

        f1, f2, f3, f4 = \
            self.encoder(x)

        # ==========================================
        # Multi-scale Depth Guidance
        # ==========================================

        d1 = depth_features[
            'd1'
        ]

        d2 = depth_features[
            'd2'
        ]

        d3 = depth_features[
            'd3'
        ]

        d4 = depth_features[
            'd4'
        ]

        f1 = self.dgsa1(
            f1,
            d1
        )

        f2 = self.dgsa2(
            f2,
            d2
        )

        f3 = self.dgsa3(
            f3,
            d3
        )

        f4 = self.dgsa4(
            f4,
            d4
        )

        # ==========================================
        # Cross-scale Fusion
        # ==========================================

        semantic = \
            self.cross_scale_fusion(
                f1,
                f2,
                f3,
                f4
            )

        # ==========================================
        # Depth for Dynamic Semantic Fusion
        # ==========================================

        depth_for_gate = d1

        depth_for_gate = F.interpolate(
            depth_for_gate,
            size=semantic.shape[2:],
            mode='bilinear',
            align_corners=False
        )

        depth_for_gate = \
            self.cross_scale_fusion.proj1(
                depth_for_gate
            )

        # ==========================================
        # Dynamic Semantic Fusion
        # ==========================================

        semantic, gate_weights = \
            self.dynamic_fusion(
                semantic,
                depth_for_gate
            )

        # ==========================================
        # Output
        # ==========================================

        out = self.output_proj(
            semantic
        )

        out = F.interpolate(
            out,
            size=x.shape[2:],
            mode='bilinear',
            align_corners=False
        )

        if return_aux:

            return (
                out,
                {
                    'f1': f1,
                    'f2': f2,
                    'f3': f3,
                    'f4': f4,
                    'gate_weights':
                        gate_weights
                }
            )

        return out


# ============================================================
# 10. V3 Main Network
# ============================================================

class fusion_net_depth_best_v3(nn.Module):
    """
    Fusion-Net-Depth-V3

    Frequency Branch:
        DWT-FFC UNet

    Semantic Branch:
        Lightweight Multi-scale Semantic Encoder
        +
        Multi-scale Depth-Guided Semantic Attention
        +
        Cross-scale Semantic Fusion
        +
        Dynamic Semantic Gating

    Depth Branch:
        Original DRDB DepthNet

    Fusion:
        Adaptive Tri-Branch Fusion
        +
        Depth-Guided Spatial Modulation
        +
        CP Attention Refinement
    """

    def __init__(
        self,
        crop_size=256,
        mlp_ratio=4,
        embed_dim=64,
        semantic_out_channels=28
    ):

        super().__init__()

        self.crop_size = \
            crop_size

        self.mlp_ratio = \
            mlp_ratio

        self.embed_dim = \
            embed_dim

        # ==========================================
        # 1. DWT Frequency Branch
        # ==========================================

        self.dwt_branch = \
            dwt_ffc_UNet2()

        # ==========================================
        # 2. Original Depth Branch
        # ==========================================

        self.depth_branch = \
            DepthNet.DN1()

        # ==========================================
        # 3. Depth Feature Adapters
        # ==========================================

        self.depth_feature_adapter = \
            MultiScaleDepthAdapterV3()

        # ==========================================
        # 4. Lightweight Semantic Branch
        # ==========================================

        self.knowledge_adaptation_branch = \
            LightweightSemanticBranchV3(
                out_channels=
                semantic_out_channels
            )

        # ==========================================
        # 5. Projection
        # ==========================================

        self.proj_dwt = nn.Sequential(

            nn.Conv2d(
                3,
                embed_dim,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                embed_dim
            ),

            nn.SiLU(
                inplace=True
            )
        )

        self.proj_ka = nn.Sequential(

            nn.Conv2d(
                semantic_out_channels,
                embed_dim,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                embed_dim
            ),

            nn.SiLU(
                inplace=True
            )
        )

        self.proj_depth = nn.Sequential(

            nn.Conv2d(
                1,
                embed_dim,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                embed_dim
            ),

            nn.SiLU(
                inplace=True
            )
        )

        # ==========================================
        # 6. Adaptive Tri-Branch Fusion
        # ==========================================

        self.fusion_router = \
            AdaptiveTriBranchFusion(
                embed_dim
            )

        # ==========================================
        # 7. Depth Guided Spatial Modulation
        # ==========================================

        self.depth_guided_modulation = \
            DepthGuidedSpatialModulation(
                embed_dim
            )

        # ==========================================
        # 8. Refinement
        # ==========================================

        self.refine = nn.Sequential(

            CP_Attention_block1(
                nn.Conv2d,
                embed_dim,
                kernel_size=3
            ),

            nn.Conv2d(
                embed_dim,
                embed_dim,
                kernel_size=3,
                padding=1,
                bias=True
            ),

            nn.SiLU(
                inplace=True
            )
        )

        # ==========================================
        # 9. Reconstruction
        # ==========================================

        self.tail = nn.Sequential(

            nn.Conv2d(
                embed_dim,
                32,
                kernel_size=3,
                padding=1,
                bias=True
            ),

            nn.SiLU(
                inplace=True
            ),

            nn.Conv2d(
                32,
                3,
                kernel_size=3,
                padding=1,
                bias=True
            ),

            nn.Tanh()
        )

    @staticmethod
    def _align_to_ref(
        x,
        ref
    ):

        if (
            x.shape[2:]
            != ref.shape[2:]
        ):

            x = F.interpolate(
                x,
                size=ref.shape[2:],
                mode='bilinear',
                align_corners=False
            )

        return x

    def forward(
        self,
        input,
        return_depth=False,
        return_aux=False
    ):

        # ==========================================
        # 1. Depth Branch
        # ==========================================

        depth_out, depth_features = \
            self.depth_branch(
                input,
                return_features=True
            )

        # ==========================================
        # 2. Adapt Depth Features
        # ==========================================

        d1, d2, d3, d4 = \
            self.depth_feature_adapter(

                depth_features['d1'],

                depth_features['d2'],

                depth_features['d3'],

                depth_features['d4']
            )

        depth_features_v3 = {

            'd1': d1,

            'd2': d2,

            'd3': d3,

            'd4': d4
        }

        # ==========================================
        # 3. DWT Branch
        # ==========================================

        dwt_out = \
            self.dwt_branch(
                input
            )

        # ==========================================
        # 4. Semantic Branch
        # ==========================================

        if return_aux:

            ka_out, semantic_aux = \
                self.knowledge_adaptation_branch(

                    input,

                    depth_features_v3,

                    return_aux=True
                )

        else:

            ka_out = \
                self.knowledge_adaptation_branch(

                    input,

                    depth_features_v3
                )

            semantic_aux = None

        # ==========================================
        # 5. Align
        # ==========================================

        ka_out = \
            self._align_to_ref(
                ka_out,
                dwt_out
            )

        depth_out = \
            self._align_to_ref(
                depth_out,
                dwt_out
            )

        # ==========================================
        # 6. Projection
        # ==========================================

        feat_dwt = \
            self.proj_dwt(
                dwt_out
            )

        feat_ka = \
            self.proj_ka(
                ka_out
            )

        feat_depth = \
            self.proj_depth(
                depth_out
            )

        # ==========================================
        # 7. Adaptive Tri-Branch Fusion
        # ==========================================

        fused = \
            self.fusion_router(

                feat_dwt,

                feat_ka,

                feat_depth
            )

        # ==========================================
        # 8. Depth-Guided Spatial Modulation
        # ==========================================

        fused = \
            self.depth_guided_modulation(

                fused,

                feat_depth
            )

        # ==========================================
        # 9. CP Attention Refinement
        # ==========================================

        fused = \
            self.refine(
                fused
            )

        # ==========================================
        # 10. Reconstruction
        # ==========================================

        out = \
            self.tail(
                fused
            )

        # [-1, 1] -> [0, 1]
        out = (
            out + 1.0
        ) / 2.0

        # ==========================================
        # 11. Return
        # ==========================================

        if return_aux:

            return (
                out,
                depth_out,
                {
                    'semantic':
                        semantic_aux,

                    'depth_features':
                        depth_features_v3
                }
            )

        if return_depth:

            return (
                out,
                depth_out
            )

        return out


class fusion_net_depth_geometry_v1(nn.Module):
    """Reliability-gated depth-assisted dehazing model.

    The depth map keeps an explicit relative-disparity meaning. Learned geometry
    features may help restoration only through a confidence-gated residual whose
    strength starts at zero, so an untrained depth branch cannot dominate fusion.
    """

    def __init__(self, crop_size=256, embed_dim=64):
        super().__init__()
        self.crop_size = crop_size
        self.dwt_branch = dwt_ffc_UNet2()
        self.knowledge_adaptation_branch = NightContextBranch(out_channels=28)
        self.depth_branch = DepthGeometryBranch(geometry_channels=32)

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
        semantic = self._resize(self.knowledge_adaptation_branch(image), target_size)
        depth = self.depth_branch(image)

        base = self.base_fusion(torch.cat((self.proj_dwt(dwt), self.proj_ka(semantic)), dim=1))
        geometry = self.proj_depth(torch.cat((depth["geometry"], depth["disparity"]), dim=1))
        correction = self.depth_adapter(torch.cat((base, geometry), dim=1))
        fused = base + torch.tanh(self.depth_scale) * depth["confidence"] * correction
        residual = 0.25 * self.tail(self.refine(fused))
        restored = torch.clamp(image + residual, 0.0, 1.0)

        if return_aux:
            return restored, {
                "depth": depth["disparity"],
                "confidence": depth["confidence"],
                "depth_scale": torch.tanh(self.depth_scale),
            }
        if return_depth:
            return restored, depth["disparity"]
        return restored
