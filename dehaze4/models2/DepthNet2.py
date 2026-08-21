import torch
import torch.nn as nn
import torch.nn.functional as F


class DRDB(nn.Module):
    def __init__(self, in_ch, growth_rate=32):
        super(DRDB, self).__init__()
        in_ch_ = in_ch
        self.Dcov1 = nn.Conv2d(in_ch_, growth_rate, 3, padding=2, dilation=2)
        in_ch_ += growth_rate
        self.Dcov2 = nn.Conv2d(in_ch_, growth_rate, 3, padding=2, dilation=2)
        in_ch_ += growth_rate
        self.Dcov3 = nn.Conv2d(in_ch_, growth_rate, 3, padding=2, dilation=2)
        in_ch_ += growth_rate
        self.Dcov4 = nn.Conv2d(in_ch_, growth_rate, 3, padding=2, dilation=2)
        in_ch_ += growth_rate
        self.Dcov5 = nn.Conv2d(in_ch_, growth_rate, 3, padding=2, dilation=2)
        in_ch_ += growth_rate
        self.conv = nn.Conv2d(in_ch_, in_ch, 1, padding=0)

    def forward(self, x):
        x1 = self.Dcov1(x)
        x1 = F.relu(x1)
        x1 = torch.cat([x, x1], dim=1)

        x2 = self.Dcov2(x1)
        x2 = F.relu(x2)
        x2 = torch.cat([x1, x2], dim=1)

        x3 = self.Dcov3(x2)
        x3 = F.relu(x3)
        x3 = torch.cat([x2, x3], dim=1)

        x4 = self.Dcov4(x3)
        x4 = F.relu(x4)
        x4 = torch.cat([x3, x4], dim=1)

        x5 = self.Dcov5(x4)
        x5 = F.relu(x5)
        x5 = torch.cat([x4, x5], dim=1)

        x6 = self.conv(x5)
        out = x + F.relu(x6)
        return out


class PatchProcessor:
    """图像分块处理器"""

    def __init__(self, patch_size=128, overlap=32):
        """
        Args:
            patch_size: 分块大小
            overlap: 块之间的重叠区域大小
        """
        self.patch_size = patch_size
        self.overlap = overlap

    def split_to_patches(self, x):
        """
        将输入图像分割成重叠的块
        """
        B, C, H, W = x.shape

        # 计算需要分多少块
        stride = self.patch_size - self.overlap * 2
        num_h = (H - self.overlap * 2 + stride - 1) // stride
        num_w = (W - self.overlap * 2 + stride - 1) // stride

        patches = []
        patch_positions = []

        for h_idx in range(num_h):
            for w_idx in range(num_w):
                # 计算块的起始和结束位置
                h_start = h_idx * stride
                w_start = w_idx * stride
                h_end = min(h_start + self.patch_size, H)
                w_end = min(w_start + self.patch_size, W)

                # 调整起始位置以确保最后一个块大小合适
                if h_end == H:
                    h_start = max(0, H - self.patch_size)
                if w_end == W:
                    w_start = max(0, W - self.patch_size)

                # 提取块
                patch = x[:, :, h_start:h_end, w_start:w_end]

                # 如果块大小不够，填充
                if patch.shape[2] < self.patch_size or patch.shape[3] < self.patch_size:
                    pad_h = self.patch_size - patch.shape[2]
                    pad_w = self.patch_size - patch.shape[3]
                    patch = F.pad(patch, (0, pad_w, 0, pad_h), mode='reflect')

                patches.append(patch)
                patch_positions.append((h_start, w_start, h_end, w_end))

        return patches, patch_positions, (H, W)

    def merge_patches(self, patches, patch_positions, original_size):
        """
        将处理后的块合并回原图大小
        """
        H, W = original_size
        B, C, _, _ = patches[0].shape

        # 创建权重矩阵用于混合重叠区域
        device = patches[0].device
        dtype = patches[0].dtype
        weight = torch.ones((self.patch_size, self.patch_size),
                            device=device, dtype=dtype)

        # 创建边缘权重（使用余弦函数平滑过渡）
        # 修复：确保传递给torch.cos的是张量
        for i in range(self.overlap):
            # 创建张量而不是使用标量
            i_tensor = torch.tensor(i, device=device, dtype=dtype)
            alpha = torch.pi * i_tensor / self.overlap
            cos_val = torch.cos(alpha)

            weight_val = 0.5 * (1 - cos_val)

            weight[:, i] *= weight_val
            weight[:, -i - 1] *= weight_val
            weight[i, :] *= weight_val
            weight[-i - 1, :] *= weight_val

        # 初始化输出和权重累积
        output = torch.zeros((B, C, H, W), device=device, dtype=dtype)
        weight_acc = torch.zeros((B, C, H, W), device=device, dtype=dtype)

        for patch, (h_start, w_start, h_end, w_end) in zip(patches, patch_positions):
            # 裁剪掉填充部分
            patch_h = min(self.patch_size, h_end - h_start)
            patch_w = min(self.patch_size, w_end - w_start)

            patch_cropped = patch[:, :, :patch_h, :patch_w]
            weight_cropped = weight[:patch_h, :patch_w].unsqueeze(0).unsqueeze(0)

            # 累加到输出
            output[:, :, h_start:h_end, w_start:w_end] += patch_cropped * weight_cropped
            weight_acc[:, :, h_start:h_end, w_start:w_end] += weight_cropped

        # 防止除以零
        weight_acc = torch.where(weight_acc > 0, weight_acc, torch.ones_like(weight_acc))
        output = output / weight_acc

        return output


class DNWithPatches(nn.Module):
    """
    支持分块处理的去噪网络
    可以设置使用或不使用分块
    """

    def __init__(self, use_patches=True, patch_size=128, overlap=32):
        super(DNWithPatches, self).__init__()
        self.use_patches = use_patches
        self.patch_processor = PatchProcessor(patch_size, overlap) if use_patches else None

        # 原始网络结构
        self.DRDB_layer1 = DRDB(in_ch=3, growth_rate=32)
        self.conv1 = nn.Conv2d(3, 24, 3, 2, 1)
        self.DRDB_layer2 = DRDB(in_ch=24, growth_rate=32)
        self.conv2 = nn.Conv2d(24, 48, 3, 2, 1)
        self.DRDB_layer3 = DRDB(in_ch=48, growth_rate=32)
        self.conv3 = nn.Conv2d(48, 96, 3, 2, 1)
        self.DRDB_layer4 = DRDB(in_ch=96, growth_rate=32)
        self.conv4 = nn.Conv2d(96, 128, 3, 2, 1)

        self.DRDB_layer5 = DRDB(in_ch=128, growth_rate=32)
        self.up1 = nn.ConvTranspose2d(128, 96, kernel_size=4, stride=2, padding=1)
        self.DRDB_layer6 = DRDB(in_ch=96, growth_rate=32)
        self.up2 = nn.ConvTranspose2d(96, 48, kernel_size=4, stride=2, padding=1)
        self.DRDB_layer7 = DRDB(in_ch=48, growth_rate=32)
        self.up3 = nn.ConvTranspose2d(48, 24, kernel_size=4, stride=2, padding=1)
        self.DRDB_layer8 = DRDB(in_ch=24, growth_rate=32)
        self.up4 = nn.ConvTranspose2d(24, 3, kernel_size=4, stride=2, padding=1)
        self.final_conv = nn.Conv2d(3, 1, 1)

    def forward(self, x):
        # 修复：调整分块条件，避免对较小图像分块
        if self.use_patches and self.patch_processor is not None and x.shape[2] > 256 and x.shape[3] > 256:
            # 分块处理模式
            patches, positions, original_size = self.patch_processor.split_to_patches(x)
            processed_patches = []

            for i, patch in enumerate(patches):
                # 处理每个块
                processed_patch = self._forward_single(patch)
                processed_patches.append(processed_patch)

                # 可选：每处理几个块就清理一次缓存
                if (i + 1) % 4 == 0:
                    torch.cuda.empty_cache()

            # 合并块
            output = self.patch_processor.merge_patches(processed_patches, positions, original_size)
            return output
        else:
            # 正常处理模式（小图像）
            return self._forward_single(x)

    def _forward_single(self, x):
        """处理单个图像或块"""
        x1 = self.DRDB_layer1(x)
        x1 = self.conv1(x1)
        x1 = self.DRDB_layer2(x1)
        x1 = self.conv2(x1)
        x1 = self.DRDB_layer3(x1)
        x1 = self.conv3(x1)
        x1 = self.DRDB_layer4(x1)
        x1 = self.conv4(x1)
        x1 = self.DRDB_layer5(x1)
        x1 = self.up1(x1)
        x1 = self.DRDB_layer6(x1)
        x1 = self.up2(x1)
        x1 = self.DRDB_layer7(x1)
        x1 = self.up3(x1)
        x1 = self.DRDB_layer8(x1)
        x1 = self.up4(x1)
        x1 = self.final_conv(x1)
        return x1


import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce


class EfficientDepthNet(nn.Module):
    """
    高效深度估计网络，借鉴DehazeXL的设计思想
    """

    def __init__(
            self,
            in_channels=3,
            base_channels=32,
            num_scales=4,
            crop_size=256,
            mlp_ratio=4,
            use_patches=True,
            patch_size=256,
            batch_size=8
    ):
        super().__init__()
        self.crop_size = crop_size
        self.use_patches = use_patches
        self.patch_size = patch_size
        self.batch_size = batch_size

        # 1. 轻量级编码器（类似DehazeXL的encoder）
        self.encoder = nn.ModuleList()
        current_channels = in_channels
        self.scale_factors = []

        for i in range(num_scales):
            # 每个尺度包含轻量级DRDB
            layer = nn.Sequential(
                nn.Conv2d(current_channels, base_channels * (2 ** i), 3, 2, 1),
                LayerNorm(base_channels * (2 ** i)),
                EfficientDRDB(base_channels * (2 ** i), growth_rate=16)  # 轻量版DRDB
            )
            self.encoder.append(layer)
            current_channels = base_channels * (2 ** i)
            self.scale_factors.append(2 ** (i + 1))

        # 2. 中间处理模块（类似DehazeXL的bottleneck）
        self.bottleneck = nn.Sequential(
            ContextAwareBlock(current_channels),
            ContextAwareBlock(current_channels),
        )

        # 3. 解码器（采用渐进上采样）
        self.decoder = nn.ModuleList()
        for i in range(num_scales - 1, -1, -1):
            in_ch = current_channels + base_channels * (2 ** i) if i < num_scales - 1 else current_channels
            layer = nn.Sequential(
                nn.ConvTranspose2d(in_ch, base_channels * max(1, 2 ** (i - 1)), 4, 2, 1),
                LayerNorm(base_channels * max(1, 2 ** (i - 1))),
                EfficientDRDB(base_channels * max(1, 2 ** (i - 1)), growth_rate=16)
            )
            self.decoder.append(layer)
            current_channels = base_channels * max(1, 2 ** (i - 1))

        # 4. 输出层
        self.output_conv = nn.Sequential(
            nn.Conv2d(current_channels, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1)
        )

    def nested_tokenization(self, x, n_regions_h, n_regions_w):
        """
        类似DehazeXL的分块策略
        """
        return rearrange(
            x,
            "N C (HP HC) (WP WC) -> (N HP WP) C HC WC",
            HP=n_regions_h,
            WP=n_regions_w,
            HC=self.patch_size,
            WC=self.patch_size,
        )

    def merge_tokens(self, x, n_regions_h, n_regions_w, original_size):
        """
        合并分块结果
        """
        H, W = original_size
        return rearrange(
            x,
            "(N HP WP) C HC WC -> N C (HP HC) (WP WC)",
            HP=n_regions_h,
            WP=n_regions_w,
        )[:, :, :H, :W]

    def forward_patchwise(self, x):
        """
        分块处理模式 - 类似DehazeXL
        """
        B, C, H, W = x.shape

        # 计算分块数量
        n_regions_h = max(1, H // self.patch_size)
        n_regions_w = max(1, W // self.patch_size)

        if n_regions_h * n_regions_w > 1:
            # 分块处理
            x_patches = self.nested_tokenization(x, n_regions_h, n_regions_w)
            B_patches = x_patches.shape[0]

            # 分批处理以避免内存爆炸
            outputs = []
            for i in range(0, B_patches, self.batch_size):
                batch = x_patches[i:i + self.batch_size]

                # 编码器阶段（分块处理）
                encoder_features = []
                current = batch
                for encoder_layer in self.encoder:
                    current = encoder_layer(current)
                    encoder_features.append(current)

                # 瓶颈层（分块处理）
                current = self.bottleneck(current)

                # 解码器阶段（分块处理）
                for idx, decoder_layer in enumerate(self.decoder):
                    if idx > 0:
                        # 添加skip connection
                        skip_feature = encoder_features[-(idx + 1)]
                        # 确保尺寸匹配
                        if skip_feature.shape[2:] != current.shape[2:]:
                            skip_feature = F.interpolate(
                                skip_feature,
                                size=current.shape[2:],
                                mode='bilinear',
                                align_corners=False
                            )
                        current = torch.cat([current, skip_feature], dim=1)
                    current = decoder_layer(current)

                # 输出
                output_patch = self.output_conv(current)
                outputs.append(output_patch)

            # 合并所有批次的输出
            all_outputs = torch.cat(outputs, dim=0)

            # 合并回原图
            output = self.merge_tokens(all_outputs, n_regions_h, n_regions_w, (H, W))
        else:
            # 不分块，直接处理
            output = self.forward_direct(x)

        return output

    def forward_direct(self, x):
        """直接处理模式（用于小图像）"""
        # 编码器
        encoder_features = []
        current = x
        for encoder_layer in self.encoder:
            current = encoder_layer(current)
            encoder_features.append(current)

        # 瓶颈层
        current = self.bottleneck(current)

        # 解码器
        for idx, decoder_layer in enumerate(self.decoder):
            if idx > 0:
                # 添加skip connection
                skip_feature = encoder_features[-(idx + 1)]
                if skip_feature.shape[2:] != current.shape[2:]:
                    skip_feature = F.interpolate(
                        skip_feature,
                        size=current.shape[2:],
                        mode='bilinear',
                        align_corners=False
                    )
                current = torch.cat([current, skip_feature], dim=1)
            current = decoder_layer(current)

        return self.output_conv(current)

    def forward(self, x):
        if self.use_patches and x.shape[2] > self.patch_size and x.shape[3] > self.patch_size:
            return self.forward_patchwise(x)
        else:
            return self.forward_direct(x)


class EfficientDRDB(nn.Module):
    """轻量级DRDB版本"""

    def __init__(self, in_ch, growth_rate=16, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList()
        current_channels = in_ch

        for i in range(num_layers):
            layer = nn.Sequential(
                nn.Conv2d(current_channels, growth_rate, 3, 1, 2, dilation=2),
                nn.ReLU(inplace=True)
            )
            self.layers.append(layer)
            current_channels += growth_rate

        self.fusion = nn.Conv2d(current_channels, in_ch, 1)

    def forward(self, x):
        features = [x]
        current = x

        for layer in self.layers:
            new_feature = layer(current)
            features.append(new_feature)
            current = torch.cat(features, dim=1)

        fused = self.fusion(current)
        return x + fused


class ContextAwareBlock(nn.Module):
    """上下文感知块，类似DehazeXL的注意力机制"""

    def __init__(self, channels):
        super().__init__()
        self.norm1 = LayerNorm(channels)
        self.conv1 = nn.Conv2d(channels, channels * 4, 1)
        self.conv2 = nn.Conv2d(channels * 4, channels, 1)
        self.norm2 = LayerNorm(channels)

        # 轻量级注意力
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 8, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 8, channels, 1),
            nn.Sigmoid()
        )

        self.spatial_attn = nn.Sequential(
            nn.Conv2d(channels, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        residual = x

        # 通道注意力
        x_norm = self.norm1(x)
        x_conv = self.conv2(F.gelu(self.conv1(x_norm)))

        # 应用注意力
        ca = self.channel_attn(x_conv)
        sa = self.spatial_attn(x_conv)

        x_attn = x_conv * ca * sa
        x_out = self.norm2(x_attn)

        return residual + x_out


class LayerNorm(nn.Module):
    """轻量级LayerNorm"""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class PatchProcessorV2:
    """
    改进的分块处理器，借鉴DehazeXL的设计
    """

    def __init__(self, crop_size=256, patch_size=128, batch_size=8):
        self.crop_size = crop_size
        self.patch_size = patch_size
        self.batch_size = batch_size

    def process_large_image(self, x, model):
        """
        处理大图像的分块策略
        """
        B, C, H, W = x.shape

        # 计算分块数量
        n_h = (H + self.crop_size - 1) // self.crop_size
        n_w = (W + self.crop_size - 1) // self.crop_size

        if n_h * n_w == 1:
            return model(x)

        # 存储结果
        output = torch.zeros(B, 1, H, W, device=x.device, dtype=x.dtype)
        weight = torch.zeros(B, 1, H, W, device=x.device, dtype=x.dtype)

        # 处理每个块
        for h_idx in range(n_h):
            for w_idx in range(n_w):
                # 计算块的位置
                h_start = h_idx * self.crop_size
                w_start = w_idx * self.crop_size
                h_end = min(h_start + self.crop_size, H)
                w_end = min(w_start + self.crop_size, W)

                # 提取块
                patch = x[:, :, h_start:h_end, w_start:w_end]

                # 处理块
                patch_output = model(patch)

                # 计算权重（边缘部分权重较低）
                patch_h, patch_w = patch_output.shape[2:]
                weight_patch = self.create_weight_map(patch_h, patch_w, device=x.device)

                # 累加到输出
                output[:, :, h_start:h_end, w_start:w_end] += patch_output * weight_patch
                weight[:, :, h_start:h_end, w_start:w_end] += weight_patch

        # 加权平均
        output = output / (weight + 1e-8)
        return output

    def create_weight_map(self, H, W, device='cpu'):
        """创建权重图，中心权重高，边缘权重低"""
        weight = torch.ones(1, 1, H, W, device=device)

        # 创建边缘衰减
        border = min(H, W) // 8
        for i in range(border):
            alpha = i / border
            weight_val = 0.5 + 0.5 * torch.cos(torch.tensor(alpha * torch.pi, device=device))

            weight[:, :, i, :] *= weight_val
            weight[:, :, -i - 1, :] *= weight_val
            weight[:, :, :, i] *= weight_val
            weight[:, :, :, -i - 1] *= weight_val

        return weight


# 使用方法示例：
if __name__ == "__main__":
    # 创建支持分块的模型
    model = DNWithPatches(use_patches=True, patch_size=128, overlap=32)
    model = model.cuda()

    # 假设输入是大图像（例如 512x512）
    batch_size = 2
    input_tensor = torch.randn(batch_size, 3, 512, 512).cuda()

    # 前向传播（会自动分块处理）
    with torch.no_grad():
        output = model(input_tensor)

    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")

    # 也可以创建不分块的版本用于小图像
    model_no_patches = DNWithPatches(use_patches=False)
    model_no_patches = model_no_patches.cuda()

    # 小图像直接处理
    small_input = torch.randn(batch_size, 3, 128, 128).cuda()
    with torch.no_grad():
        small_output = model_no_patches(small_input)

    print(f"Small input shape: {small_input.shape}")
    print(f"Small output shape: {small_output.shape}")
import torch
import torch.nn as nn
import torch.nn.functional as F


class DRDB(nn.Module):
    def __init__(self, in_ch, growth_rate=32):
        super(DRDB, self).__init__()
        in_ch_ = in_ch
        self.Dcov1 = nn.Conv2d(in_ch_, growth_rate, 3, padding=2, dilation=2)
        in_ch_ += growth_rate
        self.Dcov2 = nn.Conv2d(in_ch_, growth_rate, 3, padding=2, dilation=2)
        in_ch_ += growth_rate
        self.Dcov3 = nn.Conv2d(in_ch_, growth_rate, 3, padding=2, dilation=2)
        in_ch_ += growth_rate
        self.Dcov4 = nn.Conv2d(in_ch_, growth_rate, 3, padding=2, dilation=2)
        in_ch_ += growth_rate
        self.Dcov5 = nn.Conv2d(in_ch_, growth_rate, 3, padding=2, dilation=2)
        in_ch_ += growth_rate
        self.conv = nn.Conv2d(in_ch_, in_ch, 1, padding=0)

    def forward(self, x):
        x1 = self.Dcov1(x)
        x1 = F.relu(x1)
        x1 = torch.cat([x, x1], dim=1)

        x2 = self.Dcov2(x1)
        x2 = F.relu(x2)
        x2 = torch.cat([x1, x2], dim=1)

        x3 = self.Dcov3(x2)
        x3 = F.relu(x3)
        x3 = torch.cat([x2, x3], dim=1)

        x4 = self.Dcov4(x3)
        x4 = F.relu(x4)
        x4 = torch.cat([x3, x4], dim=1)

        x5 = self.Dcov5(x4)
        x5 = F.relu(x5)
        x5 = torch.cat([x4, x5], dim=1)

        x6 = self.conv(x5)
        out = x + F.relu(x6)
        return out


class PatchProcessor:
    """图像分块处理器"""

    def __init__(self, patch_size=128, overlap=32):
        """
        Args:
            patch_size: 分块大小
            overlap: 块之间的重叠区域大小
        """
        self.patch_size = patch_size
        self.overlap = overlap

    def split_to_patches(self, x):
        """
        将输入图像分割成重叠的块
        """
        B, C, H, W = x.shape

        # 计算需要分多少块
        stride = self.patch_size - self.overlap * 2
        num_h = (H - self.overlap * 2 + stride - 1) // stride
        num_w = (W - self.overlap * 2 + stride - 1) // stride

        patches = []
        patch_positions = []

        for h_idx in range(num_h):
            for w_idx in range(num_w):
                # 计算块的起始和结束位置
                h_start = h_idx * stride
                w_start = w_idx * stride
                h_end = min(h_start + self.patch_size, H)
                w_end = min(w_start + self.patch_size, W)

                # 调整起始位置以确保最后一个块大小合适
                if h_end == H:
                    h_start = max(0, H - self.patch_size)
                if w_end == W:
                    w_start = max(0, W - self.patch_size)

                # 提取块
                patch = x[:, :, h_start:h_end, w_start:w_end]

                # 如果块大小不够，填充
                if patch.shape[2] < self.patch_size or patch.shape[3] < self.patch_size:
                    pad_h = self.patch_size - patch.shape[2]
                    pad_w = self.patch_size - patch.shape[3]
                    patch = F.pad(patch, (0, pad_w, 0, pad_h), mode='reflect')

                patches.append(patch)
                patch_positions.append((h_start, w_start, h_end, w_end))

        return patches, patch_positions, (H, W)

    def merge_patches(self, patches, patch_positions, original_size):
        """
        将处理后的块合并回原图大小
        """
        H, W = original_size
        B, C, _, _ = patches[0].shape

        # 创建权重矩阵用于混合重叠区域
        device = patches[0].device
        dtype = patches[0].dtype
        weight = torch.ones((self.patch_size, self.patch_size),
                            device=device, dtype=dtype)

        # 创建边缘权重（使用余弦函数平滑过渡）
        # 修复：确保传递给torch.cos的是张量
        for i in range(self.overlap):
            # 创建张量而不是使用标量
            i_tensor = torch.tensor(i, device=device, dtype=dtype)
            alpha = torch.pi * i_tensor / self.overlap
            cos_val = torch.cos(alpha)

            weight_val = 0.5 * (1 - cos_val)

            weight[:, i] *= weight_val
            weight[:, -i - 1] *= weight_val
            weight[i, :] *= weight_val
            weight[-i - 1, :] *= weight_val

        # 初始化输出和权重累积
        output = torch.zeros((B, C, H, W), device=device, dtype=dtype)
        weight_acc = torch.zeros((B, C, H, W), device=device, dtype=dtype)

        for patch, (h_start, w_start, h_end, w_end) in zip(patches, patch_positions):
            # 裁剪掉填充部分
            patch_h = min(self.patch_size, h_end - h_start)
            patch_w = min(self.patch_size, w_end - w_start)

            patch_cropped = patch[:, :, :patch_h, :patch_w]
            weight_cropped = weight[:patch_h, :patch_w].unsqueeze(0).unsqueeze(0)

            # 累加到输出
            output[:, :, h_start:h_end, w_start:w_end] += patch_cropped * weight_cropped
            weight_acc[:, :, h_start:h_end, w_start:w_end] += weight_cropped

        # 防止除以零
        weight_acc = torch.where(weight_acc > 0, weight_acc, torch.ones_like(weight_acc))
        output = output / weight_acc

        return output


class DNWithPatches(nn.Module):
    """
    支持分块处理的去噪网络
    可以设置使用或不使用分块
    """

    def __init__(self, use_patches=True, patch_size=128, overlap=32):
        super(DNWithPatches, self).__init__()
        self.use_patches = use_patches
        self.patch_processor = PatchProcessor(patch_size, overlap) if use_patches else None

        # 原始网络结构
        self.DRDB_layer1 = DRDB(in_ch=3, growth_rate=32)
        self.conv1 = nn.Conv2d(3, 24, 3, 2, 1)
        self.DRDB_layer2 = DRDB(in_ch=24, growth_rate=32)
        self.conv2 = nn.Conv2d(24, 48, 3, 2, 1)
        self.DRDB_layer3 = DRDB(in_ch=48, growth_rate=32)
        self.conv3 = nn.Conv2d(48, 96, 3, 2, 1)
        self.DRDB_layer4 = DRDB(in_ch=96, growth_rate=32)
        self.conv4 = nn.Conv2d(96, 128, 3, 2, 1)

        self.DRDB_layer5 = DRDB(in_ch=128, growth_rate=32)
        self.up1 = nn.ConvTranspose2d(128, 96, kernel_size=4, stride=2, padding=1)
        self.DRDB_layer6 = DRDB(in_ch=96, growth_rate=32)
        self.up2 = nn.ConvTranspose2d(96, 48, kernel_size=4, stride=2, padding=1)
        self.DRDB_layer7 = DRDB(in_ch=48, growth_rate=32)
        self.up3 = nn.ConvTranspose2d(48, 24, kernel_size=4, stride=2, padding=1)
        self.DRDB_layer8 = DRDB(in_ch=24, growth_rate=32)
        self.up4 = nn.ConvTranspose2d(24, 3, kernel_size=4, stride=2, padding=1)
        self.final_conv = nn.Conv2d(3, 1, 1)

    def forward(self, x):
        # 修复：调整分块条件，避免对较小图像分块
        if self.use_patches and self.patch_processor is not None and x.shape[2] > 256 and x.shape[3] > 256:
            # 分块处理模式
            patches, positions, original_size = self.patch_processor.split_to_patches(x)
            processed_patches = []

            for i, patch in enumerate(patches):
                # 处理每个块
                processed_patch = self._forward_single(patch)
                processed_patches.append(processed_patch)

                # 可选：每处理几个块就清理一次缓存
                if (i + 1) % 4 == 0:
                    torch.cuda.empty_cache()

            # 合并块
            output = self.patch_processor.merge_patches(processed_patches, positions, original_size)
            return output
        else:
            # 正常处理模式（小图像）
            return self._forward_single(x)

    def _forward_single(self, x):
        """处理单个图像或块"""
        x1 = self.DRDB_layer1(x)
        x1 = self.conv1(x1)
        x1 = self.DRDB_layer2(x1)
        x1 = self.conv2(x1)
        x1 = self.DRDB_layer3(x1)
        x1 = self.conv3(x1)
        x1 = self.DRDB_layer4(x1)
        x1 = self.conv4(x1)
        x1 = self.DRDB_layer5(x1)
        x1 = self.up1(x1)
        x1 = self.DRDB_layer6(x1)
        x1 = self.up2(x1)
        x1 = self.DRDB_layer7(x1)
        x1 = self.up3(x1)
        x1 = self.DRDB_layer8(x1)
        x1 = self.up4(x1)
        x1 = self.final_conv(x1)
        return x1


import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce


class EfficientDepthNet(nn.Module):
    """
    高效深度估计网络，借鉴DehazeXL的设计思想
    """

    def __init__(
            self,
            in_channels=3,
            base_channels=32,
            num_scales=4,
            crop_size=256,
            mlp_ratio=4,
            use_patches=True,
            patch_size=256,
            batch_size=8
    ):
        super().__init__()
        self.crop_size = crop_size
        self.use_patches = use_patches
        self.patch_size = patch_size
        self.batch_size = batch_size

        # 1. 轻量级编码器（类似DehazeXL的encoder）
        self.encoder = nn.ModuleList()
        current_channels = in_channels
        self.scale_factors = []

        for i in range(num_scales):
            # 每个尺度包含轻量级DRDB
            layer = nn.Sequential(
                nn.Conv2d(current_channels, base_channels * (2 ** i), 3, 2, 1),
                LayerNorm(base_channels * (2 ** i)),
                EfficientDRDB(base_channels * (2 ** i), growth_rate=16)  # 轻量版DRDB
            )
            self.encoder.append(layer)
            current_channels = base_channels * (2 ** i)
            self.scale_factors.append(2 ** (i + 1))

        # 2. 中间处理模块（类似DehazeXL的bottleneck）
        self.bottleneck = nn.Sequential(
            ContextAwareBlock(current_channels),
            ContextAwareBlock(current_channels),
        )

        # 3. 解码器（采用渐进上采样）
        self.decoder = nn.ModuleList()
        for i in range(num_scales - 1, -1, -1):
            in_ch = current_channels + base_channels * (2 ** i) if i < num_scales - 1 else current_channels
            layer = nn.Sequential(
                nn.ConvTranspose2d(in_ch, base_channels * max(1, 2 ** (i - 1)), 4, 2, 1),
                LayerNorm(base_channels * max(1, 2 ** (i - 1))),
                EfficientDRDB(base_channels * max(1, 2 ** (i - 1)), growth_rate=16)
            )
            self.decoder.append(layer)
            current_channels = base_channels * max(1, 2 ** (i - 1))

        # 4. 输出层
        self.output_conv = nn.Sequential(
            nn.Conv2d(current_channels, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1)
        )

    def nested_tokenization(self, x, n_regions_h, n_regions_w):
        """
        类似DehazeXL的分块策略
        """
        return rearrange(
            x,
            "N C (HP HC) (WP WC) -> (N HP WP) C HC WC",
            HP=n_regions_h,
            WP=n_regions_w,
            HC=self.patch_size,
            WC=self.patch_size,
        )

    def merge_tokens(self, x, n_regions_h, n_regions_w, original_size):
        """
        合并分块结果
        """
        H, W = original_size
        return rearrange(
            x,
            "(N HP WP) C HC WC -> N C (HP HC) (WP WC)",
            HP=n_regions_h,
            WP=n_regions_w,
        )[:, :, :H, :W]

    def forward_patchwise(self, x):
        """
        分块处理模式 - 类似DehazeXL
        """
        B, C, H, W = x.shape

        # 计算分块数量
        n_regions_h = max(1, H // self.patch_size)
        n_regions_w = max(1, W // self.patch_size)

        if n_regions_h * n_regions_w > 1:
            # 分块处理
            x_patches = self.nested_tokenization(x, n_regions_h, n_regions_w)
            B_patches = x_patches.shape[0]

            # 分批处理以避免内存爆炸
            outputs = []
            for i in range(0, B_patches, self.batch_size):
                batch = x_patches[i:i + self.batch_size]

                # 编码器阶段（分块处理）
                encoder_features = []
                current = batch
                for encoder_layer in self.encoder:
                    current = encoder_layer(current)
                    encoder_features.append(current)

                # 瓶颈层（分块处理）
                current = self.bottleneck(current)

                # 解码器阶段（分块处理）
                for idx, decoder_layer in enumerate(self.decoder):
                    if idx > 0:
                        # 添加skip connection
                        skip_feature = encoder_features[-(idx + 1)]
                        # 确保尺寸匹配
                        if skip_feature.shape[2:] != current.shape[2:]:
                            skip_feature = F.interpolate(
                                skip_feature,
                                size=current.shape[2:],
                                mode='bilinear',
                                align_corners=False
                            )
                        current = torch.cat([current, skip_feature], dim=1)
                    current = decoder_layer(current)

                # 输出
                output_patch = self.output_conv(current)
                outputs.append(output_patch)

            # 合并所有批次的输出
            all_outputs = torch.cat(outputs, dim=0)

            # 合并回原图
            output = self.merge_tokens(all_outputs, n_regions_h, n_regions_w, (H, W))
        else:
            # 不分块，直接处理
            output = self.forward_direct(x)

        return output

    def forward_direct(self, x):
        """直接处理模式（用于小图像）"""
        # 编码器
        encoder_features = []
        current = x
        for encoder_layer in self.encoder:
            current = encoder_layer(current)
            encoder_features.append(current)

        # 瓶颈层
        current = self.bottleneck(current)

        # 解码器
        for idx, decoder_layer in enumerate(self.decoder):
            if idx > 0:
                # 添加skip connection
                skip_feature = encoder_features[-(idx + 1)]
                if skip_feature.shape[2:] != current.shape[2:]:
                    skip_feature = F.interpolate(
                        skip_feature,
                        size=current.shape[2:],
                        mode='bilinear',
                        align_corners=False
                    )
                current = torch.cat([current, skip_feature], dim=1)
            current = decoder_layer(current)

        return self.output_conv(current)

    def forward(self, x):
        if self.use_patches and x.shape[2] > self.patch_size and x.shape[3] > self.patch_size:
            return self.forward_patchwise(x)
        else:
            return self.forward_direct(x)


class EfficientDRDB(nn.Module):
    """轻量级DRDB版本"""

    def __init__(self, in_ch, growth_rate=16, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList()
        current_channels = in_ch

        for i in range(num_layers):
            layer = nn.Sequential(
                nn.Conv2d(current_channels, growth_rate, 3, 1, 2, dilation=2),
                nn.ReLU(inplace=True)
            )
            self.layers.append(layer)
            current_channels += growth_rate

        self.fusion = nn.Conv2d(current_channels, in_ch, 1)

    def forward(self, x):
        features = [x]
        current = x

        for layer in self.layers:
            new_feature = layer(current)
            features.append(new_feature)
            current = torch.cat(features, dim=1)

        fused = self.fusion(current)
        return x + fused


class ContextAwareBlock(nn.Module):
    """上下文感知块，类似DehazeXL的注意力机制"""

    def __init__(self, channels):
        super().__init__()
        self.norm1 = LayerNorm(channels)
        self.conv1 = nn.Conv2d(channels, channels * 4, 1)
        self.conv2 = nn.Conv2d(channels * 4, channels, 1)
        self.norm2 = LayerNorm(channels)

        # 轻量级注意力
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 8, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 8, channels, 1),
            nn.Sigmoid()
        )

        self.spatial_attn = nn.Sequential(
            nn.Conv2d(channels, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        residual = x

        # 通道注意力
        x_norm = self.norm1(x)
        x_conv = self.conv2(F.gelu(self.conv1(x_norm)))

        # 应用注意力
        ca = self.channel_attn(x_conv)
        sa = self.spatial_attn(x_conv)

        x_attn = x_conv * ca * sa
        x_out = self.norm2(x_attn)

        return residual + x_out


class LayerNorm(nn.Module):
    """轻量级LayerNorm"""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class PatchProcessorV2:
    """
    改进的分块处理器，借鉴DehazeXL的设计
    """

    def __init__(self, crop_size=256, patch_size=128, batch_size=8):
        self.crop_size = crop_size
        self.patch_size = patch_size
        self.batch_size = batch_size

    def process_large_image(self, x, model):
        """
        处理大图像的分块策略
        """
        B, C, H, W = x.shape

        # 计算分块数量
        n_h = (H + self.crop_size - 1) // self.crop_size
        n_w = (W + self.crop_size - 1) // self.crop_size

        if n_h * n_w == 1:
            return model(x)

        # 存储结果
        output = torch.zeros(B, 1, H, W, device=x.device, dtype=x.dtype)
        weight = torch.zeros(B, 1, H, W, device=x.device, dtype=x.dtype)

        # 处理每个块
        for h_idx in range(n_h):
            for w_idx in range(n_w):
                # 计算块的位置
                h_start = h_idx * self.crop_size
                w_start = w_idx * self.crop_size
                h_end = min(h_start + self.crop_size, H)
                w_end = min(w_start + self.crop_size, W)

                # 提取块
                patch = x[:, :, h_start:h_end, w_start:w_end]

                # 处理块
                patch_output = model(patch)

                # 计算权重（边缘部分权重较低）
                patch_h, patch_w = patch_output.shape[2:]
                weight_patch = self.create_weight_map(patch_h, patch_w, device=x.device)

                # 累加到输出
                output[:, :, h_start:h_end, w_start:w_end] += patch_output * weight_patch
                weight[:, :, h_start:h_end, w_start:w_end] += weight_patch

        # 加权平均
        output = output / (weight + 1e-8)
        return output

    def create_weight_map(self, H, W, device='cpu'):
        """创建权重图，中心权重高，边缘权重低"""
        weight = torch.ones(1, 1, H, W, device=device)

        # 创建边缘衰减
        border = min(H, W) // 8
        for i in range(border):
            alpha = i / border
            weight_val = 0.5 + 0.5 * torch.cos(torch.tensor(alpha * torch.pi, device=device))

            weight[:, :, i, :] *= weight_val
            weight[:, :, -i - 1, :] *= weight_val
            weight[:, :, :, i] *= weight_val
            weight[:, :, :, -i - 1] *= weight_val

        return weight


# 使用方法示例：
if __name__ == "__main__":
    # 创建支持分块的模型
    model = DNWithPatches(use_patches=True, patch_size=128, overlap=32)
    model = model.cuda()

    # 假设输入是大图像（例如 512x512）
    batch_size = 2
    input_tensor = torch.randn(batch_size, 3, 512, 512).cuda()

    # 前向传播（会自动分块处理）
    with torch.no_grad():
        output = model(input_tensor)

    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")

    # 也可以创建不分块的版本用于小图像
    model_no_patches = DNWithPatches(use_patches=False)
    model_no_patches = model_no_patches.cuda()

    # 小图像直接处理
    small_input = torch.randn(batch_size, 3, 128, 128).cuda()
    with torch.no_grad():
        small_output = model_no_patches(small_input)

    print(f"Small input shape: {small_input.shape}")
    print(f"Small output shape: {small_output.shape}")