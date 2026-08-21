import torch
import torch.hub
from einops import rearrange
from torch import nn
from ..context_encoders import ContextEncoderConfig
from .utils import get_2d_sincos_pos_embed, LlamaRMSNorm
from ..backbones import *
import numpy as np
from .FreqFusion import FreqFusion

# default_decoder_filters = [48, 96, 176, 256]
# default_last = 48


class AbstractModel(nn.Module):
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                m.weight.data = nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class LlamaMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.pretraining_tp = 1
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.GELU()

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


class LLMLayer(nn.Module):
    def __init__(
            self, dim, inner_dim, num_heads, causal=False, attention_method="hyper"
    ):
        super().__init__()
        # num_heads = dim // 128
        if attention_method == "hyper":
            from ..context_encoders.attention import LLMAttention
            self.attn = LLMAttention(dim, dim, num_heads, causal=causal)
            # from ..context_encoders.attentionmla import LLM_mlh_Attention
            # self.attn = LLM_mlh_Attention(dim, dim, num_heads, causal=causal)
        else:
            from ..context_encoders.attention import ViTAttention
            self.attn = ViTAttention(dim, dim, num_heads, causal=causal)
        self.input_layernorm = LlamaRMSNorm(dim, eps=1e-05)
        self.post_attention_layernorm = LlamaRMSNorm(dim, eps=1e-05)
        self.mlp = LlamaMLP(dim, inner_dim)
        self.causal = causal

    # def forward(self, hidden_states, residual_in=-1):
    def forward(self, hidden_states, residual_in=-1, kv_cache=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.attn(hidden_states)
        # hidden_states, kv_cache = self.attn(hidden_states, kv_cache)

        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        if residual_in != -1:
            return hidden_states, 0.0, kv_cache
        return hidden_states


class LLMBottleneck(nn.Module):
    def __init__(
            self,
            in_dim,
            mlp_ratio=4,
            hidden_size=768,
            num_heads=8,
            n_layers=2,
            attention_method=None,
    ):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_size)
        assert attention_method in ["hyper", "naive"]
        self.layers = nn.Sequential(
            *[
                LLMLayer(
                    hidden_size,
                    hidden_size * mlp_ratio,
                    num_heads,
                    causal=False,
                    attention_method=attention_method,
                )
                for _ in range(n_layers)
            ]
        )
        self.hidden_size = hidden_size

    def _create_pos_embed_rectangular(self, embed_dim, grid_h, grid_w):

        # Éú³É¸ß¶ÈºÍ¿í¶ÈµÄÍø¸ñ
        grid_h_range = np.arange(grid_h, dtype=np.float32)
        grid_w_range = np.arange(grid_w, dtype=np.float32)
        grid = np.meshgrid(grid_w_range, grid_h_range)  # ×¢ÒâË³Ðò£º¿í¶ÈÔÚÇ°
        grid = np.stack(grid, axis=0)  # (2, H, W)

        # Õ¹Æ½Íø¸ñ
        grid_flat = grid.reshape(2, -1)  # (2, H*W)

        # ·Ö±ð¼ÆËã¸ß¶ÈºÍ¿í¶ÈµÄÎ»ÖÃ±àÂë
        emb_h = self._get_1d_sincos_pos_embed(embed_dim // 2, grid_flat[0])
        emb_w = self._get_1d_sincos_pos_embed(embed_dim // 2, grid_flat[1])

        # Æ´½Ó
        pos_embed = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
        return pos_embed

    def _get_1d_sincos_pos_embed(self, embed_dim, pos):

        assert embed_dim % 2 == 0
        omega = np.arange(embed_dim // 2, dtype=np.float32)
        omega /= embed_dim / 2.0
        omega = 1.0 / 10000 ** omega

        pos = pos.reshape(-1)
        out = np.einsum("m,d->md", pos, omega)

        emb_sin = np.sin(out)
        emb_cos = np.cos(out)

        emb = np.concatenate([emb_sin, emb_cos], axis=1)
        return emb

    def forward(self, x):
        x = x[-1]
        n, _, h, w = x.shape

        # Ê¹ÓÃÐÂµÄ¾ØÐÎÎ»ÖÃ±àÂëº¯Êý
        pos_embed = self._create_pos_embed_rectangular(self.hidden_size, h, w)

        x = rearrange(x, "n c h w -> n (h w) c")
        x = self.input_proj(x)

        # Ìí¼ÓÎ»ÖÃ±àÂë
        pos_embed = torch.tensor(pos_embed, dtype=x.dtype, device=x.device).unsqueeze(0)
        x = x + pos_embed

        residual = None
        kv_cache = None

        for i, blk in enumerate(self.layers):
            x, residual, kv_cache = blk(x, residual, kv_cache)
            if i == len(self.layers) - 1:
                x = (x + residual) if residual is not None else x
        x = rearrange(x, "n (h w) c -> n c h w", h=h, w=w)
        return x


class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        if isinstance(input_resolution, int):
            self.input_resolution = (input_resolution, input_resolution)
        elif isinstance(input_resolution, tuple):
            self.input_resolution = input_resolution
        else:
            raise TypeError("input_resolution should be tuple or int.")
        self.dim = dim
        self.expand = nn.Linear(dim, int(dim_scale * dim // 4), bias=False)
        self.norm = norm_layer(int(dim // dim_scale))

    def forward(self, x):
        """
        x: B, H*W, C
        """
        h, w = self.input_resolution
        x = self.expand(x)
        b, l, c = x.shape
        assert l == h * w, "input feature has wrong size"

        x = x.view(b, h, w, c)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=c // 4)
        x = x.view(b, -1, c // 4)
        x = self.norm(x)

        return x


class PatchExpandConv(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4):
        super().__init__()
        if isinstance(input_resolution, int):
            self.input_resolution = (input_resolution, input_resolution)
        elif isinstance(input_resolution, tuple):
            self.input_resolution = input_resolution
        else:
            raise TypeError("input_resolution should be tuple or int.")
        self.dim = dim
        self.expand = nn.ConvTranspose2d(dim, int(dim // dim_scale), kernel_size=2, stride=2)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        h, w = self.input_resolution
        x = rearrange(x, "n (h w) c -> n c h w", h=h, w=w)
        x = self.expand(x)
        x = rearrange(x, "n c h w -> n (h w) c")
        return x


# 在文件开头添加一个标志来控制是否使用FreqFusion
USE_FREQFUSION = True  # 先设置为False，让模型能运行


class LayerUp(nn.Module):
    def __init__(self, dim, enc_dim, input_resolution, depth, num_heads, window_size, dim_scale=4,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=True):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.dim_scale = dim_scale

        # 维度适配层
        self.dim_adapter = nn.LazyLinear(dim)

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # 根据标志决定是否使用FreqFusion
        if upsample is not None and USE_FREQFUSION:
            upsampled_channels = dim // dim_scale
            self.freq_fusion = FreqFusion(
                hr_channels=upsampled_channels,
                lr_channels=enc_dim
            )
            print(f"LayerUp: 启用FreqFusion, hr_channels={upsampled_channels}, lr_channels={enc_dim}")
        else:
            self.freq_fusion = None
            if upsample is not None:
                print(f"LayerUp: 禁用FreqFusion")

        if upsample is not None:
            self.upsample = PatchExpandConv(input_resolution, dim=dim, dim_scale=dim_scale)
        else:
            self.upsample = None

    def forward(self, x, enc_feat=None):
        """处理解码器特征和编码器特征"""
        # 如果维度不匹配，进行适配
        if x.shape[-1] != self.dim:
            x = self.dim_adapter(x)

        # Swin Transformer块
        for blk in self.blocks:
            x = blk(x)

        # 上采样和特征融合
        if self.upsample is not None:
            x_upsampled = self.upsample(x)

            if enc_feat is not None and self.freq_fusion is not None and USE_FREQFUSION:
                # 转换到2D格式
                B, N, C = x_upsampled.shape
                H = W = int(N ** 0.5)
                x_upsampled_2d = rearrange(x_upsampled, "b (h w) c -> b c h w", h=H, w=W)

                # 编码器特征转换
                B_enc, N_enc, C_enc = enc_feat.shape
                H_enc = W_enc = int(N_enc ** 0.5)
                enc_feat_2d = rearrange(enc_feat, "b (h w) c -> b c h w", h=H_enc, w=W_enc)

                # 应用FreqFusion
                try:
                    _, fused_hr, _ = self.freq_fusion(
                        hr_feat=x_upsampled_2d,
                        lr_feat=enc_feat_2d
                    )
                    # 转换回序列格式
                    print("应用FreqFusion")
                    x = rearrange(fused_hr, "b c h w -> b (h w) c")
                except Exception as e:
                    print(f"FreqFusion失败，使用上采样结果: {e}")
                    x = x_upsampled
            else:
                # 没有编码器特征或FreqFusion未启用时，直接上采样
                x = x_upsampled

        return x


class SwinDecoder(nn.Module):
    def __init__(self, in_resolution_h, in_resolution_w, enc_channels, bottleneck_channels,
                 depths=[2, 2, 2, 2], num_heads=[3, 6, 12, 24]):
        """
        参数:
        - enc_channels: 编码器特征通道数列表 [C1, C2, C3, C4]
        - bottleneck_channels: 瓶颈层输出通道数
        """
        super().__init__()
        self.num_layers = len(depths)
        self.crop_size = 256
        drop_path_rate = 0.1
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers_up = nn.ModuleList()

        # 确保enc_channels长度正确
        if len(enc_channels) != self.num_layers:
            raise ValueError(f"编码器通道数列表长度({len(enc_channels)})应与解码器层数({self.num_layers})相同")

        # 计算每一层的维度
        decoder_dims = []
        for i_layer in range(self.num_layers):
            # 解码器当前层输入通道数
            decoder_dim = bottleneck_channels // (2 ** i_layer)
            decoder_dims.append(decoder_dim)

        for i_layer in range(self.num_layers):
            current_res_h = in_resolution_h // (2 ** (self.num_layers - 1 - i_layer))
            current_res_w = in_resolution_w // (2 ** (self.num_layers - 1 - i_layer))

            # 解码器当前层维度
            decoder_dim = decoder_dims[i_layer]
            # 对应编码器特征通道数（逆序）
            enc_dim = enc_channels[self.num_layers - 1 - i_layer]

            print(f"解码器层 {i_layer}: 输入分辨率 {current_res_h}x{current_res_w}, "
                  f"解码器维度 {decoder_dim}, 编码器维度 {enc_dim}")

            up = LayerUp(
                dim=decoder_dim,  # 解码器维度
                enc_dim=enc_dim,  # 编码器维度
                input_resolution=(current_res_h, current_res_w),
                depth=depths[(self.num_layers - 1 - i_layer)],
                num_heads=num_heads[(self.num_layers - 1 - i_layer)],
                dim_scale=4,
                drop_path=dpr[sum(depths[:(self.num_layers - 1 - i_layer)]):sum(
                    depths[:(self.num_layers - 1 - i_layer) + 1])],
                window_size=8,
                upsample=True
            )
            self.layers_up.append(up)

        self.batch_size = 16

        # 修正：计算最后一层LayerUp的输出通道数
        # 最后一层decoder_dim = decoder_dims[-1] = bottleneck_channels // (2 ** (num_layers-1))
        # 上采样后通道数 = decoder_dim // 4
        last_layer_output_channels = decoder_dims[-1] // 4

        print(f"最后一层输入通道数: {decoder_dims[-1]}")
        print(f"最后一层输出通道数: {last_layer_output_channels}")

        # 最终上采样层：使用最后一层的输出通道数作为输入
        self.up_x2 = PatchExpandConv(
            2 * max(in_resolution_h, in_resolution_w),
            last_layer_output_channels,  # 修正：使用正确的输入通道数
            dim_scale=4
        )

        # 输出层：up_x2 上采样后通道数会变为 last_layer_output_channels // 4
        self.output = nn.Conv2d(last_layer_output_channels // 4, 3, 1, bias=False)

    def forward(self, x, fm_list, n_regions_h, n_regions_w):
        # 添加调试信息
        print(f"解码器输入 x 形状: {x.shape}")
        for i, feat in enumerate(fm_list):
            print(f"编码器特征 fm_list[{i}] 形状: {feat.shape}")

        # 检查维度
        expected_channels = self.layers_up[0].dim
        if x.shape[1] != expected_channels:
            print(f"警告：解码器输入通道数 {x.shape[1]} 与期望 {expected_channels} 不匹配")

        x = rearrange(x, "N C (HP HC) (WP WC)-> (N HP WP) (HC WC) C",
                      HP=n_regions_h, WP=n_regions_w)
        fm_list = [rearrange(i, "N C (HP HC) (WP WC)-> (N HP WP) (HC WC) C",
                             HP=n_regions_h, WP=n_regions_w) for i in fm_list]

        print(f"重排后解码器输入 x 形状: {x.shape}")
        for i, feat in enumerate(fm_list):
            print(f"重排后编码器特征 [{i}] 形状: {feat.shape}")

        n = x.shape[0]
        outputs = []
        for i in range(0, n, self.batch_size):
            end = min(i + self.batch_size, n)
            x_batch = x[i:end]
            fm_batch = [fm[i:end] for fm in fm_list]

            for j in range(self.num_layers):
                enc_feat = fm_batch[self.num_layers - 1 - j]
                print(f"处理层 {j}: 解码器特征形状 {x_batch.shape}, 编码器特征形状 {enc_feat.shape}")
                x_batch = self.layers_up[j](x_batch, enc_feat=enc_feat)
                print(f"层 {j} 输出形状: {x_batch.shape}")

            outputs.append(x_batch)

        x = torch.cat(outputs, dim=0)
        print(f"解码器最终输出形状（上采样前）: {x.shape}")

        # 检查 up_x2 的输入通道数
        B, N, C = x.shape
        expected_channels = self.up_x2.dim
        print(f"up_x2 期望输入通道数: {expected_channels}, 实际输入通道数: {C}")

        if C != expected_channels:
            print(f"错误：up_x2 输入通道数不匹配！期望 {expected_channels}，实际 {C}")
            # 尝试修复：添加一个线性层来适配通道数
            if not hasattr(self, 'channel_adapter'):
                self.channel_adapter = nn.Linear(C, expected_channels).to(x.device)
            x = self.channel_adapter(x)
            print(f"通道适配后形状: {x.shape}")

        x = self.up_x2(x)
        print(f"上采样后形状: {x.shape}")

        x = rearrange(x, "n (HC WC) C -> n C HC WC",
                      HC=self.crop_size, WC=self.crop_size)
        print(f"重排为2D后形状: {x.shape}")

        x = self.output(x)
        print(f"输出层后形状: {x.shape}")

        x = rearrange(x, "(N HP WP) C HC WC -> N C (HP HC) (WP WC)",
                      HP=n_regions_h, WP=n_regions_w)
        print(f"最终输出形状: {x.shape}")

        return x


class DehazeXL2(AbstractModel):
    def __init__(
            self,
            backbone: nn.Module = swinv2_tiny_window16_256_timm(input_size=256),
            xl_config: ContextEncoderConfig = ContextEncoderConfig,
            channels_last: bool = True,
            crop_size: int = 256,
            mlp_ratio: int = 4,
    ):
        self.channels_last = channels_last
        self.crop_size = crop_size
        self.filters = [f["num_chs"] for f in backbone.feature_info]
        self.mlp_ratio = mlp_ratio
        self.xl_config = xl_config

        super().__init__()

        self.batch_size = 16
        self._initialize_weights()

        self.encoder = backbone
        self.bottleneck = LLMBottleneck(
            in_dim=self.filters[-1],
            mlp_ratio=self.mlp_ratio,
            hidden_size=self.xl_config.hidden_size,
            n_layers=self.xl_config.n_layer,
            attention_method=self.xl_config.attention_method,
        )

        # 获取编码器通道数
        # 假设编码器输出4个特征图，通道数分别为: [96, 192, 384, 768]
        enc_channels = self.filters  # 例如: [96, 192, 384, 768]

        # 计算瓶颈层输出维度（应该是隐藏大小）
        bottleneck_out_channels = self.xl_config.hidden_size

        self.decoder = SwinDecoder(
            in_resolution_h=crop_size // 4,
            in_resolution_w=crop_size // 4,
            enc_channels=enc_channels,  # 传递编码器通道数
            bottleneck_channels=bottleneck_out_channels,  # 瓶颈层输出通道数
        )

    def forward(self, x):
        # Encoder
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        x_skip = x

        n_regions_h = x.shape[2] // self.crop_size
        n_regions_w = x.shape[3] // self.crop_size

        if n_regions_h > 0 and n_regions_w > 0:
            x = self.nested_tokenization(x, n_regions_h, n_regions_w)

            n = x.shape[0]
            outputs = []
            for i in range(0, n, self.batch_size):
                batch = x[i:min(i + self.batch_size, n)]
                output = self.encoder(batch)
                outputs.append(output)
            enc_results = [torch.cat([outputs[j][i] for j in range(len(outputs))], dim=0) for i in range(4)]

            enc_results = list(
                [
                    rearrange(
                        i,
                        "(N HP WP) C HC WC -> N C (HP HC) (WP WC)",
                        HP=n_regions_h,
                        WP=n_regions_w,
                    )
                    for i in enc_results
                ]
            )
        else:
            enc_results = self.encoder(x)
            n_regions_h = 1
            n_regions_w = 1

        output = self.bottleneck(enc_results)
        output = self.decoder(output, enc_results, n_regions_h, n_regions_w)
        output += x_skip
        return output

    def nested_tokenization(self, x, n_regions_h, n_regions_w):
        x = rearrange(
            x,
            "N C (HP HC) (WP WC)-> (N HP WP) C HC WC ",
            HP=n_regions_h,
            WP=n_regions_w,
            HC=self.crop_size,
            WC=self.crop_size,
        )
        return x

