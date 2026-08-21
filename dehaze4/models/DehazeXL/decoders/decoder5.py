import torch
import torch.hub
from einops import rearrange
from torch import nn
from ..context_encoders import ContextEncoderConfig
from .utils import LlamaRMSNorm
from ..backbones import *
import numpy as np
from .decoder1 import AbstractModel, LLMBottleneck, PatchExpandConv, LayerUp

class SwinDecoder(nn.Module):
    def __init__(self, in_resolution_h, in_resolution_w, in_dim=96, depths=[2, 2, 2, 2], num_heads=[3, 6, 12, 24]):
        super().__init__()
        self.num_layers = len(depths)
        self.crop_size = 256
        drop_path_rate = 0.1
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers_up = nn.ModuleList()
        
        for i_layer in range(self.num_layers):
            current_res_h = in_resolution_h // (2 ** (self.num_layers - 1 - i_layer))
            current_res_w = in_resolution_w // (2 ** (self.num_layers - 1 - i_layer))
            
            up = LayerUp(dim=int(in_dim * 2 ** (self.num_layers - i_layer)), dim_scale=4,
                         input_resolution=(current_res_h, current_res_w),
                         num_heads=num_heads[(self.num_layers - 1 - i_layer)],
                         depth=depths[(self.num_layers - 1 - i_layer)],
                         drop_path=dpr[sum(depths[:(self.num_layers - 1 - i_layer)]):sum(
                             depths[:(self.num_layers - 1 - i_layer) + 1])],
                         window_size=8,
                         upsample=True)
            self.layers_up.append(up)
        self.batch_size = 16
        self.up_x2 = PatchExpandConv(2 * max(in_resolution_h, in_resolution_w), in_dim // 2, dim_scale=4)
        # Explicitly set output channels to 3 to match input image (x_skip)
        self.output = nn.Conv2d(in_dim // 8, 3, 1, bias=False)

    def forward(self, x, fm_list, n_regions_h, n_regions_w):
        x = rearrange(x, "N C (HP HC) (WP WC)-> (N HP WP) (HC WC) C", 
                     HP=n_regions_h, WP=n_regions_w)
        fm_list = [rearrange(i, "N C (HP HC) (WP WC)-> (N HP WP) (HC WC) C",
                             HP=n_regions_h, WP=n_regions_w) for i in fm_list]

        n = x.shape[0]
        outputs = []
        for i in range(0, n, self.batch_size):
            end = min(i + self.batch_size, n)
            x_batch = x[i:end]
            fm_batch = [fm[i:end] for fm in fm_list]
            for j in range(self.num_layers):
                x_batch = self.layers_up[j](torch.cat([x_batch, fm_batch[self.num_layers - 1 - j]], dim=-1))
            outputs.append(x_batch)
        x = torch.cat(outputs, dim=0)

        x = self.up_x2(x)
        # Use actual patch size
        x = rearrange(x, "n (HC WC) C -> n C HC WC", 
                     HC=self.crop_size, WC=self.crop_size)
        x = self.output(x)
        x = rearrange(x, "(N HP WP) C HC WC -> N C (HP HC) (WP WC)", 
                     HP=n_regions_h, WP=n_regions_w)
        return x

class AuxAdapter(nn.Module):
    """
    Adapter to downsample DWT (3ch) + Depth (1ch) features to match 
    the dimension of the main backbone output (usually 1/32 scale).
    Structure: 5 layers of stride-2 convolutions.
    """
    def __init__(self, in_ch=4, out_ch=768):
        super().__init__()
        dims = [in_ch, 32, 64, 128, 256, out_ch]
        layers = []
        for i in range(5): # 2^5 = 32 downsampling
            layers.append(nn.Conv2d(dims[i], dims[i+1], kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(dims[i+1]))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class DehazeXL_TriInput(AbstractModel):
    def __init__(
            self,
            backbone: nn.Module = swinv2_tiny_window16_256_timm(input_size=256),
            xl_config: ContextEncoderConfig = ContextEncoderConfig,
            channels_last: bool = True,
            crop_size: int = 256,
            mlp_ratio: int = 4,
    ):
        super().__init__()
        self.channels_last = channels_last
        self.crop_size = crop_size
        self.filters = [f["num_chs"] for f in backbone.feature_info]
        self.mlp_ratio = mlp_ratio
        self.xl_config = xl_config
        self.batch_size = 16

        self._initialize_weights()

        self.encoder = backbone
        
        # Main Bottleneck
        self.bottleneck = LLMBottleneck(
            in_dim=self.filters[-1],
            mlp_ratio=self.mlp_ratio,
            hidden_size=self.xl_config.hidden_size,
            n_layers=self.xl_config.n_layer,
            attention_method=self.xl_config.attention_method,
        )
        
        # New: Adapter for DWT+Depth
        # Input 4 channels (3 DWT + 1 Depth), Output matches bottleneck input dim
        # But wait, bottleneck input is filters[-1], output is hidden_size.
        # AuxAdapter should output hidden_size to match bottleneck output for summation, 
        # or we concat and reduce.
        # Let's target hidden_size.
        target_dim = self.xl_config.hidden_size 
        
        # Re-init Adapter with correct dim
        self.aux_adapter = AuxAdapter(in_ch=4, out_ch=target_dim)
        
        # Fusion Layer: Simple projection to merge them
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(target_dim * 2, target_dim, 1),
            nn.Sigmoid() # Gate mechanism? Or just conv? Let's use simple conv + residual
        )
        self.fusion_proj = nn.Conv2d(target_dim * 2, target_dim, 1)

        self.decoder = SwinDecoder(
            in_resolution_h=crop_size // 4,
            in_resolution_w=crop_size // 4,
            in_dim=96 
        )

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def nested_tokenization(self, x, n_regions_h, n_regions_w):
        # x: (N, C, H, W) -> (N*HP*WP, C, HC, WC)
        x = rearrange(
            x,
            "N C (HP HC) (WP WC)-> (N HP WP) C HC WC ",
            HP=n_regions_h,
            WP=n_regions_w,
            HC=self.crop_size,
            WC=self.crop_size,
        )
        return x

    def forward(self, x, dwt_feat, depth_feat):
        # x: [B, 3, H, W]
        # dwt_feat: [B, 3, H, W]
        # depth_feat: [B, 1, H, W]
        
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        x_skip = x
        
        # Padding if needed (DehazeXL usually handles exact patches, but let's check size)
        # Assuming caller (fusion_net) handles padding or ensuring divisibility.
        # If input size is not divisible by crop_size, nested_tokenization will fail or cut.
        # Original DehazeXL assumes input is multiples of crop_size? 
        # Actually in train4.py we pad/crop. In original decoder1.py, it calculates n_regions.
        # If not divisible, re-arrange will fail. 
        # We assume input is already padded/aligned by caller.
        
        n_regions_h = x.shape[2] // self.crop_size
        n_regions_w = x.shape[3] // self.crop_size
        
        aux_input = torch.cat([dwt_feat, depth_feat], dim=1) # [B, 4, H, W]

        # --- 1. Encoder Path ---
        if n_regions_h > 0 and n_regions_w > 0:
            # Patching
            x_patched = self.nested_tokenization(x, n_regions_h, n_regions_w)
            aux_patched = self.nested_tokenization(aux_input, n_regions_h, n_regions_w)
            
            n = x_patched.shape[0]
            enc_outputs = []
            aux_outputs = []
            
            for i in range(0, n, self.batch_size):
                # Main Encoder Batch
                batch_x = x_patched[i:min(i + self.batch_size, n)]
                out_x = self.encoder(batch_x)
                enc_outputs.append(out_x)
                
                # Aux Adapter Batch
                batch_aux = aux_patched[i:min(i + self.batch_size, n)]
                out_aux = self.aux_adapter(batch_aux)
                aux_outputs.append(out_aux)

            # Re-assemble Main Encoder Features (Multiscale list)
            # enc_results: list of [B', C, H, W] for each scale
            enc_results = [torch.cat([enc_outputs[j][k] for j in range(len(enc_outputs))], dim=0) for k in range(4)]
            
            # Re-assemble Aux Features (Single scale)
            aux_results = torch.cat(aux_outputs, dim=0) # [N_patches, C_hidden, 8, 8] (assuming 256/32=8)

            # Un-patch (rearrange back to full image structure) for Bottleneck
            # Decoder1.py logic:
            # Re-arrange enc_results to "N C (HP HC) (WP WC)"
            enc_results = [
                rearrange(
                    res,
                    "(N HP WP) C HC WC -> N C (HP HC) (WP WC)",
                    HP=n_regions_h, WP=n_regions_w
                ) for res in enc_results
            ]
            
            # Aux results also need un-patching?
            # Wait, AuxAdapter does 32x downsample. 
            # Input patch: 256x256 -> Output patch: 8x8.
            # We want to reconstruct the full 32x downsampled feature map.
            # HC_feat = 8, WC_feat = 8.
            # But the Bottleneck inside LLM might use 2D embeddings or something?
            # decoder1.py bottleneck(enc_results):
            #   x = x[-1] -> (N, C, H, W)
            #   rearrange(x, "n c h w -> n (h w) c")
            # So bottleneck takes standard BCHW (or B(sequence)C).
            # Yes, we should un-patch aux_results too if we want global processing.
            
            # But `bottleneck` in decoder1.py processes the list `enc_results`.
            # `LLMBottleneck` uses x[-1].
            
            # Since we want to fuse BEFORE decoder but AFTER Bottleneck (or parallel):
            # Let's Un-patch Aux Features too.
            aux_full = rearrange(
                aux_results,
                "(N HP WP) C HC WC -> N C (HP HC) (WP WC)",
                HP=n_regions_h, WP=n_regions_w
            )
            
        else:
            # No patching (image small or test mode without crop)
            enc_results = self.encoder(x)
            aux_full = self.aux_adapter(aux_input)
            n_regions_h = 1
            n_regions_w = 1

        # --- 2. Bottleneck Path (Main) ---
        # enc_results contains multi-scale features. Bottleneck uses the last one.
        main_feat = self.bottleneck(enc_results) # Returns (N, C, H_feat, W_feat)
        
        # --- 3. Fusion ---
        # main_feat and aux_full should have same spatial dim (H/32, W/32) and same C?
        # Check alignment
        if main_feat.shape != aux_full.shape:
            # Interpolate aux if needed (though strides match theoretically)
            aux_full = nn.functional.interpolate(aux_full, size=main_feat.shape[2:], mode='bilinear', align_corners=False)
        
        # Concat & Project
        fused_feat = torch.cat([main_feat, aux_full], dim=1)
        fused_feat = self.fusion_proj(fused_feat)
        
        # --- 4. Decoder Path ---
        output = self.decoder(fused_feat, enc_results, n_regions_h, n_regions_w)
        output += x_skip # Global residual
        
        return output

