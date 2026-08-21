# 精简项目逐文件清单

## 1. 整理结论

- 源项目：`D:/下载/DWT/new/dehaze1`，盘点 900 个文件，约 563 MB。
- 精简项目：`D:/下载/DWT/new/dehaze4`，保留 88 个非缓存文件；体积主要由 Depth Anything V2 ViT-B 权重（约 372 MB）和 RA-Depth 权重（约 39 MB）构成。
- 主入口收敛为 `train_fusion_net_depth_best.py` 与 `test_fusion_net_depth_best.py`，命令行只允许构建 `fusion_net_depth_best`（版本 13）。重复的历史训练/测试入口未交付。
- RA-Depth 与 Depth Anything 已统一放在 `depth_teachers/` 下。RA-Depth 删除姿态估计、数据集、训练、评估、演示及 pose 权重；Depth Anything V2 删除完整仓库中的 demo、video、metric-depth、assets 与说明，只保留推理核心。

## 2. 入口与依赖关系

```text
train_fusion_net_depth_best.py ─┬─ dataset.py ── haze_synthesizer.py
                         ├─ model1/ + models/ + models2/
                         ├─ loss/ + perceptual.py
                         └─ depth_teachers/ra_depth 或 depth_anything_v1

test_fusion_net_depth_best.py ──┬─ model1/ + models/ + models2/
                         └─ utils/metrics.py + pytorch_ssim/

precompute_depth.py ───── depth_teachers/ra_depth/
precompute_depth_v2.py ── depth_teachers/depth_anything_v2/
```

## 3. 逐文件用途

| 文件 | 类型/阶段 | 用途与保留理由 |
|---|---|---|
| `augmentations.py` | 公共/文档 | 训练期图像增强函数。 |
| `dataset.py` | 数据 | 主训练数据集加载、裁剪/增强与可选合成雾调用。 |
| `depth_teachers/__init__.py` | 教师网络 | Python 包初始化与必要符号导出。 |
| `depth_teachers/depth_anything_v2/__init__.py` | 教师网络 | Depth Anything V2 包初始化。 |
| `depth_teachers/depth_anything_v2/checkpoints/depth_anything_v2_vitb.pth` | 教师网络 | Depth Anything V2 ViT-B 教师权重；仅深度先验预计算或 V2 教师分支需要。 |
| `depth_teachers/depth_anything_v2/dinov2_layers/__init__.py` | 教师网络 | Depth Anything V2 的 DINOv2 基础层（注意力/Block/MLP/嵌入等）；V2 推理核心需要。 |
| `depth_teachers/depth_anything_v2/dinov2_layers/attention.py` | 教师网络 | Depth Anything V2 的 DINOv2 基础层（注意力/Block/MLP/嵌入等）；V2 推理核心需要。 |
| `depth_teachers/depth_anything_v2/dinov2_layers/block.py` | 教师网络 | Depth Anything V2 的 DINOv2 基础层（注意力/Block/MLP/嵌入等）；V2 推理核心需要。 |
| `depth_teachers/depth_anything_v2/dinov2_layers/drop_path.py` | 教师网络 | Depth Anything V2 的 DINOv2 基础层（注意力/Block/MLP/嵌入等）；V2 推理核心需要。 |
| `depth_teachers/depth_anything_v2/dinov2_layers/layer_scale.py` | 教师网络 | Depth Anything V2 的 DINOv2 基础层（注意力/Block/MLP/嵌入等）；V2 推理核心需要。 |
| `depth_teachers/depth_anything_v2/dinov2_layers/mlp.py` | 教师网络 | Depth Anything V2 的 DINOv2 基础层（注意力/Block/MLP/嵌入等）；V2 推理核心需要。 |
| `depth_teachers/depth_anything_v2/dinov2_layers/patch_embed.py` | 教师网络 | Depth Anything V2 的 DINOv2 基础层（注意力/Block/MLP/嵌入等）；V2 推理核心需要。 |
| `depth_teachers/depth_anything_v2/dinov2_layers/swiglu_ffn.py` | 教师网络 | Depth Anything V2 的 DINOv2 基础层（注意力/Block/MLP/嵌入等）；V2 推理核心需要。 |
| `depth_teachers/depth_anything_v2/dinov2.py` | 教师网络 | Depth Anything V2 的 DINOv2 图像编码器；V2 推理需要。 |
| `depth_teachers/depth_anything_v2/dpt.py` | 教师网络 | Depth Anything V2 深度头与推理接口；precompute_depth_v2.py 需要。 |
| `depth_teachers/depth_anything_v2/util/blocks.py` | 教师网络 | Depth Anything V2 解码融合块与输入变换；V2 推理需要。 |
| `depth_teachers/depth_anything_v2/util/transform.py` | 教师网络 | Depth Anything V2 解码融合块与输入变换；V2 推理需要。 |
| `depth_teachers/ra_depth/__init__.py` | 教师网络 | Python 包初始化与必要符号导出。 |
| `depth_teachers/ra_depth/layers.py` | 教师网络 | RA-Depth 深度到视差等基础层；MSF 解码器导入。 |
| `depth_teachers/ra_depth/networks/__init__.py` | 教师网络 | Python 包初始化与必要符号导出。 |
| `depth_teachers/ra_depth/networks/depth_decoder_msf.py` | 教师网络 | RA-Depth 多尺度深度解码器；训练蒸馏/RA 深度预计算需要。 |
| `depth_teachers/ra_depth/networks/hrnet_config.py` | 教师网络 | RA-Depth HRNet 教师编码器及其结构配置；训练蒸馏/RA 深度预计算需要。 |
| `depth_teachers/ra_depth/networks/hrnet_encoder.py` | 教师网络 | RA-Depth HRNet 教师编码器及其结构配置；训练蒸馏/RA 深度预计算需要。 |
| `depth_teachers/ra_depth/weights/depth.pth` | 教师网络 | RA-Depth 教师权重；encoder.pth 为 HRNet 特征编码器，depth.pth 为 MSF 深度解码器；仅在启用 RA 深度蒸馏/预计算时加载。 |
| `depth_teachers/ra_depth/weights/encoder.pth` | 教师网络 | RA-Depth 教师权重；encoder.pth 为 HRNet 特征编码器，depth.pth 为 MSF 深度解码器；仅在启用 RA 深度蒸馏/预计算时加载。 |
| `haze_synthesizer.py` | 公共/文档 | 依据深度/透射率生成合成雾样本，供 dataset.py 训练增强。 |
| `loss/CR_loss.py` | 损失 | 对比损失；主训练需要。 |
| `model1/__init__.py` | 模型 | Python 包初始化与必要符号导出。 |
| `model1/model_convnext.py` | 模型 | 主模型集合与融合网络核心；主训练和统一测试的关键模型分支。 |
| `model1/myFFCResblock0.py` | 模型 | FFC 残差块适配层；主融合网络依赖。 |
| `models/__init__.py` | 模型 | Python 包初始化与必要符号导出。 |
| `models/DehazeXL/__init__.py` | 模型 | Python 包初始化与必要符号导出。 |
| `models/DehazeXL/backbones/__init__.py` | 模型 | DehazeXL 的 Swin 主干及包导出；多代解码器需要。 |
| `models/DehazeXL/backbones/swin.py` | 模型 | DehazeXL 的 Swin 主干及包导出；多代解码器需要。 |
| `models/DehazeXL/backbones/swin1.py` | 模型 | DehazeXL 的 Swin 主干及包导出；多代解码器需要。 |
| `models/DehazeXL/context_encoders/__init__.py` | 模型 | DehazeXL 可选上下文注意力组件；启用 context encoder 的模型分支需要。 |
| `models/DehazeXL/context_encoders/attention.py` | 模型 | DehazeXL 可选上下文注意力组件；启用 context encoder 的模型分支需要。 |
| `models/DehazeXL/context_encoders/attentionmla.py` | 模型 | DehazeXL 可选上下文注意力组件；启用 context encoder 的模型分支需要。 |
| `models/DehazeXL/context_encoders/hyper_attn/attention/angular_lsh.py` | 模型 | DehazeXL 可选上下文注意力组件；启用 context encoder 的模型分支需要。 |
| `models/DehazeXL/context_encoders/hyper_attn/attention/flash_attn_triton_for_hyper.py` | 模型 | DehazeXL 可选上下文注意力组件；启用 context encoder 的模型分支需要。 |
| `models/DehazeXL/context_encoders/hyper_attn/attention/hyper_attn.py` | 模型 | DehazeXL 可选上下文注意力组件；启用 context encoder 的模型分支需要。 |
| `models/DehazeXL/context_encoders/hyper_attn/attention/modeling_chatglm_fast_attention.py` | 模型 | DehazeXL 可选上下文注意力组件；启用 context encoder 的模型分支需要。 |
| `models/DehazeXL/context_encoders/hyper_attn/attention/utils.py` | 模型 | DehazeXL 可选上下文注意力组件；启用 context encoder 的模型分支需要。 |
| `models/DehazeXL/context_encoders/hyper_attn/replace_llm_attention.py` | 模型 | DehazeXL 可选上下文注意力组件；启用 context encoder 的模型分支需要。 |
| `models/DehazeXL/context_encoders/transformer_xl.py` | 模型 | DehazeXL 可选上下文注意力组件；启用 context encoder 的模型分支需要。 |
| `models/DehazeXL/decoders/__init__.py` | 模型 | DehazeXL 解码支持模块（频率融合、通用层或插件）；相应解码器需要。 |
| `models/DehazeXL/decoders/decoder.py` | 模型 | DehazeXL 解码器代际/工具；model_convnext 各兼容模型分支在导入或构造时需要。 |
| `models/DehazeXL/decoders/decoder1.py` | 模型 | DehazeXL 解码器代际/工具；model_convnext 各兼容模型分支在导入或构造时需要。 |
| `models/DehazeXL/decoders/decoder2.py` | 模型 | DehazeXL 解码器代际/工具；model_convnext 各兼容模型分支在导入或构造时需要。 |
| `models/DehazeXL/decoders/decoder3.py` | 模型 | DehazeXL 解码器代际/工具；model_convnext 各兼容模型分支在导入或构造时需要。 |
| `models/DehazeXL/decoders/decoder4.py` | 模型 | DehazeXL 解码器代际/工具；model_convnext 各兼容模型分支在导入或构造时需要。 |
| `models/DehazeXL/decoders/decoder5.py` | 模型 | DehazeXL 解码器代际/工具；model_convnext 各兼容模型分支在导入或构造时需要。 |
| `models/DehazeXL/decoders/FreqFusion.py` | 模型 | DehazeXL 解码支持模块（频率融合、通用层或插件）；相应解码器需要。 |
| `models/DehazeXL/decoders/plug_and_play.py` | 模型 | DehazeXL 解码支持模块（频率融合、通用层或插件）；相应解码器需要。 |
| `models/DehazeXL/decoders/utils.py` | 模型 | DehazeXL 解码支持模块（频率融合、通用层或插件）；相应解码器需要。 |
| `models2/__init__.py` | 模型 | models2 包导出或其内部网络支持文件；主融合网络需要。 |
| `models2/DepthNet.py` | 模型 | 内部深度估计分支；model_convnext/model_vssm 在构造主要模型时需要。 |
| `models2/DepthNet2.py` | 模型 | 内部深度估计分支；model_convnext/model_vssm 在构造主要模型时需要。 |
| `perceptual.py` | 损失 | VGG 特征感知损失封装，主训练需要。 |
| `precompute_depth_v2.py` | 预处理 | Depth Anything V2 浮点深度先验生成入口；默认使用已整理的 V2 包与 ViT-B 权重。 |
| `precompute_depth.py` | 预处理 | RA-Depth 离线深度图生成入口；读取教师编码器/解码器。 |
| `pytorch_msssim/__init__.py` | 公共/文档 | 项目内置 MS-SSIM 实现；主训练损失需要。 |
| `pytorch_ssim/__init__.py` | 公共/文档 | 项目内置兼容 SSIM 实现；统一测试需要。 |
| `README.md` | 公共/文档 | 精简版快速入口、目录约定与使用说明。 |
| `requirements.txt` | 公共/文档 | Python 运行依赖清单。 |
| `saicinpainting/__init__.py` | 公共/文档 | 最小 FFC 支持模块；myFFCResblock0 及主融合网络需要。 |
| `saicinpainting/training/__init__.py` | 公共/文档 | 最小 FFC 支持模块；myFFCResblock0 及主融合网络需要。 |
| `saicinpainting/training/modules/__init__.py` | 公共/文档 | 最小 FFC 支持模块；myFFCResblock0 及主融合网络需要。 |
| `saicinpainting/training/modules/base.py` | 公共/文档 | 最小 FFC 支持模块；myFFCResblock0 及主融合网络需要。 |
| `saicinpainting/training/modules/depthwise_sep_conv.py` | 公共/文档 | 最小 FFC 支持模块；myFFCResblock0 及主融合网络需要。 |
| `saicinpainting/training/modules/ffc0.py` | 公共/文档 | 最小 FFC 支持模块；myFFCResblock0 及主融合网络需要。 |
| `saicinpainting/training/modules/multidilated_conv.py` | 公共/文档 | 最小 FFC 支持模块；myFFCResblock0 及主融合网络需要。 |
| `saicinpainting/training/modules/spatial_transform.py` | 公共/文档 | 最小 FFC 支持模块；myFFCResblock0 及主融合网络需要。 |
| `saicinpainting/training/modules/squeeze_excitation.py` | 公共/文档 | 最小 FFC 支持模块；myFFCResblock0 及主融合网络需要。 |
| `test_fusion_net_depth_best.py` | 测试 | 唯一测试/TTA 入口；只注册 `fusion_net_depth_best`，加载 checkpoint，输出图像及 PSNR/SSIM/耗时统计。 |
| `train_fusion_net_depth_best.py` | 训练 | 唯一训练入口；命令行将模型版本限制为 13，支持 PSNR/感知/对比损失和可选 RA-Depth 蒸馏。 |
| `upsample/__init__.py` | 公共/文档 | Converse 上采样模块；DehazeXL decoder3 的兼容导入/模型分支需要。 |
| `upsample/util_converse.py` | 公共/文档 | Converse 上采样模块；DehazeXL decoder3 的兼容导入/模型分支需要。 |
| `utils/__init__.py` | 公共/文档 | 通用图像、并行、计时或指标辅助模块；保留用于模型/评估兼容。 |
| `utils/common.py` | 公共/文档 | 通用图像、并行、计时或指标辅助模块；保留用于模型/评估兼容。 |
| `utils/data_parallel.py` | 公共/文档 | 通用图像、并行、计时或指标辅助模块；保留用于模型/评估兼容。 |
| `utils/matlab_functions.py` | 公共/文档 | 通用图像、并行、计时或指标辅助模块；保留用于模型/评估兼容。 |
| `utils/metric_util.py` | 公共/文档 | 通用图像、并行、计时或指标辅助模块；保留用于模型/评估兼容。 |
| `utils/metrics.py` | 公共/文档 | PSNR/SSIM 指标；训练验证与测试需要。 |
| `utils/psnr_ssim.py` | 公共/文档 | 通用图像、并行、计时或指标辅助模块；保留用于模型/评估兼容。 |
| `utils/time.py` | 公共/文档 | 通用图像、并行、计时或指标辅助模块；保留用于模型/评估兼容。 |
| `utils/utils.py` | 公共/文档 | 通用图像、并行、计时或指标辅助模块；保留用于模型/评估兼容。 |

## 4. 明确排除的内容

- `__pycache__/`、`.pyc`、`.idea/`：解释器缓存和 IDE 元数据。
- `.synth_samples/`、`assets/`：生成样例和论文展示图片，不影响运行。
- `classification/`、`torchhub/`：外部完整仓库/缓存副本；主入口没有直接引用。
- `DS-DGVMamba/`、`dehaze2/`：独立实验工程，与根项目模型重复；当前保留的 VMamba/GeoPhys 分支已覆盖其有效方向。
- `Depth-Anything-V2-main` 除推理核心外的全部内容：演示、视频、metric-depth、Docker、文档和资产。
- RA-Depth 的数据集、训练器、评估/演示、位姿网络，以及 `pose_encoder.pth`、`pose.pth`：去雾蒸馏只使用 encoder/depth。
- 根目录历史脚本：`train.py` 至 `train5.py`、`train_best_weight.py`、`train_color*.py`、`train_depth.py`、`train_dense_haze.py`、`train_optimized.py`、`train_psnr.py`、`train_psnr_advanced.py`、`train_save5model.py`、无监督实验脚本，以及 `test.py` 至 `test6.py`、`test_tta2.py`、`testtta_color.py`。这些是同一模型演进中的旧入口或单一版本硬编码入口，已由统一入口替代。
- 重复/临时文件：`ss.py`（与 `augmentations.py` 内容相同）、`gen_vmamba_dg.py`、`_smoke_tta_weights.py`、临时编译文件及旧设计草稿。

## 5. 运行注意事项

1. 命令应在 `dehaze4` 根目录执行，确保相对权重路径正确。
2. `train_fusion_net_depth_best.py` 的 `--use_depth` 使用 RA-Depth；不开启时不加载教师权重。
3. `precompute_depth_v2.py` 默认配置为 ViT-B，并指向随项目保留的本地 ViT-B 权重；若选择 vits/vitl，必须另行提供匹配权重。
4. 原代码硬编码的 ConvNeXt-XL 初始化权重路径已改为 `weights/convnext_xlarge_22k_1k_384_ema.pth`。该权重在源项目中不存在，因此缺失时模型会告警并采用随机初始化；恢复训练 checkpoint 不受影响，从零训练若追求原始初始化效果需自行补入该文件。
5. 数据集和去雾模型 checkpoint 不属于源码，除两类教师权重外未复制任何训练产物。

## 6. 验证记录

- 对交付目录全部 Python 文件执行 `compileall`：通过。
- 对两个主入口和两个深度预计算入口执行 AST/编译检查：通过。
- 扫描旧目录硬编码引用并修正 RA-Depth、Depth Anything 的导入与默认权重路径。
- 使用用户环境 `D:/conda_envs/torch/python.exe` 验证：PyTorch `2.11.0+cpu`、torchvision `0.26.0+cpu`，CUDA 不可用。
- `test_fusion_net_depth_best.py --help` 与 `train_fusion_net_depth_best.py --help` 在用户指定环境中退出码均为 0。
- `fusion_net_depth_best(crop_size=64)` 构建成功，共 375,349,702 个参数。CPU 环境中的随机前向检查因模型体积较大未完成，因此未把“CPU 前向通过”写成验收结论。
- 当前环境尚缺 `matplotlib`、`yacs`、`scikit-image`、`tensorboard`、`thop`，因此 RA-Depth 预计算及涉及这些库的可选功能需先按 `requirements.txt` 补装依赖。
