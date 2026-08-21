# Dehaze 精简运行版

本目录由 `dehaze1` 按实际导入关系整理而来。推荐使用：

- 训练：`python train_fusion_net_depth_best.py --help`
- 测试/TTA：`python test_fusion_net_depth_best.py --help`
- RA-Depth 深度预计算：`python precompute_depth.py --help`
- Depth Anything V2 深度预计算：`python precompute_depth_v2.py --help`
- 专项分支：`train_geophys.py`、`train_night_v1.py`、`train_vmamba_depth.py`

数据集保持原约定：根目录包含 `GT/`、`hazy/`，使用深度先验的分支另需 `DepthTeacher/` 或对应脚本生成的深度目录。

深度教师已合并到 `depth_teachers/`：RA-Depth 只保留 HRNet 编码器、MSF 深度解码器和两份必需权重；Depth Anything 只保留 V1/V2 推理核心与本地 V2 ViT-B 权重。完整上游仓库中的训练、演示、视频、评估和缓存文件均未复制。

完整文件职责、保留理由、已排除内容和验证结果见 `PROJECT_FILE_MANIFEST.md`。
