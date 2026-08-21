import os
import csv
import json
import argparse
import time
import random
from glob import glob
from typing import Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm

try:
    BICUBIC_RESAMPLE = Image.Resampling.BICUBIC
except AttributeError:
    BICUBIC_RESAMPLE = getattr(Image, "BICUBIC", 3)

# Models
from model1.model_convnext import fusion_net_depth_best

# Metrics
from utils.metrics import psnr, ssim as calc_ssim
from pytorch_ssim import ssim as legacy_ssim


MODEL_REGISTRY = {
    "fusion_net_depth_best": lambda crop_size: fusion_net_depth_best(crop_size=crop_size),
}

AUTO_CROP_PAD_MODELS = {
    "fusion_net_depth_best",
}

DEFAULT_TTA_WEIGHTS = {
    # Order: id, h, v, hv
    "hvflip": [0.40, 0.20, 0.20, 0.20],
    # Order: id, r90, r180, r270, h, hr90, hr180, hr270
    "d4": [0.30, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10],
}


def configure_determinism(seed=0):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def pad_image_reflect(image, multiple=32):
    h, w = image.shape[2], image.shape[3]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h > 0 or pad_w > 0:
        image = F.pad(image, (0, pad_w, 0, pad_h), mode="reflect")
    return image, h, w


def _find_model_required_crop_multiple(model):
    """Infer crop alignment requirement from DehazeXL-like submodules."""
    candidates = []

    for module in model.modules():
        crop = getattr(module, "crop_size", None)
        if not isinstance(crop, int) or crop <= 1:
            continue

        # DehazeXL-style models expose nested_tokenization and require H/W divisibility by crop_size.
        if hasattr(module, "nested_tokenization") or "dehazexl" in module.__class__.__name__.lower():
            candidates.append(crop)

    return max(candidates) if candidates else None


def _effective_pad_multiple(model_name, model, args):
    required_crop = _find_model_required_crop_multiple(model)
    if required_crop is None:
        return args.pad_multiple, None
    return max(args.pad_multiple, required_crop), required_crop


def crop_image(image, h, w):
    return image[:, :, :h, :w]


class DehazeTestDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir

        if os.path.exists(os.path.join(root_dir, "hazy")):
            self.hazy_dir = os.path.join(root_dir, "hazy")
        else:
            self.hazy_dir = root_dir

        gt_candidates = [
            os.path.join(root_dir, "gt"),
            os.path.join(root_dir, "GT"),
            os.path.join(root_dir, "clear"),
            root_dir.replace("hazy", "GT"),
        ]
        self.clear_dir = next((p for p in gt_candidates if os.path.exists(p)), None)

        self.hazy_files = sorted(glob(os.path.join(self.hazy_dir, "*.*")))
        self.hazy_files = [
            f
            for f in self.hazy_files
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
        ]

        print(f"Found {len(self.hazy_files)} images in {self.hazy_dir}")
        if self.clear_dir is None:
            print("No GT folder found. Will run in inference-only mode.")
        else:
            print(f"GT folder: {self.clear_dir}")

        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.hazy_files)

    def __getitem__(self, idx):
        hazy_path = self.hazy_files[idx]
        file_name = os.path.basename(hazy_path)

        hazy_img = Image.open(hazy_path).convert("RGB")
        hazy_tensor = self.transform(hazy_img)

        has_clear = False
        clear_tensor = torch.zeros_like(hazy_tensor)
        if self.clear_dir is not None:
            clear_path = os.path.join(self.clear_dir, file_name)
            if os.path.exists(clear_path):
                clear_img = Image.open(clear_path).convert("RGB")
                if clear_img.size != hazy_img.size:
                    clear_img = clear_img.resize(hazy_img.size, BICUBIC_RESAMPLE)
                clear_tensor = self.transform(clear_img)
                has_clear = True

        return hazy_tensor, clear_tensor, file_name, has_clear


def _forward_tensor(model, x):
    out = model(x)
    if isinstance(out, (tuple, list)):
        out = out[0]
    if not torch.is_tensor(out):
        raise TypeError(f"Model output is {type(out)}, expected Tensor/tuple[list] containing Tensor")
    return out


def tta_predict(model, x, mode="hvflip"):
    return _tta_predict_with_weights(model, x, mode=mode, agg="mean", custom_weights=None)[0]


def _parse_tta_weights(raw_text: str):
    if not raw_text:
        return None
    vals = [v.strip() for v in raw_text.split(",") if v.strip()]
    if not vals:
        return None
    weights = [float(v) for v in vals]
    if any(w < 0 for w in weights):
        raise ValueError("--tta_weights must be non-negative")
    if sum(weights) <= 0:
        raise ValueError("--tta_weights sum must be > 0")
    return weights


def _aggregate_tta_outputs(outs, mode="hvflip", agg="mean", custom_weights=None):
    if not outs:
        raise RuntimeError("No TTA outputs collected")

    if agg == "mean":
        weights = [1.0 / len(outs)] * len(outs)
    else:
        if custom_weights is not None:
            if len(custom_weights) != len(outs):
                raise ValueError(
                    f"--tta_weights length mismatch: got {len(custom_weights)}, expected {len(outs)} for mode={mode}"
                )
            weights = list(custom_weights)
        else:
            weights = DEFAULT_TTA_WEIGHTS.get(mode, [1.0 / len(outs)] * len(outs))
            if len(weights) != len(outs):
                weights = [1.0 / len(outs)] * len(outs)

        s = float(sum(weights))
        if s <= 0:
            raise ValueError("TTA weights sum must be > 0")
        weights = [w / s for w in weights]

    stack = torch.stack(outs, dim=0)
    w = torch.tensor(weights, dtype=stack.dtype, device=stack.device).view(-1, 1, 1, 1, 1)
    out = torch.sum(stack * w, dim=0)
    return out, weights


def _tta_predict_with_weights(model, x, mode="hvflip", agg="mean", custom_weights=None):
    outs = []

    def add_variant(x_variant, inverse_fn):
        out_variant = _forward_tensor(model, x_variant)
        outs.append(inverse_fn(out_variant))

    if mode == "d4":
        variants = [
            (lambda t: t, lambda t: t),
            (lambda t: torch.rot90(t, 1, [2, 3]), lambda t: torch.rot90(t, 3, [2, 3])),
            (lambda t: torch.rot90(t, 2, [2, 3]), lambda t: torch.rot90(t, 2, [2, 3])),
            (lambda t: torch.rot90(t, 3, [2, 3]), lambda t: torch.rot90(t, 1, [2, 3])),
            (lambda t: torch.flip(t, [3]), lambda t: torch.flip(t, [3])),
            (lambda t: torch.rot90(torch.flip(t, [3]), 1, [2, 3]), lambda t: torch.flip(torch.rot90(t, 3, [2, 3]), [3])),
            (lambda t: torch.rot90(torch.flip(t, [3]), 2, [2, 3]), lambda t: torch.flip(torch.rot90(t, 2, [2, 3]), [3])),
            (lambda t: torch.rot90(torch.flip(t, [3]), 3, [2, 3]), lambda t: torch.flip(torch.rot90(t, 1, [2, 3]), [3])),
        ]
        for apply_fn, inverse_fn in variants:
            add_variant(apply_fn(x), inverse_fn)
        return _aggregate_tta_outputs(outs, mode=mode, agg=agg, custom_weights=custom_weights)

    outs.append(_forward_tensor(model, x))

    if mode in ("hflip", "hvflip"):
        x_h = torch.flip(x, [3])
        out_h = _forward_tensor(model, x_h)
        outs.append(torch.flip(out_h, [3]))

    if mode in ("vflip", "hvflip"):
        x_v = torch.flip(x, [2])
        out_v = _forward_tensor(model, x_v)
        outs.append(torch.flip(out_v, [2]))

    if mode == "hvflip":
        x_hv = torch.flip(x, [2, 3])
        out_hv = _forward_tensor(model, x_hv)
        outs.append(torch.flip(out_hv, [2, 3]))

    return _aggregate_tta_outputs(outs, mode=mode, agg=agg, custom_weights=custom_weights)


def postprocess_prediction(x, output_range="clamp"):
    if output_range == "unit_interval":
        return torch.clamp((x + 1.0) * 0.5, 0, 1)
    return torch.clamp(x, 0, 1)


def _calc_ssim_legacy_nhaze(pred, clear):
    # Keep the historical NH-HAZE SSIM protocol used by earlier test scripts.
    _, _, h, w = pred.size()
    down_ratio = max(1, round(min(h, w) / 256))
    pooled_h = int(h / down_ratio)
    pooled_w = int(w / down_ratio)
    pred_pool = F.adaptive_avg_pool2d(pred, (pooled_h, pooled_w))
    clear_pool = F.adaptive_avg_pool2d(clear, (pooled_h, pooled_w))
    return float(legacy_ssim(pred_pool, clear_pool, size_average=False).item())


def compute_ssim(pred, clear, mode):
    ssim_full = float(calc_ssim(pred, clear).item())
    ssim_legacy = _calc_ssim_legacy_nhaze(pred, clear)

    if mode == "full":
        selected = ssim_full
    elif mode == "legacy_nhaze":
        selected = ssim_legacy
    else:  # mode == "both"
        selected = ssim_full

    return selected, ssim_full, ssim_legacy


def _resolve_eval_defaults(args):
    """Apply profile-specific defaults without overriding explicit CLI choices."""
    if args.eval_profile == "legacy":
        if args.ssim_mode == "full":
            args.ssim_mode = "legacy_nhaze"
        if args.tta_mode is None:
            args.tta_mode = "d4"
        if args.output_range is None:
            args.output_range = "clamp"
        if args.pad_multiple == 32:
            args.pad_multiple = 512
        args.metric_clamp = True
        args.deterministic = True
    else:
        if args.tta_mode is None:
            args.tta_mode = "hvflip"
        if args.output_range is None:
            args.output_range = "clamp"
        args.metric_clamp = True


def _parse_model_names(model_names_arg):
    if model_names_arg.strip().lower() == "all":
        return list(MODEL_REGISTRY.keys())

    names = [x.strip() for x in model_names_arg.split(",") if x.strip()]
    invalid = [n for n in names if n not in MODEL_REGISTRY]
    if invalid:
        raise ValueError(f"Unknown model names: {invalid}. Available: {list(MODEL_REGISTRY.keys())}")
    return names


def _parse_checkpoint_map(raw_map):
    mapping = {}
    if not raw_map:
        return mapping
    # Format: model_a=path_a,model_b=path_b
    for item in raw_map.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid checkpoint_map item: '{item}', expected model=path")
        k, v = item.split("=", 1)
        mapping[k.strip()] = v.strip()
    return mapping


def _resolve_checkpoint(model_name, args, checkpoint_map):
    if model_name in checkpoint_map:
        return checkpoint_map[model_name]

    if args.checkpoint_dir:
        direct = os.path.join(args.checkpoint_dir, f"{model_name}.pth")
        if os.path.exists(direct):
            return direct

    # Backward compatibility: single model + single checkpoint
    if args.checkpoint and len(_parse_model_names(args.models)) == 1:
        return args.checkpoint

    return None


def _clean_state_dict(state_dict):
    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            cleaned[k[7:]] = v
        else:
            cleaned[k] = v
    return cleaned


def _inspect_weight_mismatch(model, loaded_state_dict):
    model_state = model.state_dict()

    missing = [k for k in model_state.keys() if k not in loaded_state_dict]
    unexpected = [k for k in loaded_state_dict.keys() if k not in model_state]
    shape_mismatch = []

    for k in loaded_state_dict.keys():
        if k in model_state and hasattr(loaded_state_dict[k], "shape"):
            if model_state[k].shape != loaded_state_dict[k].shape:
                shape_mismatch.append(
                    {
                        "name": k,
                        "model_shape": tuple(model_state[k].shape),
                        "ckpt_shape": tuple(loaded_state_dict[k].shape),
                    }
                )

    return missing, unexpected, shape_mismatch


def load_checkpoint_flexible(model, checkpoint_path, device):
    if checkpoint_path is None:
        raise FileNotFoundError("Checkpoint path is None")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        # Older torch versions do not support weights_only.
        checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint and isinstance(checkpoint["model"], dict):
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    state_dict = _clean_state_dict(state_dict)

    try:
        model.load_state_dict(state_dict, strict=True)
        return "strict"
    except RuntimeError as strict_err:
        missing, unexpected, shape_mismatch = _inspect_weight_mismatch(model, state_dict)

        if missing or unexpected or shape_mismatch:
            print("[Checkpoint Mismatch] strict load failed, falling back to filtered non-strict load.")
            print(
                f"[Checkpoint Mismatch] missing={len(missing)}, "
                f"unexpected={len(unexpected)}, shape_mismatch={len(shape_mismatch)}"
            )

            for item in shape_mismatch[:10]:
                print(
                    "  - shape mismatch: "
                    f"{item['name']} model={item['model_shape']} ckpt={item['ckpt_shape']}"
                )

            if len(shape_mismatch) > 10:
                print(f"  - ... and {len(shape_mismatch) - 10} more shape mismatches")

        # strict=False still errors on shape mismatch, so drop incompatible keys first.
        filtered_state_dict = {
            k: v
            for k, v in state_dict.items()
            if (k in model.state_dict()) and (model.state_dict()[k].shape == v.shape)
        }

        missing_after, unexpected_after = model.load_state_dict(filtered_state_dict, strict=False)
        return (
            "non-strict "
            f"(missing={len(missing_after)}, unexpected={len(unexpected_after)}, "
            f"shape_mismatch={len(shape_mismatch)}, strict_error={str(strict_err).splitlines()[0]})"
        )


def run_single_model(model_name, args, device, loader, checkpoint_path):
    model_out_dir = os.path.join(args.output_dir, model_name)
    os.makedirs(model_out_dir, exist_ok=True)

    model = MODEL_REGISTRY[model_name](args.crop_size).to(device)
    load_mode = load_checkpoint_flexible(model, checkpoint_path, device)
    model.eval()

    pad_multiple, required_crop = _effective_pad_multiple(model_name, model, args)
    if required_crop is not None and args.pad_multiple < required_crop:
        print(
            f"[Warning] {model_name} requires crop-size alignment (crop_size={required_crop}). "
            f"Auto-upgrading --pad_multiple from {args.pad_multiple} to {pad_multiple}."
        )
    elif pad_multiple != args.pad_multiple:
        print(f"[Info] {model_name}: using effective pad_multiple={pad_multiple} for inference.")

    times, psnrs, ssims = [], [], []
    ssims_full, ssims_legacy = [], []
    per_image_rows = []
    custom_tta_weights = _parse_tta_weights(args.tta_weights)
    effective_tta_weights = None

    pbar = tqdm(loader, desc=f"[{model_name}]", leave=False)
    with torch.no_grad():
        for hazy, clear, name, has_clear in pbar:
            filename = str(name[0])
            row: Dict[str, Any] = {
                "image": filename,
                "ok": 1,
                "time_s": None,
                "psnr": None,
                "ssim": None,
                "ssim_full": None,
                "ssim_legacy_nhaze": None,
                "error": "",
            }
            try:
                hazy = hazy.to(device)
                clear = clear.to(device)

                hazy_pad, h, w = pad_image_reflect(hazy, multiple=pad_multiple)

                t0 = time.time()
                pred_pad, used_weights = _tta_predict_with_weights(
                    model,
                    hazy_pad,
                    mode=args.tta_mode,
                    agg=args.tta_agg,
                    custom_weights=custom_tta_weights,
                )
                if effective_tta_weights is None:
                    effective_tta_weights = [float(w) for w in used_weights]
                t1 = time.time()

                pred_raw = crop_image(pred_pad, h, w)
                pred_vis = postprocess_prediction(pred_raw, output_range=args.output_range)
                pred_metric = pred_vis if args.metric_clamp else pred_raw

                save_image(pred_vis, os.path.join(str(model_out_dir), filename))

                dt = t1 - t0
                row["time_s"] = dt
                times.append(dt)

                if bool(has_clear.item()):
                    p = float(psnr(pred_metric, clear))
                    row["psnr"] = p
                    psnrs.append(p)

                    ssim_val, ssim_full, ssim_legacy = compute_ssim(pred_metric, clear, args.ssim_mode)
                    row["ssim"] = ssim_val
                    row["ssim_full"] = ssim_full
                    row["ssim_legacy_nhaze"] = ssim_legacy
                    ssims.append(ssim_val)
                    ssims_full.append(ssim_full)
                    ssims_legacy.append(ssim_legacy)
            except Exception as e:
                row["ok"] = 0
                row["error"] = str(e)

            per_image_rows.append(row)

    failed_rows = [r for r in per_image_rows if r["ok"] == 0]
    if per_image_rows and not times:
        print(f"[Warning] {model_name}: all {len(per_image_rows)} images failed; summary will be empty.")
        for item in failed_rows[:3]:
            print(f"  - {item['image']}: {item['error']}")
        if len(failed_rows) > 3:
            print(f"  - ... and {len(failed_rows) - 3} more failures")

    # Save per-image csv
    csv_path = os.path.join(str(model_out_dir), "per_image_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "ok",
                "time_s",
                "psnr",
                "ssim",
                "ssim_full",
                "ssim_legacy_nhaze",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(per_image_rows)

    summary = {
        "model": model_name,
        "checkpoint": checkpoint_path,
        "load_mode": load_mode,
        "num_images": len(per_image_rows),
        "num_success": int(sum(r["ok"] for r in per_image_rows)),
        "avg_time_s": float(np.mean(times)) if times else None,
        "avg_psnr": float(np.mean(psnrs)) if psnrs else None,
        "avg_ssim": float(np.mean(ssims)) if ssims else None,
        "avg_ssim_full": float(np.mean(ssims_full)) if ssims_full else None,
        "avg_ssim_legacy_nhaze": float(np.mean(ssims_legacy)) if ssims_legacy else None,
        "ssim_mode": args.ssim_mode,
        "metric_clamp": bool(args.metric_clamp),
        "eval_profile": args.eval_profile,
        "tta_agg": args.tta_agg,
        "tta_weights": effective_tta_weights,
        "output_range": args.output_range,
        "deterministic": bool(args.deterministic),
        "seed": int(args.seed),
        "tta_mode": args.tta_mode,
        "pad_multiple": pad_multiple,
    }

    with open(os.path.join(str(model_out_dir), "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Unified multi-model TTA dehazing benchmark")
    parser.add_argument("--input_dir", type=str, required=True, help='Input dir containing "hazy" folder or images')
    parser.add_argument("--output_dir", type=str, default="results/TTA", help="Output root dir")

    # Checkpoint options
    parser.add_argument("--checkpoint", type=str, default=None, help="Single checkpoint path (single-model mode)")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory containing <model_name>.pth")
    parser.add_argument(
        "--checkpoint_map",
        type=str,
        default="",
        help="Comma list: model=path,model=path. Overrides checkpoint_dir/checkpoint.",
    )

    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help=f"'all' or comma-separated names. Available: {list(MODEL_REGISTRY.keys())}",
    )
    parser.add_argument(
        "--ssim_mode",
        type=str,
        default="full",
        choices=["full", "legacy_nhaze", "both"],
        help="SSIM protocol: full (utils.metrics), legacy_nhaze (downsampled pytorch_ssim), both (report both).",
    )
    parser.add_argument("--tta_mode", type=str, default=None, choices=["none", "hflip", "vflip", "hvflip", "d4"])
    parser.add_argument("--tta_agg", type=str, default="mean", choices=["mean", "weighted"], help="TTA aggregation method")
    parser.add_argument("--tta_weights", type=str, default="", help="Optional comma-separated weights overriding default TTA weights")
    parser.add_argument(
        "--eval_profile",
        type=str,
        default="modern",
        choices=["modern", "legacy"],
        help="Convenience profile: modern uses clamped metrics + full SSIM, legacy mirrors older test behavior.",
    )
    parser.add_argument("--pad_multiple", type=int, default=32, help="Pad input H/W to this multiple before inference")
    parser.add_argument("--crop_size", type=int, default=256, help="Model crop_size constructor argument")
    parser.add_argument(
        "--output_range",
        type=str,
        default=None,
        choices=["clamp", "unit_interval"],
        help="Post-process model output before metrics/saving",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")
    parser.add_argument("--deterministic", action="store_true", help="Enable deterministic inference for reproducibility")
    parser.add_argument("--seed", type=int, default=0, help="Random seed used when --deterministic is enabled")

    args = parser.parse_args()
    _resolve_eval_defaults(args)

    os.makedirs(args.output_dir, exist_ok=True)
    if args.deterministic:
        configure_determinism(seed=args.seed)
        print(f"Deterministic mode enabled (seed={args.seed})")
    device = torch.device("cuda:1" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")
    print(
        f"Eval profile: {args.eval_profile} | TTA={args.tta_mode} | SSIM={args.ssim_mode} | "
        f"output_range={args.output_range} | metric_clamp={args.metric_clamp}"
    )

    dataset = DehazeTestDataset(args.input_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model_names = _parse_model_names(args.models)
    checkpoint_map = _parse_checkpoint_map(args.checkpoint_map)

    all_summaries = []
    for model_name in model_names:
        ckpt = _resolve_checkpoint(model_name, args, checkpoint_map)
        if ckpt is None:
            print(f"[Skip] {model_name}: no checkpoint configured")
            all_summaries.append(
                {
                    "model": model_name,
                    "checkpoint": None,
                    "status": "skipped_no_checkpoint",
                }
            )
            continue

        print(f"\n=== Testing {model_name} ===")
        print(f"Checkpoint: {ckpt}")

        try:
            summary = run_single_model(model_name, args, device, loader, ckpt)
            summary["status"] = "ok"
            all_summaries.append(summary)
            print(
                f"{model_name} | time={summary['avg_time_s']} | "
                f"PSNR={summary['avg_psnr']} | SSIM={summary['avg_ssim']} | profile={summary['eval_profile']}"
            )
        except Exception as e:
            print(f"[Failed] {model_name}: {e}")
            all_summaries.append(
                {
                    "model": model_name,
                    "checkpoint": ckpt,
                    "status": "failed",
                    "error": str(e),
                }
            )

    # Global summary
    summary_json = os.path.join(args.output_dir, "all_models_summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)

    summary_csv = os.path.join(args.output_dir, "all_models_summary.csv")
    csv_fields = [
        "model",
        "status",
        "checkpoint",
        "num_images",
        "num_success",
        "avg_time_s",
        "avg_psnr",
        "avg_ssim",
        "avg_ssim_full",
        "avg_ssim_legacy_nhaze",
        "ssim_mode",
        "metric_clamp",
        "eval_profile",
        "tta_agg",
        "tta_weights",
        "output_range",
        "deterministic",
        "seed",
        "load_mode",
        "error",
    ]
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for item in all_summaries:
            row = {k: item.get(k, None) for k in csv_fields}
            writer.writerow(row)

    print("\nDone. Global summary:")
    print(f"- {summary_json}")
    print(f"- {summary_csv}")


if __name__ == "__main__":
    main()

