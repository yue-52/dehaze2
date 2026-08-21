"""Precompute float relative-depth priors with official Depth Anything V2.

The script processes both ``GT`` and ``hazy`` with the same frozen teacher and
saves unquantized ``.npy`` arrays under ``DepthTeacher/clear`` and
``DepthTeacher/hazy``. By default the official near-depth output is negated so
that larger saved values consistently mean farther scene regions.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download


MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64,
             "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128,
             "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256,
             "out_channels": [256, 512, 1024, 1024]},
}

HF_MODELS = {
    "vits": (
        "depth-anything/Depth-Anything-V2-Small",
        "depth_anything_v2_vits.pth",
    ),
    "vitb": (
        "depth-anything/Depth-Anything-V2-Base",
        "depth_anything_v2_vitb.pth",
    ),
    "vitl": (
        "depth-anything/Depth-Anything-V2-Large",
        "depth_anything_v2_vitl.pth",
    ),
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate V23 clear/hazy float depth priors."
    )
    parser.add_argument(
        "--data_dir", required=True,
        help="Dataset root containing GT/ and hazy/.",
    )
    parser.add_argument(
        "--repo_dir", default="./depth_teachers",
        help="Directory containing the bundled depth_anything_v2 package.",
    )
    parser.add_argument(
        "--encoder", choices=tuple(MODEL_CONFIGS), default="vits",
        help="V2 encoder. vits is recommended for preprocessing speed.",
    )
    parser.add_argument(
        "--checkpoint", default="./depth_teachers/depth_anything_v2/checkpoints/depth_anything_v2_vitb.pth",
        help="Local .pth weight. If omitted, download the official HF weight.",
    )
    parser.add_argument("--input_size", type=int, default=518)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--depth_direction", choices=("far", "official"), default="far",
        help="'far' negates official output so larger values mean farther.",
    )
    parser.add_argument(
        "--preview_count", type=int, default=4,
        help="Number of normalized PNG previews saved per split.",
    )
    parser.add_argument(
        "--save_dtype", choices=("float16", "float32"), default="float32",
        help="Stored array precision. float16 halves disk usage.",
    )
    parser.add_argument(
        "--output_max_side", type=int, default=0,
        help="Downsample saved depth so its longest side is at most this value; 0 keeps original size.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def import_official_model(repo_dir):
    repo_dir = Path(repo_dir).resolve()
    module_file = repo_dir / "depth_anything_v2" / "dpt.py"
    if not module_file.is_file():
        raise FileNotFoundError(
            f"Official model code not found: {module_file}\n"
            "Clone https://github.com/DepthAnything/Depth-Anything-V2 first."
        )
    sys.path.insert(0, str(repo_dir))
    try:
        from depth_anything_v2.dpt import DepthAnythingV2
    except ImportError as error:
        raise ImportError(
            f"Failed to import official Depth Anything V2 from {repo_dir}"
        ) from error
    return DepthAnythingV2, repo_dir


def resolve_checkpoint(encoder, checkpoint):
    if checkpoint:
        path = Path(checkpoint).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path
    repo_id, filename = HF_MODELS[encoder]
    return Path(hf_hub_download(repo_id=repo_id, filename=filename))


def load_teacher(repo_dir, encoder, checkpoint, device):
    model_class, resolved_repo = import_official_model(repo_dir)
    checkpoint_path = resolve_checkpoint(encoder, checkpoint)
    model = model_class(**MODEL_CONFIGS[encoder])
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model = model.to(device).eval()
    return model, resolved_repo, checkpoint_path


def list_images(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory not found: {directory}")
    images = sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise ValueError(f"No images found in {directory}")
    stems = [path.stem for path in images]
    if len(stems) != len(set(stems)):
        raise ValueError(
            f"Duplicate file stems in {directory}; output .npy names would collide"
        )
    return images


def save_preview(depth, path):
    finite = depth[np.isfinite(depth)]
    if finite.size == 0:
        return
    low, high = np.percentile(finite, (2, 98))
    preview = np.clip((depth - low) / max(high - low, 1e-6), 0.0, 1.0)
    preview = np.round(preview * 65535).astype(np.uint16)
    success, encoded = cv2.imencode(path.suffix, preview)
    if not success:
        raise ValueError(f"Failed to encode preview: {path}")
    encoded.tofile(path)


def read_image(path):
    """Read an image through bytes to support non-ASCII Windows paths."""
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return image


@torch.inference_mode()
def process_split(
    model,
    input_dir,
    output_dir,
    preview_dir,
    input_size,
    direction,
    preview_count,
    overwrite,
    save_dtype="float32",
    output_max_side=0,
):
    images = list_images(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    completed = 0
    skipped = 0
    for index, image_path in enumerate(images):
        output_path = output_dir / f"{image_path.stem}.npy"
        if output_path.exists() and not overwrite:
            skipped += 1
            continue
        image = read_image(image_path)
        depth = np.asarray(
            model.infer_image(image, input_size), dtype=np.float32
        )
        if depth.shape != image.shape[:2]:
            raise ValueError(
                f"Teacher returned {depth.shape} for image {image.shape[:2]}"
            )
        if not np.isfinite(depth).all():
            raise FloatingPointError(f"Non-finite depth predicted for {image_path}")
        if direction == "far":
            depth = -depth
        if output_max_side > 0 and max(depth.shape) > output_max_side:
            scale = output_max_side / max(depth.shape)
            saved_size = (
                max(1, round(depth.shape[1] * scale)),
                max(1, round(depth.shape[0] * scale)),
            )
            depth = cv2.resize(depth, saved_size, interpolation=cv2.INTER_AREA)
        depth = depth.astype(save_dtype)
        np.save(output_path, depth, allow_pickle=False)
        if index < preview_count:
            save_preview(depth, preview_dir / f"{image_path.stem}.png")
        completed += 1
        if completed % 10 == 0 or completed + skipped == len(images):
            print(
                f"{input_dir.name}: {completed + skipped}/{len(images)} "
                f"(written={completed}, skipped={skipped})"
            )
    return {"images": len(images), "written": completed, "skipped": skipped}


def main(args):
    data_dir = Path(args.data_dir).resolve()
    output_root = data_dir / "DepthTeacher"
    device = resolve_device(args.device)
    model, repo_dir, checkpoint = load_teacher(
        args.repo_dir, args.encoder, args.checkpoint, device
    )
    print(f"Device: {device}")
    print(f"Model: Depth Anything V2 {args.encoder}")
    print(f"Checkpoint: {checkpoint}")

    clear_stats = process_split(
        model=model,
        input_dir=data_dir / "GT",
        output_dir=output_root / "clear",
        preview_dir=output_root / "preview_clear",
        input_size=args.input_size,
        direction=args.depth_direction,
        preview_count=args.preview_count,
        overwrite=args.overwrite,
        save_dtype=args.save_dtype,
        output_max_side=args.output_max_side,
    )
    hazy_stats = process_split(
        model=model,
        input_dir=data_dir / "hazy",
        output_dir=output_root / "hazy",
        preview_dir=output_root / "preview_hazy",
        input_size=args.input_size,
        direction=args.depth_direction,
        preview_count=args.preview_count,
        overwrite=args.overwrite,
        save_dtype=args.save_dtype,
        output_max_side=args.output_max_side,
    )
    metadata = {
        "teacher": "Depth Anything V2",
        "encoder": args.encoder,
        "official_repo": str(repo_dir),
        "checkpoint": str(checkpoint),
        "input_size": args.input_size,
        "saved_dtype": args.save_dtype,
        "output_max_side": args.output_max_side,
        "per_image_normalized": False,
        "depth_direction": (
            "larger_is_far"
            if args.depth_direction == "far"
            else "official_larger_is_near"
        ),
        "clear": clear_stats,
        "hazy": hazy_stats,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    with open(output_root / "metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
    print(f"Done. Float priors: {output_root}")


if __name__ == "__main__":
    main(parse_args())
