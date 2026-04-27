#!/usr/bin/env python3
"""
Cosmos Transfer 2.5 — Drive-Dreams Dataset Preparation Script (v2)

Organizes the downloaded NVIDIA Cosmos-Drive-Dreams dataset into
Cosmos Transfer 2.5 single-view HDMap training format.

Source: https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicle-Cosmos-Drive-Dreams

Actual dataset structure (after extraction):
    cosmos_synthetic/single_view/
    ├── hdmap/          — 12,000 MP4 control videos (1280x704, 30fps, 121 frames)
    │   └── {uuid}_{ts_start}_{ts_end}_{variant}.mp4
    ├── generation/     — 84,000 MP4 driving videos (1280x704, 24fps, 121 frames)
    │   └── {uuid}_{ts_start}_{ts_end}_{variant}_{Weather}.mp4
    └── caption/        — 84,000 TXT captions
        └── {uuid}_{ts_start}_{ts_end}_{variant}_{Weather}.txt

Naming: one HDMap clip maps to 7 weather variants (Sunny, Rainy, Snowy,
Night, Foggy, Morning, Golden_hour).

Cosmos Transfer expects: 1280x720, 10fps, 29 frames per clip.

Output (Cosmos Transfer single-view format):
    drive_dreams_cosmos_dataset/
    ├── videos/
    │   └── front_wide/
    │       ├── clip_0000.mp4
    │       └── ...
    ├── control_input_hdmap_bbox/
    │   └── front_wide/
    │       ├── clip_0000.mp4
    │       └── ...
    └── captions/
        └── front_wide/
            ├── clip_0000.json
            └── ...

Prerequisites:
    pip install opencv-python-headless numpy tqdm

Usage:
    # Organize already-downloaded data
    python prepare_drive_dreams_dataset.py \
        --raw-dir ~/cosmos_drive_dreams/drive_dreams_raw/cosmos_synthetic/single_view \
        --output-dir ./drive_dreams_cosmos_dataset

    # Limit to N clips
    python prepare_drive_dreams_dataset.py \
        --raw-dir ~/cosmos_drive_dreams/drive_dreams_raw/cosmos_synthetic/single_view \
        --output-dir ./drive_dreams_cosmos_dataset \
        --num-clips 200

    # Filter by weather
    python prepare_drive_dreams_dataset.py \
        --raw-dir ~/cosmos_drive_dreams/drive_dreams_raw/cosmos_synthetic/single_view \
        --output-dir ./drive_dreams_cosmos_dataset \
        --weather Sunny,Rainy

    # Hold out test set
    python prepare_drive_dreams_dataset.py \
        --raw-dir ~/cosmos_drive_dreams/drive_dreams_raw/cosmos_synthetic/single_view \
        --output-dir ./drive_dreams_cosmos_dataset \
        --num-clips 200 \
        --test-clips 50
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# ============================================================
# Constants
# ============================================================
TARGET_WIDTH = 1280
TARGET_HEIGHT = 704  # Source is 704, keep original resolution
TARGET_FPS = 24  # Match source generation fps to get enough frames
FRAMES_PER_CLIP = 93  # Cosmos Transfer expects 93 pixel frames (state_t=24)

SOURCE_WIDTH = 1280
SOURCE_HEIGHT = 704
SOURCE_FPS_HDMAP = 30
SOURCE_FPS_GEN = 24
SOURCE_FRAMES = 121

CAMERA_NAME = "front_wide"  # Single-view, front camera

# Weather variants in the dataset
ALL_WEATHERS = ["Sunny", "Rainy", "Snowy", "Night", "Foggy", "Morning", "Golden_hour"]


def parse_hdmap_filename(filename):
    """
    Parse HDMap filename: {uuid}_{ts_start}_{ts_end}_{variant}.mp4
    Returns (base_id, variant) where base_id = uuid_ts_start_ts_end
    """
    stem = Path(filename).stem
    # Last part after final underscore is variant (0 or 1)
    parts = stem.rsplit("_", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return stem, "0"


def parse_generation_filename(filename):
    """
    Parse generation filename: {uuid}_{ts_start}_{ts_end}_{variant}_{Weather}.mp4
    Returns (base_id, variant, weather) where base_id = uuid_ts_start_ts_end
    """
    stem = Path(filename).stem
    # Last part is weather, second to last is variant
    parts = stem.rsplit("_", 2)
    if len(parts) == 3:
        base_id = parts[0]
        variant = parts[1]
        weather = parts[2]
        return base_id, variant, weather
    return stem, "0", "Unknown"


def convert_video(input_path, output_path, source_fps, target_fps=TARGET_FPS,
                  target_w=TARGET_WIDTH, target_h=TARGET_HEIGHT,
                  num_frames=FRAMES_PER_CLIP):
    """
    Read source video, subsample to target FPS, resize to target resolution,
    and write exactly num_frames as MP4.
    """
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open: {input_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS) or source_fps

    # Calculate frame indices to sample
    # We want num_frames at target_fps from a source at actual_fps
    # Step = actual_fps / target_fps (e.g., 30/10=3 or 24/10=2.4)
    step = actual_fps / target_fps
    frame_indices = [int(i * step) for i in range(num_frames)]

    # Make sure we don't exceed total frames
    if frame_indices[-1] >= total_frames:
        # Adjust: use available frames
        max_possible = int(total_frames / step)
        if max_possible < num_frames:
            cap.release()
            raise ValueError(
                f"Not enough frames: {total_frames} at {actual_fps}fps "
                f"can only produce {max_possible} frames at {target_fps}fps"
            )
        frame_indices = [int(i * step) for i in range(num_frames)]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, target_fps, (target_w, target_h))

    current_frame = 0
    idx_ptr = 0

    while idx_ptr < len(frame_indices):
        ret, frame = cap.read()
        if not ret:
            break

        if current_frame == frame_indices[idx_ptr]:
            # Resize 1280x704 -> 1280x720
            if frame.shape[0] != target_h or frame.shape[1] != target_w:
                frame = cv2.resize(frame, (target_w, target_h),
                                   interpolation=cv2.INTER_LINEAR)
            writer.write(frame)
            idx_ptr += 1

        current_frame += 1

    writer.release()
    cap.release()

    return idx_ptr  # Number of frames written


def discover_pairs(raw_dir, weathers=None):
    """
    Discover HDMap → generation video pairs.

    Returns list of dicts:
        {hdmap_path, gen_path, caption_path, base_id, variant, weather}
    """
    raw_dir = Path(raw_dir)
    hdmap_dir = raw_dir / "hdmap"
    gen_dir = raw_dir / "generation"
    caption_dir = raw_dir / "caption"

    if not hdmap_dir.exists():
        print(f"ERROR: HDMap directory not found: {hdmap_dir}")
        sys.exit(1)

    # Index HDMap files by (base_id, variant)
    hdmap_index = {}
    for f in sorted(hdmap_dir.glob("*.mp4")):
        base_id, variant = parse_hdmap_filename(f.name)
        key = f"{base_id}_{variant}"
        hdmap_index[key] = f

    print(f"[INFO] HDMap videos: {len(hdmap_index)}")

    # Index generation files
    gen_files = list(gen_dir.glob("*.mp4")) if gen_dir.exists() else []
    print(f"[INFO] Generation videos: {len(gen_files)}")

    # Index caption files
    caption_files = {Path(f).stem: f for f in caption_dir.glob("*.txt")} if caption_dir.exists() else {}
    print(f"[INFO] Caption files: {len(caption_files)}")

    # Build pairs
    pairs = []
    for gen_path in sorted(gen_files):
        base_id, variant, weather = parse_generation_filename(gen_path.name)

        # Filter by weather if specified
        if weathers and weather not in weathers:
            continue

        # Find matching HDMap
        hdmap_key = f"{base_id}_{variant}"
        if hdmap_key not in hdmap_index:
            continue

        # Find matching caption
        caption_key = gen_path.stem
        caption_path = caption_files.get(caption_key)

        pairs.append({
            "hdmap_path": hdmap_index[hdmap_key],
            "gen_path": gen_path,
            "caption_path": caption_path,
            "base_id": base_id,
            "variant": variant,
            "weather": weather,
        })

    print(f"[INFO] Matched pairs: {len(pairs)}")

    # Show weather distribution
    weather_counts = {}
    for p in pairs:
        weather_counts[p["weather"]] = weather_counts.get(p["weather"], 0) + 1
    print(f"[INFO] Weather distribution:")
    for w, c in sorted(weather_counts.items()):
        print(f"    {w}: {c}")

    return pairs


def prepare_dataset(args):
    """Main dataset preparation pipeline."""
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)

    # Parse weather filter
    weathers = None
    if args.weather:
        weathers = [w.strip() for w in args.weather.split(",")]
        print(f"[INFO] Filtering to weathers: {weathers}")

    # Discover pairs
    pairs = discover_pairs(raw_dir, weathers=weathers)

    if not pairs:
        print("\nERROR: No matched pairs found.")
        print("Make sure hdmap.tar.gz and generation.tar.gz.part-* are extracted.")
        sys.exit(1)

    # Split train/test
    total_clips = min(args.num_clips, len(pairs)) if args.num_clips else len(pairs)
    test_clips = args.test_clips if args.test_clips else 0

    if test_clips >= total_clips:
        print(f"ERROR: test_clips ({test_clips}) >= total clips ({total_clips})")
        sys.exit(1)

    train_clips = total_clips - test_clips
    train_pairs = pairs[:train_clips]
    test_pairs = pairs[train_clips:total_clips] if test_clips > 0 else []

    print(f"\n[INFO] Will create {train_clips} training clips + {test_clips} test clips")

    # Process train and test sets
    for split_name, split_pairs in [("train", train_pairs), ("test", test_pairs)]:
        if not split_pairs:
            continue

        if split_name == "test":
            split_dir = output_dir.parent / (output_dir.name + "_test")
        else:
            split_dir = output_dir

        video_dir = split_dir / "videos" / CAMERA_NAME
        hdmap_dir = split_dir / "control_input_hdmap_bbox" / CAMERA_NAME
        caption_dir = split_dir / "captions" / CAMERA_NAME

        os.makedirs(video_dir, exist_ok=True)
        os.makedirs(hdmap_dir, exist_ok=True)
        os.makedirs(caption_dir, exist_ok=True)

        print(f"\n{'='*50}")
        print(f" Processing {split_name} set: {len(split_pairs)} clips -> {split_dir}")
        print(f"{'='*50}")

        success = 0
        for i, pair in enumerate(tqdm(split_pairs, desc=f"  {split_name}")):
            clip_name = f"clip_{i:04d}"

            try:
                # Convert generation video (RGB ground truth)
                n_gen = convert_video(
                    pair["gen_path"],
                    video_dir / f"{clip_name}.mp4",
                    source_fps=SOURCE_FPS_GEN,
                )

                # Convert HDMap control video
                n_hdmap = convert_video(
                    pair["hdmap_path"],
                    hdmap_dir / f"{clip_name}.mp4",
                    source_fps=SOURCE_FPS_HDMAP,
                )

                # Write caption as JSON (Cosmos Transfer format)
                caption_text = ""
                if pair["caption_path"] and pair["caption_path"].exists():
                    caption_text = pair["caption_path"].read_text().strip()
                else:
                    caption_text = (
                        f"Driving scene, {pair['weather'].lower()} weather, "
                        f"urban environment with HD map overlay."
                    )

                caption_json = {"caption": caption_text}
                with open(caption_dir / f"{clip_name}.json", "w") as f:
                    json.dump(caption_json, f, indent=2)

                success += 1

            except Exception as e:
                print(f"\n  [WARN] Skipping {pair['gen_path'].name}: {e}")
                continue

        # Write metadata
        meta = {
            "dataset": "Cosmos-Drive-Dreams (NVIDIA)",
            "source": "https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicle-Cosmos-Drive-Dreams",
            "split": split_name,
            "control_type": "hdmap_bbox",
            "mode": "single_view",
            "camera": CAMERA_NAME,
            "num_clips": success,
            "frames_per_clip": FRAMES_PER_CLIP,
            "fps": TARGET_FPS,
            "resolution": f"{TARGET_WIDTH}x{TARGET_HEIGHT}",
            "source_resolution": f"{SOURCE_WIDTH}x{SOURCE_HEIGHT}",
            "license": "CC BY 4.0",
        }
        with open(split_dir / "dataset_info.json", "w") as f:
            json.dump(meta, f, indent=2)

        total_size = sum(f.stat().st_size for f in split_dir.rglob("*") if f.is_file())
        print(f"\n  {split_name}: {success} clips, {total_size / (1024**3):.2f} GB")
        print(f"  Output: {split_dir}")

    print(f"\n{'='*50}")
    print(f" Dataset preparation complete!")
    print(f"{'='*50}")
    print(f"\nNext: copy to GPU instance for training:")
    print(f"  scp -r {output_dir} ec2-p4de:~/cosmos_drive_dreams/cosmos-transfer2.5/assets/")
    if test_clips > 0:
        test_dir = output_dir.parent / (output_dir.name + "_test")
        print(f"  scp -r {test_dir} ec2-p4de:~/cosmos_drive_dreams/cosmos-transfer2.5/assets/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare Drive-Dreams dataset for Cosmos Transfer 2.5 single-view fine-tuning"
    )
    parser.add_argument(
        "--raw-dir", required=True,
        help="Path to extracted cosmos_synthetic/single_view/ directory",
    )
    parser.add_argument(
        "--output-dir",
        default="./drive_dreams_cosmos_dataset",
        help="Output directory for training dataset (default: ./drive_dreams_cosmos_dataset)",
    )
    parser.add_argument(
        "--num-clips", type=int, default=None,
        help="Total number of clips to create (default: all available)",
    )
    parser.add_argument(
        "--test-clips", type=int, default=0,
        help="Number of clips to hold out for evaluation (default: 0)",
    )
    parser.add_argument(
        "--weather", type=str, default=None,
        help="Comma-separated weather filter, e.g. 'Sunny,Rainy' (default: all)",
    )

    args = parser.parse_args()

    if not Path(args.raw_dir).exists():
        print(f"ERROR: Raw data directory not found: {args.raw_dir}")
        print(f"\nDownload and extract first:")
        print(f"  python download.py --odir ./drive_dreams_raw --file_types synthetic --workers 4")
        print(f"  cd drive_dreams_raw/cosmos_synthetic/single_view")
        print(f"  tar xzf hdmap.tar.gz")
        print(f"  tar xzf generation.tar.gz.part-000")
        sys.exit(1)

    prepare_dataset(args)
