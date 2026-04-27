#!/usr/bin/env python3
"""
Cosmos Transfer 2.5 — Drive-Dreams Multiview Dataset Preparation

Extends the single-view preparation to all 7 camera views required
by the multiview training pipeline.

The multiview training expects this structure:
    drive_dreams_multiview_dataset/
    ├── videos/
    │   ├── ftheta_camera_front_wide_120fov/
    │   │   ├── clip_0000.mp4
    │   │   └── ...
    │   ├── ftheta_camera_cross_left_120fov/
    │   ├── ftheta_camera_cross_right_120fov/
    │   ├── ftheta_camera_rear_left_120fov/
    │   ├── ftheta_camera_rear_right_120fov/
    │   ├── ftheta_camera_front_tele_30fov/
    │   └── ftheta_camera_rear_tele_60fov/
    ├── control_input_hdmap_bbox/
    │   ├── ftheta_camera_front_wide_120fov/
    │   └── ... (same 7 camera dirs)
    └── captions/
        ├── ftheta_camera_front_wide_120fov/
        └── ... (same 7 camera dirs)

Drive-Dreams raw data has all 7 views in the generation and hdmap archives.
Each video filename encodes the camera view.

Usage:
    # Prepare all views, 200 clips per view
    python prepare_multiview_dataset.py \
        --raw-dir ~/drive_dreams_raw/cosmos_synthetic/single_view \
        --output-dir /fsx/datasets/drive_dreams_multiview \
        --num-clips 200

    # Quick test with 10 clips
    python prepare_multiview_dataset.py \
        --raw-dir ~/drive_dreams_raw/cosmos_synthetic/single_view \
        --output-dir ./test_multiview \
        --num-clips 10 --test-clips 5
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# ── Constants ──
TARGET_WIDTH = 1280
TARGET_HEIGHT = 704
TARGET_FPS = 24
FRAMES_PER_CLIP = 93

SOURCE_FPS_HDMAP = 30
SOURCE_FPS_GEN = 24

# The 7 camera views used by Cosmos Transfer multiview training
CAMERA_VIEWS = [
    "ftheta_camera_front_wide_120fov",
    "ftheta_camera_cross_left_120fov",
    "ftheta_camera_cross_right_120fov",
    "ftheta_camera_rear_left_120fov",
    "ftheta_camera_rear_right_120fov",
    "ftheta_camera_front_tele_30fov",
    "ftheta_camera_rear_tele_60fov",
]

ALL_WEATHERS = ["Sunny", "Rainy", "Snowy", "Night", "Foggy", "Morning", "Golden_hour"]


def convert_video(input_path, output_path, source_fps, target_fps=TARGET_FPS,
                  target_w=TARGET_WIDTH, target_h=TARGET_HEIGHT,
                  num_frames=FRAMES_PER_CLIP):
    """Read source video, subsample to target FPS, resize, write num_frames."""
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open: {input_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS) or source_fps
    step = actual_fps / target_fps
    frame_indices = [int(i * step) for i in range(num_frames)]

    if frame_indices[-1] >= total_frames:
        max_possible = int(total_frames / step)
        if max_possible < num_frames:
            cap.release()
            raise ValueError(f"Not enough frames: {total_frames} at {actual_fps}fps")
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
            if frame.shape[0] != target_h or frame.shape[1] != target_w:
                frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            writer.write(frame)
            idx_ptr += 1
        current_frame += 1

    writer.release()
    cap.release()
    return idx_ptr


def discover_multiview_scenes(raw_dir, weathers=None):
    """
    Discover scenes that have all 7 camera views available.

    Drive-Dreams organizes by scene UUID. Each scene has hdmap + generation
    videos for multiple camera views. We group by scene and only keep
    scenes where all 7 views are present.

    Returns list of scene dicts with per-view file paths.
    """
    raw_dir = Path(raw_dir)
    hdmap_dir = raw_dir / "hdmap"
    gen_dir = raw_dir / "generation"
    caption_dir = raw_dir / "caption"

    if not hdmap_dir.exists() or not gen_dir.exists():
        print(f"ERROR: hdmap/ or generation/ not found in {raw_dir}")
        sys.exit(1)

    # Index all generation files by (scene_id, camera_view)
    # Filename pattern: {uuid}_{ts}_{ts}_{variant}_{Weather}.mp4
    # For multiview, camera view is encoded in the path or filename
    scenes = defaultdict(lambda: {"hdmap": {}, "gen": {}, "caption": {}})

    # For the Drive-Dreams dataset, all views are in the same flat directory
    # We need to parse the filenames to extract scene ID and camera view
    # If the dataset has per-view subdirectories, handle that too
    for gen_path in sorted(gen_dir.rglob("*.mp4")):
        # Check if organized by camera subdirectory
        rel = gen_path.relative_to(gen_dir)
        parts = rel.parts

        if len(parts) == 2:
            # Organized: generation/camera_view/clip.mp4
            camera = parts[0]
            clip_name = parts[1]
        else:
            # Flat: generation/clip.mp4 (single-view dataset)
            camera = "front_wide"
            clip_name = parts[0]

        stem = Path(clip_name).stem
        # Extract scene base ID (everything before weather suffix)
        scene_parts = stem.rsplit("_", 1)
        weather = scene_parts[1] if len(scene_parts) == 2 else "Unknown"
        scene_id = scene_parts[0] if len(scene_parts) == 2 else stem

        if weathers and weather not in weathers:
            continue

        key = f"{scene_id}_{weather}"
        scenes[key]["gen"][camera] = gen_path
        scenes[key]["weather"] = weather
        scenes[key]["scene_id"] = scene_id

    # Index hdmap files
    for hdmap_path in sorted(hdmap_dir.rglob("*.mp4")):
        rel = hdmap_path.relative_to(hdmap_dir)
        parts = rel.parts
        if len(parts) == 2:
            camera = parts[0]
            clip_name = parts[1]
        else:
            camera = "front_wide"
            clip_name = parts[0]

        stem = Path(clip_name).stem
        # HDMap doesn't have weather suffix, match by scene_id
        for key, scene in scenes.items():
            if scene["scene_id"] == stem or key.startswith(stem):
                scene["hdmap"][camera] = hdmap_path

    # Index captions
    for cap_path in sorted(caption_dir.rglob("*.txt")):
        stem = cap_path.stem
        for key, scene in scenes.items():
            if key.startswith(stem) or stem.startswith(scene.get("scene_id", "")):
                scene["caption"] = cap_path

    # Filter to scenes with all views (or at least front_wide for single-view fallback)
    complete_scenes = []
    for key, scene in scenes.items():
        gen_views = set(scene["gen"].keys())
        if len(gen_views) >= len(CAMERA_VIEWS):
            complete_scenes.append(scene)
        elif "front_wide" in gen_views:
            # Single-view fallback — still usable
            complete_scenes.append(scene)

    print(f"[INFO] Total scenes discovered: {len(scenes)}")
    print(f"[INFO] Scenes with data: {len(complete_scenes)}")

    return complete_scenes


def prepare_multiview_dataset(args):
    """Main dataset preparation pipeline for multiview training."""
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)

    weathers = None
    if args.weather:
        weathers = [w.strip() for w in args.weather.split(",")]
        print(f"[INFO] Filtering to weathers: {weathers}")

    scenes = discover_multiview_scenes(raw_dir, weathers=weathers)
    if not scenes:
        print("ERROR: No scenes found. Check raw-dir path and extracted data.")
        sys.exit(1)

    total = min(args.num_clips, len(scenes)) if args.num_clips else len(scenes)
    test_count = args.test_clips or 0
    train_count = total - test_count

    train_scenes = scenes[:train_count]
    test_scenes = scenes[train_count:total] if test_count > 0 else []

    print(f"[INFO] Train: {train_count} scenes, Test: {test_count} scenes")

    for split_name, split_scenes in [("train", train_scenes), ("test", test_scenes)]:
        if not split_scenes:
            continue

        split_dir = output_dir if split_name == "train" else Path(str(output_dir) + "_test")
        print(f"\n{'='*60}")
        print(f" Processing {split_name}: {len(split_scenes)} scenes -> {split_dir}")
        print(f"{'='*60}")

        success = 0
        for i, scene in enumerate(tqdm(split_scenes, desc=f"  {split_name}")):
            clip_name = f"clip_{i:04d}"

            try:
                available_views = set(scene["gen"].keys())

                for camera in CAMERA_VIEWS:
                    if camera not in available_views:
                        # Skip views not available for this scene
                        continue

                    # Convert generation video
                    gen_out = split_dir / "videos" / camera / f"{clip_name}.mp4"
                    convert_video(scene["gen"][camera], gen_out, SOURCE_FPS_GEN)

                    # Convert HDMap control video (if available for this view)
                    if camera in scene.get("hdmap", {}):
                        hdmap_out = split_dir / "control_input_hdmap_bbox" / camera / f"{clip_name}.mp4"
                        convert_video(scene["hdmap"][camera], hdmap_out, SOURCE_FPS_HDMAP)

                    # Write caption
                    caption_text = ""
                    cap_path = scene.get("caption")
                    if cap_path and Path(cap_path).exists():
                        caption_text = Path(cap_path).read_text().strip()
                    else:
                        weather = scene.get("weather", "clear")
                        caption_text = f"Driving scene, {weather.lower()} weather, {camera} camera view."

                    caption_out = split_dir / "captions" / camera / f"{clip_name}.json"
                    os.makedirs(caption_out.parent, exist_ok=True)
                    with open(caption_out, "w") as f:
                        json.dump({"caption": caption_text}, f, indent=2)

                success += 1
            except Exception as e:
                print(f"\n  [WARN] Skipping scene {i}: {e}")
                continue

        # Write metadata
        meta = {
            "dataset": "Cosmos-Drive-Dreams Multiview",
            "split": split_name,
            "control_type": "hdmap_bbox",
            "mode": "multiview",
            "cameras": CAMERA_VIEWS,
            "num_scenes": success,
            "frames_per_clip": FRAMES_PER_CLIP,
            "fps": TARGET_FPS,
            "resolution": f"{TARGET_WIDTH}x{TARGET_HEIGHT}",
        }
        os.makedirs(split_dir, exist_ok=True)
        with open(split_dir / "dataset_info.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"  {split_name}: {success} scenes prepared at {split_dir}")

    print(f"\n{'='*60}")
    print(f" Dataset preparation complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare Drive-Dreams multiview dataset for Cosmos Transfer 2.5"
    )
    parser.add_argument("--raw-dir", required=True,
                        help="Path to extracted cosmos_synthetic/single_view/")
    parser.add_argument("--output-dir", default="./drive_dreams_multiview_dataset",
                        help="Output directory")
    parser.add_argument("--num-clips", type=int, default=None,
                        help="Total number of scenes to process (default: all)")
    parser.add_argument("--test-clips", type=int, default=0,
                        help="Number of scenes to hold out for evaluation")
    parser.add_argument("--weather", type=str, default=None,
                        help="Comma-separated weather filter (default: all)")
    args = parser.parse_args()

    if not Path(args.raw_dir).exists():
        print(f"ERROR: Raw data directory not found: {args.raw_dir}")
        sys.exit(1)

    prepare_multiview_dataset(args)
