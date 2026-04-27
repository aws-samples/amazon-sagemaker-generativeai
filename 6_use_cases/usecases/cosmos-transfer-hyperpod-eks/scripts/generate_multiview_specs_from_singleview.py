#!/usr/bin/env python3
"""
Two-Stage Multiview Pipeline — Step 2: Generate Multiview Specs

Takes single-view inference outputs (from the fine-tuned Drive-Dreams model)
and generates multiview inference specs that feed them into the base
Cosmos Transfer multiview model to produce 7-camera surround views.

The idea:
  Stage 1: Fine-tuned single-view model generates front_wide from HDMap control
  Stage 2: Base multiview model extends front_wide to all 7 camera views

This script creates the multiview spec JSON files for Stage 2.

Usage:
    python generate_multiview_specs_from_singleview.py \
        --singleview-output /fsx/output/singleview_inference \
        --control-dir /fsx/datasets/drive_dreams_cosmos_dataset/control_input_hdmap_bbox \
        --caption-dir /fsx/datasets/drive_dreams_cosmos_dataset/captions \
        --output-dir /fsx/output/multiview_specs

The multiview model expects a spec with paths for each camera view.
For the two-stage approach:
  - front_wide: uses the GENERATED video from Stage 1 as input
  - other views: uses the HDMap control signal (same for all views)
  - The multiview model generates consistent views from the front_wide input
"""

import argparse
import json
import os
from pathlib import Path


# Camera views used by Cosmos Transfer multiview
VIEW_MAP = {
    "front_wide":  "ftheta_camera_front_wide_120fov",
    "cross_left":  "ftheta_camera_cross_left_120fov",
    "cross_right": "ftheta_camera_cross_right_120fov",
    "rear_left":   "ftheta_camera_rear_left_70fov",
    "rear_right":  "ftheta_camera_rear_right_70fov",
    "rear":        "ftheta_camera_rear_tele_30fov",
    "front_tele":  "ftheta_camera_front_tele_30fov",
}


def generate_specs(args):
    singleview_dir = Path(args.singleview_output)
    control_dir = Path(args.control_dir)
    caption_dir = Path(args.caption_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all generated single-view videos
    generated_videos = sorted(singleview_dir.glob("**/*.mp4"))
    if not generated_videos:
        print(f"ERROR: No generated videos found in {singleview_dir}")
        return

    print(f"Found {len(generated_videos)} single-view generated videos")

    count = 0
    for gen_video in generated_videos:
        clip_name = gen_video.stem  # e.g., clip_0000

        # Find matching HDMap control
        control_path = control_dir / f"{clip_name}.mp4"
        if not control_path.exists():
            # Try with front_wide subfolder
            control_path = control_dir / "front_wide" / f"{clip_name}.mp4"
        if not control_path.exists():
            print(f"  [SKIP] No control for {clip_name}")
            continue

        # Load caption
        caption_path = caption_dir / f"{clip_name}.json"
        if not caption_path.exists():
            caption_path = caption_dir / "front_wide" / f"{clip_name}.json"

        caption = "A driving scene captured from multiple camera views."
        if caption_path.exists():
            with open(caption_path) as f:
                data = json.load(f)
            caption = data.get("caption", caption)

        # Create spec directory
        spec_dir = output_dir / clip_name
        spec_dir.mkdir(parents=True, exist_ok=True)

        # Write prompt
        with open(spec_dir / "prompt.txt", "w") as f:
            f.write(caption)

        # Build multiview spec
        # front_wide uses the GENERATED video as input (Stage 1 output)
        # All views use the same HDMap control signal
        spec = {
            "name": f"multiview_{clip_name}",
            "prompt_path": str(spec_dir / "prompt.txt"),
            "fps": 24,
        }

        for view_key, folder_name in VIEW_MAP.items():
            if view_key == "front_wide":
                # Use the generated single-view video as input
                spec[view_key] = {
                    "input_path": str(gen_video.resolve()),
                    "control_path": str(control_path.resolve()),
                }
            else:
                # Other views: provide control signal, model generates the view
                # The multiview model uses cross-view attention from front_wide
                spec[view_key] = {
                    "input_path": "",  # No input — model generates this view
                    "control_path": str(control_path.resolve()),
                }

        spec_path = spec_dir / "multiview_spec.json"
        with open(spec_path, "w") as f:
            json.dump(spec, f, indent=2)

        count += 1

    print(f"\nGenerated {count} multiview specs in {output_dir}")
    print(f"\nNext: run multiview inference with the base model:")
    print(f"  torchrun --nproc_per_node=8 --master_port=12341 \\")
    print(f"    -m examples.multiview \\")
    print(f"    -i <spec_dir>/multiview_spec.json \\")
    print(f"    -o /fsx/output/multiview_output \\")
    print(f"    --experiment transfer2_auto_multiview_post_train_example \\")
    print(f"    --disable-guardrails")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate multiview specs from single-view inference outputs"
    )
    parser.add_argument("--singleview-output", required=True,
                        help="Directory with generated single-view videos from Stage 1")
    parser.add_argument("--control-dir", required=True,
                        help="HDMap control signal directory")
    parser.add_argument("--caption-dir", required=True,
                        help="Caption directory")
    parser.add_argument("--output-dir", default="./multiview_specs",
                        help="Output directory for multiview spec files")
    args = parser.parse_args()
    generate_specs(args)
