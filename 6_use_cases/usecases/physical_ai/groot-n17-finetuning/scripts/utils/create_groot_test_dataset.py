"""
Create a HELD-OUT TEST dataset for GR00T evaluation.

Downloads the same BridgeData V2 source but uses episodes that were
NOT in the training set (episodes 600+), so we can measure generalization.

Training used: episodes 0-599 (--max-episodes 600)
This script uses: episodes 600-699 (100 test episodes by default)

Usage:
    python create_groot_test_dataset.py
    python create_groot_test_dataset.py --num-episodes 50
    python create_groot_test_dataset.py --start-episode 600 --num-episodes 200
"""

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "-U", "datasets", "huggingface_hub", "fsspec"])
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

ACTION_DIM = 7
FPS = 5


def interpolate_frames(frames, target_count=30):
    if len(frames) >= target_count:
        return frames[:target_count]
    result = []
    n_orig = len(frames)
    for i in range(target_count):
        src_pos = i * (n_orig - 1) / (target_count - 1)
        src_idx = int(src_pos)
        frac = src_pos - src_idx
        if src_idx >= n_orig - 1:
            result.append(frames[-1])
        elif frac < 0.01:
            result.append(frames[src_idx])
        else:
            img1 = np.array(frames[src_idx].convert("RGB"), dtype=np.float32)
            img2 = np.array(frames[src_idx + 1].convert("RGB"), dtype=np.float32)
            blended = ((1 - frac) * img1 + frac * img2).astype(np.uint8)
            result.append(Image.fromarray(blended))
    return result


def estimate_actions_from_optical_flow(frames, action_dim=7):
    actions = []
    for i in range(len(frames) - 1):
        img1 = np.array(frames[i].convert("RGB"))
        img2 = np.array(frames[i + 1].convert("RGB"))
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )
        mean_flow = np.mean(flow, axis=(0, 1))
        action = np.zeros(action_dim, dtype=np.float32)
        action[0] = mean_flow[0] / 100.0
        action[1] = mean_flow[1] / 100.0
        actions.append(action)
    actions.append(np.zeros(action_dim, dtype=np.float32))
    return np.array(actions, dtype=np.float32)


def images_to_video(images, output_path, fps=5):
    with tempfile.TemporaryDirectory() as tmpdir:
        for i, img in enumerate(images):
            if not isinstance(img, Image.Image):
                img = Image.fromarray(np.array(img))
            img.save(os.path.join(tmpdir, f"frame_{i:06d}.png"))
        cmd = [
            "ffmpeg", "-y", "-framerate", str(fps),
            "-i", os.path.join(tmpdir, "frame_%06d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-crf", "23", "-preset", "fast", output_path,
        ]
        subprocess.run(cmd, capture_output=True, check=True)


def create_test_dataset(output_path, start_episode, num_episodes):
    print("=== Downloading BridgeData V2 Scripted Images ===")
    ds = load_dataset("VyoJ/BridgeData-V2-Scripted-Images")
    data = ds["train"]

    total_available = len(data)
    end_episode = min(start_episode + num_episodes, total_available)
    actual_count = end_episode - start_episode

    print(f"Source: {total_available} episodes")
    print(f"Test set: episodes {start_episode} to {end_episode - 1} ({actual_count} episodes)")

    output_path = Path(output_path)
    data_dir = output_path / "data" / "chunk-000"
    video_dir = output_path / "videos" / "chunk-000" / "observation.images.front"
    meta_dir = output_path / "meta"
    for d in [data_dir, video_dir, meta_dir]:
        d.mkdir(parents=True, exist_ok=True)

    episodes_meta = []
    tasks_set = {}
    total_frames = 0
    skipped = 0
    all_actions, all_states, all_timestamps = [], [], []

    for local_idx, src_idx in enumerate(tqdm(
        range(start_episode, end_episode), desc="Processing test episodes"
    )):
        sample = data[src_idx]
        frames = [f for f in [sample["first_image"], sample["intermediate_image"], sample["frame_43_image"]] if f is not None]
        if len(frames) < 2:
            skipped += 1
            continue

        frames = interpolate_frames(frames, target_count=30)
        num_frames = len(frames)
        actions = estimate_actions_from_optical_flow(frames, ACTION_DIM)
        states = actions.copy()

        instruction = "move object to target"
        if instruction not in tasks_set:
            tasks_set[instruction] = len(tasks_set)
        task_index = tasks_set[instruction]

        all_actions.extend(actions.tolist())
        all_states.extend(states.tolist())
        all_timestamps.extend([fi / FPS for fi in range(num_frames)])

        rows = []
        for fi in range(num_frames):
            rows.append({
                "observation.state": states[fi].tolist(),
                "action": actions[fi].tolist(),
                "episode_index": local_idx, "frame_index": fi,
                "index": total_frames + fi, "task_index": task_index,
                "timestamp": fi / FPS,
            })
        pq.write_table(pa.Table.from_pylist(rows), data_dir / f"episode_{local_idx:06d}.parquet")

        try:
            images_to_video(frames, str(video_dir / f"episode_{local_idx:06d}.mp4"), fps=FPS)
        except Exception as e:
            print(f"Warning: Video failed for episode {local_idx}: {e}")
            skipped += 1
            continue

        episodes_meta.append({"episode_index": local_idx, "tasks": [instruction], "length": num_frames})
        total_frames += num_frames

    # Write metadata
    info = {
        "codebase_version": "v2.1", "robot_type": "bridge_v2",
        "total_episodes": len(episodes_meta), "total_frames": total_frames,
        "total_tasks": len(tasks_set), "chunks_size": 1000, "fps": FPS,
        "splits": {"train": f"0:{len(episodes_meta)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "action": {"dtype": "float32", "shape": [7], "names": ["x","y","z","roll","pitch","yaw","gripper"]},
            "observation.state": {"dtype": "float32", "shape": [7], "names": ["x","y","z","roll","pitch","yaw","gripper"]},
            "observation.images.front": {
                "dtype": "video", "shape": [480,640,3], "names": ["height","width","channels"],
                "info": {"video.height":480,"video.width":640,"video.codec":"av1","video.pix_fmt":"yuv420p",
                         "video.is_depth_map":False,"video.fps":FPS,"video.channels":3,"has_audio":False},
            },
            "timestamp": {"dtype":"float32","shape":[1],"names":None},
            "frame_index": {"dtype":"int64","shape":[1],"names":None},
            "episode_index": {"dtype":"int64","shape":[1],"names":None},
            "index": {"dtype":"int64","shape":[1],"names":None},
            "task_index": {"dtype":"int64","shape":[1],"names":None},
        },
        "total_chunks": 1, "total_videos": len(episodes_meta),
    }
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)
    with open(meta_dir / "episodes.jsonl", "w") as f:
        for ep in episodes_meta:
            f.write(json.dumps(ep) + "\n")
    with open(meta_dir / "tasks.jsonl", "w") as f:
        for task_text, task_idx in sorted(tasks_set.items(), key=lambda x: x[1]):
            f.write(json.dumps({"task_index": task_idx, "task": task_text}) + "\n")
    modality = {
        "video": {"front": {"original_key": "observation.images.front"}},
        "state": {"arm": {"start": 0, "end": 7}},
        "action": {"arm": {"start": 0, "end": 7}},
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }
    with open(meta_dir / "modality.json", "w") as f:
        json.dump(modality, f, indent=2)

    all_actions_np = np.array(all_actions, dtype=np.float32)
    all_states_np = np.array(all_states, dtype=np.float32)
    all_timestamps_np = np.array(all_timestamps, dtype=np.float32).reshape(-1, 1)

    def compute_stats(arr):
        return {
            "mean": np.mean(arr, axis=0).tolist(), "std": np.std(arr, axis=0).tolist(),
            "min": np.min(arr, axis=0).tolist(), "max": np.max(arr, axis=0).tolist(),
            "q01": np.percentile(arr, 1, axis=0).tolist(), "q99": np.percentile(arr, 99, axis=0).tolist(),
        }

    stats = {
        "action": compute_stats(all_actions_np),
        "observation.state": compute_stats(all_states_np),
        "timestamp": compute_stats(all_timestamps_np),
    }
    with open(meta_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nTest dataset complete: {len(episodes_meta)} episodes, {total_frames} frames")
    print(f"Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Create held-out test dataset for GR00T evaluation")
    parser.add_argument("--output-path", default="./datasets/bridge_lerobot_test")
    parser.add_argument("--start-episode", type=int, default=600)
    parser.add_argument("--num-episodes", type=int, default=100)
    args = parser.parse_args()
    create_test_dataset(args.output_path, args.start_episode, args.num_episodes)


if __name__ == "__main__":
    main()
