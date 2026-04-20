"""
Data utility functions for OpenVLA fine-tuning pipeline.
Includes optical flow action estimation, dataset conversion, and validation.
"""
import os
import json
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
from tqdm import tqdm


ACTION_DIM = 7


def estimate_actions_from_optical_flow(frames, action_dim=7):
    """
    Estimate 7-DoF actions from optical flow between consecutive frames.
    Returns actions of shape (T, action_dim).
    """
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


def validate_episode(ep_path, action_dim=7):
    """Validate a single episode directory for completeness and correctness."""
    actions_file = os.path.join(ep_path, "actions.npy")
    if not os.path.exists(actions_file):
        return False, "missing actions.npy"

    try:
        actions = np.load(actions_file, allow_pickle=True)
        if actions.dtype == object:
            actions = np.array(actions.tolist(), dtype=np.float32)

        image_dir = os.path.join(ep_path, "images")
        has_images = len([f for f in os.listdir(image_dir)
                         if f.endswith((".jpg", ".png"))]) > 0
        has_lang = os.path.exists(os.path.join(ep_path, "language.txt"))
        no_nan = not np.any(np.isnan(actions)) and not np.any(np.isinf(actions))

        if has_images and has_lang and no_nan and actions.shape[-1] == action_dim:
            return True, "valid"
        else:
            return False, "invalid data"
    except Exception as e:
        return False, str(e)


def validate_dataset_dir(dataset_root, action_dim=7):
    """Validate all episodes in a dataset directory."""
    episodes = sorted([d for d in os.listdir(dataset_root)
                       if os.path.isdir(os.path.join(dataset_root, d))
                       and d.startswith("episode_")])

    valid, invalid, missing = 0, 0, 0
    for ep_name in episodes:
        ep_path = os.path.join(dataset_root, ep_name)
        is_valid, reason = validate_episode(ep_path, action_dim)
        if is_valid:
            valid += 1
        elif reason == "missing actions.npy":
            missing += 1
        else:
            invalid += 1

    return {"total": len(episodes), "valid": valid, "invalid": invalid, "missing": missing}


def convert_episodes_to_hf(source_dir, output_dir):
    """Convert episode folders to HuggingFace DatasetDict format."""
    from datasets import Dataset, DatasetDict
    import random

    episodes = sorted([e for e in os.listdir(source_dir)
                       if os.path.isdir(os.path.join(source_dir, e))
                       and e.startswith("episode_")])

    data = []
    failed = 0
    for episode in tqdm(episodes, desc="Converting to HF"):
        ep_path = os.path.join(source_dir, episode)
        try:
            image_dir = os.path.join(ep_path, "images")
            image_files = sorted([f for f in os.listdir(image_dir)
                                  if f.endswith((".jpg", ".png")) and not f.startswith(".")])
            if not image_files:
                failed += 1
                continue

            imgs = [Image.open(os.path.join(image_dir, f)) for f in image_files]
            actions = np.load(os.path.join(ep_path, "actions.npy"), allow_pickle=True)

            if actions.dtype == object:
                actions = np.array(actions.tolist(), dtype=np.float32)
            if np.any(np.isnan(actions)) or np.any(np.isinf(actions)):
                failed += 1
                continue

            with open(os.path.join(ep_path, "language.txt")) as f:
                language = f.read().strip()

            data.append({
                "observation/image": imgs,
                "actions": actions.tolist(),
                "language_instruction": language,
            })
        except Exception:
            failed += 1

    random.seed(42)
    random.shuffle(data)
    split_idx = int(0.9 * len(data))

    dataset_dict = DatasetDict({
        "train": Dataset.from_list(data[:split_idx]),
        "validation": Dataset.from_list(data[split_idx:]),
    })
    dataset_dict.save_to_disk(output_dir)
    return {"total": len(episodes), "converted": len(data), "failed": failed}


def validate_hf_dataset(dataset_path):
    """Validate a HuggingFace dataset for NaN, Inf, and shape issues."""
    from datasets import load_from_disk

    dataset = load_from_disk(dataset_path)
    results = {}

    for split_name in dataset.keys():
        split = dataset[split_name]
        issues = {"nan": 0, "inf": 0, "wrong_shape": 0}

        for idx in range(len(split)):
            actions = np.array(split[idx]["actions"])
            if np.any(np.isnan(actions)):
                issues["nan"] += 1
            if np.any(np.isinf(actions)):
                issues["inf"] += 1
            if actions.ndim == 2 and actions.shape[1] != 7:
                issues["wrong_shape"] += 1
            elif actions.ndim == 1 and actions.shape[0] != 7:
                issues["wrong_shape"] += 1

        results[split_name] = {"total": len(split), **issues}

    return results
