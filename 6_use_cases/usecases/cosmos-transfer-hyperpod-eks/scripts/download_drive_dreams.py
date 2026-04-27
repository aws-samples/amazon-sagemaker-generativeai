#!/usr/bin/env python3
"""
download_drive_dreams.py
------------------------
Download the Cosmos-Drive-Dreams dataset from HuggingFace.

Two data types available:

  SYNTHETIC (tar archives under cosmos_synthetic/single_view/):
    - hdmap.tar.gz              (~2 GB)   — 12,000 HDMap control videos
    - generation.tar.gz.part-*  (~700 GB) — 84,000 driving videos (17 split parts)
    - caption.tar.gz            (~50 MB)  — 84,000 text captions

  REAL (individual files in top-level folders):
    - 3d_crosswalks, 3d_lanelines, 3d_lanes, 3d_poles, 3d_road_boundaries,
      3d_road_markings, 3d_traffic_lights, 3d_traffic_signs, 3d_wait_lines
    - all_object_info, captions, car_mask_coarse
    - ftheta_intrinsic, pinhole_intrinsic, pose, vehicle_pose
    - lidar_raw

Usage:
    # Synthetic HDMap + captions only (~2 GB, good for testing pipeline)
    python download_drive_dreams.py --odir ./drive_dreams_raw --components hdmap,caption

    # All synthetic data (~700 GB)
    python download_drive_dreams.py --odir ./drive_dreams_raw

    # Specific generation parts (each ~40 GB)
    python download_drive_dreams.py --odir ./drive_dreams_raw --gen-parts 0,1,2

    # Real-world data (HD map annotations, poses, intrinsics, etc.)
    python download_drive_dreams.py --odir ./drive_dreams_raw --components real

    # Specific real-world folders
    python download_drive_dreams.py --odir ./drive_dreams_raw --components real --real-folders 3d_lanes,captions,pose

    # Real + synthetic HDMap
    python download_drive_dreams.py --odir ./drive_dreams_raw --components hdmap,caption,real

    # Everything (synthetic + real)
    python download_drive_dreams.py --odir ./drive_dreams_raw --components hdmap,caption,generation,real

    # Auto-extract synthetic tar archives after download
    python download_drive_dreams.py --odir ./drive_dreams_raw --extract

Prerequisites:
    pip install huggingface_hub tqdm
    huggingface-cli login  # gated dataset, need access first
"""

from __future__ import annotations
import argparse
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

from huggingface_hub import HfApi, HfFileSystem
from tqdm import tqdm

DATASET_REPO = "nvidia/PhysicalAI-Autonomous-Vehicle-Cosmos-Drive-Dreams"
SYNTHETIC_PREFIX = "cosmos_synthetic/single_view"

REAL_FOLDERS = [
    "3d_crosswalks", "3d_lanelines", "3d_lanes", "3d_poles",
    "3d_road_boundaries", "3d_road_markings", "3d_traffic_lights",
    "3d_traffic_signs", "3d_wait_lines",
    "all_object_info", "captions", "car_mask_coarse",
    "ftheta_intrinsic", "pinhole_intrinsic", "pose", "vehicle_pose",
    "lidar_raw",
]

api = HfApi()
fs = HfFileSystem()


def verify_access() -> bool:
    try:
        fs.ls(f"datasets/{DATASET_REPO}")
        return True
    except Exception:
        return False


def list_synthetic_files() -> dict[str, list[str]]:
    """List synthetic tar archives, grouped by component."""
    all_files = fs.ls(f"datasets/{DATASET_REPO}/{SYNTHETIC_PREFIX}", detail=False)
    components = {"hdmap": [], "caption": [], "generation": []}
    for path in all_files:
        rel = path[len(f"datasets/{DATASET_REPO}/"):]
        name = path.split("/")[-1]
        if name.startswith("hdmap"):
            components["hdmap"].append(rel)
        elif name.startswith("caption"):
            components["caption"].append(rel)
        elif name.startswith("generation"):
            components["generation"].append(rel)
    return components


def list_real_files(folders: list[str] | None = None) -> list[str]:
    """List individual files under real-world data folders."""
    target_folders = folders if folders else REAL_FOLDERS
    prefix = f"datasets/{DATASET_REPO}/"
    rel_paths = []
    for folder in target_folders:
        full_path = f"{prefix}{folder}"
        try:
            files = fs.find(full_path)
            for f in files:
                rel_paths.append(f[len(prefix):])
        except Exception as e:
            print(f"  [WARN] Could not list {folder}: {e}")
    return rel_paths


def download_file(rel_path: str, odir: str, max_retries: int = 5) -> bool:
    """Download a single file with retries and exponential backoff."""
    local_target = os.path.join(odir, rel_path)
    if os.path.exists(local_target):
        size_mb = os.path.getsize(local_target) / (1024 * 1024)
        print(f"  [SKIP] Already exists: {rel_path} ({size_mb:.1f} MB)")
        return True

    for attempt in range(max_retries):
        try:
            api.hf_hub_download(
                repo_id=DATASET_REPO,
                filename=rel_path,
                repo_type="dataset",
                local_dir=odir,
            )
            size_mb = os.path.getsize(local_target) / (1024 * 1024)
            print(f"  [OK] {rel_path} ({size_mb:.1f} MB)")
            return True
        except KeyboardInterrupt:
            raise
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"  [RETRY {attempt+1}/{max_retries}] {rel_path}: {e} (waiting {wait}s)")
                time.sleep(wait)
            else:
                print(f"  [FAIL] {rel_path}: {e}")
    return False


def download_files_parallel(rel_paths: list[str], odir: str, workers: int = 4) -> int:
    """Download multiple files in parallel. Returns success count."""
    success = 0

    def _download(rel: str) -> tuple[str, bool]:
        return rel, download_file(rel, odir)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_download, rel): rel for rel in rel_paths}
        for f in tqdm(as_completed(futures), total=len(futures), desc="  Downloading", unit="file"):
            rel, ok = f.result()
            success += ok
            if not ok:
                print(f"  [FAIL] {rel}")

    return success


def extract_tar(tar_path: str, extract_dir: str):
    print(f"  Extracting {tar_path} -> {extract_dir}")
    os.makedirs(extract_dir, exist_ok=True)
    subprocess.run(["tar", "xzf", tar_path, "-C", extract_dir], check=True)


def extract_split_tar(parts: list[str], extract_dir: str):
    print(f"  Extracting {len(parts)} split parts -> {extract_dir}")
    os.makedirs(extract_dir, exist_ok=True)
    cat_cmd = ["cat"] + sorted(parts)
    tar_cmd = ["tar", "xzf", "-", "-C", extract_dir]
    cat_proc = subprocess.Popen(cat_cmd, stdout=subprocess.PIPE)
    tar_proc = subprocess.Popen(tar_cmd, stdin=cat_proc.stdout)
    cat_proc.stdout.close()
    tar_proc.communicate()
    if tar_proc.returncode != 0:
        print(f"  [ERROR] Extraction failed with return code {tar_proc.returncode}")


def main():
    parser = argparse.ArgumentParser(description="Download Cosmos-Drive-Dreams dataset")
    parser.add_argument("--odir", required=True, help="Output directory")
    parser.add_argument(
        "--components", default="hdmap,caption,generation",
        help="Comma-separated: hdmap, caption, generation, real (default: hdmap,caption,generation)",
    )
    parser.add_argument(
        "--gen-parts", default=None,
        help="Generation part numbers to download, e.g. '0,1,2' (default: all)",
    )
    parser.add_argument(
        "--real-folders", default=None,
        help="Specific real-data folders to download, e.g. '3d_lanes,captions,pose' (default: all)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit real data to first N files per folder (for quick testing)",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Parallel download threads for real data (default: 4)",
    )
    parser.add_argument("--extract", action="store_true", help="Extract synthetic tar archives after download")
    args = parser.parse_args()

    os.makedirs(args.odir, exist_ok=True)

    if not verify_access():
        print(f"ERROR: No access to {DATASET_REPO}")
        print("  1. Accept license: https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicle-Cosmos-Drive-Dreams")
        print("  2. Login: huggingface-cli login")
        sys.exit(1)

    requested = {c.strip() for c in args.components.split(",")}
    synthetic_components = requested & {"hdmap", "caption", "generation"}
    download_real = "real" in requested

    total_success = 0
    total_files = 0

    # --- Synthetic data (tar archives) ---
    if synthetic_components:
        print("\nListing synthetic files...")
        components = list_synthetic_files()
        synthetic_files = []

        for comp in synthetic_components:
            comp_files = components.get(comp, [])
            if comp == "generation" and args.gen_parts:
                wanted = {int(p.strip()) for p in args.gen_parts.split(",")}
                comp_files = [f for f in comp_files if any(f.endswith(f"part-{p:03d}") for p in wanted)]
            synthetic_files.extend(comp_files)
            print(f"  {comp}: {len(comp_files)} files")

        if synthetic_files:
            print(f"\nDownloading {len(synthetic_files)} synthetic archives...")
            print("=" * 50)
            for rel_path in synthetic_files:
                if download_file(rel_path, args.odir):
                    total_success += 1
                total_files += 1

    # --- Real data (individual files) ---
    if download_real:
        real_folders = None
        if args.real_folders:
            real_folders = [f.strip() for f in args.real_folders.split(",")]
            invalid = [f for f in real_folders if f not in REAL_FOLDERS]
            if invalid:
                print(f"WARNING: Unknown folders: {', '.join(invalid)}")
                print(f"  Available: {', '.join(REAL_FOLDERS)}")
            real_folders = [f for f in real_folders if f in REAL_FOLDERS]

        print(f"\nListing real-world data files ({len(real_folders or REAL_FOLDERS)} folders)...")
        real_files = list_real_files(real_folders)

        if args.limit:
            # Limit to N files per folder
            limited = []
            folder_counts = {}
            for f in real_files:
                folder = f.split("/")[0]
                folder_counts.setdefault(folder, 0)
                if folder_counts[folder] < args.limit:
                    limited.append(f)
                    folder_counts[folder] += 1
            print(f"  Found {len(real_files)} files, limited to {len(limited)} ({args.limit} per folder)")
            real_files = limited
        else:
            print(f"  Found {len(real_files)} files")

        if real_files:
            print(f"\nDownloading {len(real_files)} real-world files ({args.workers} workers)...")
            print("=" * 50)
            success = download_files_parallel(real_files, args.odir, args.workers)
            total_success += success
            total_files += len(real_files)

    # --- Summary ---
    print(f"\n{'=' * 50}")
    print(f"Downloaded {total_success}/{total_files} files")

    # --- Extract synthetic archives ---
    if args.extract and synthetic_components:
        print(f"\nExtracting synthetic archives...")
        sv_dir = os.path.join(args.odir, SYNTHETIC_PREFIX)
        extract_to = os.path.join(args.odir, "extracted")

        hdmap_tar = os.path.join(sv_dir, "hdmap.tar.gz")
        if os.path.exists(hdmap_tar):
            extract_tar(hdmap_tar, extract_to)

        caption_tar = os.path.join(sv_dir, "caption.tar.gz")
        if os.path.exists(caption_tar):
            extract_tar(caption_tar, extract_to)

        gen_parts = sorted([
            os.path.join(sv_dir, f)
            for f in os.listdir(sv_dir)
            if f.startswith("generation.tar.gz.part-")
        ]) if os.path.isdir(sv_dir) else []
        if gen_parts:
            extract_split_tar(gen_parts, extract_to)

        print(f"Extracted to: {extract_to}")

    # --- Next steps ---
    print(f"\nDone.")
    if synthetic_components:
        print(f"Synthetic archives: {args.odir}/{SYNTHETIC_PREFIX}/")
        if not args.extract:
            print(f"\nTo extract manually:")
            print(f"  cd {args.odir}/{SYNTHETIC_PREFIX}")
            print(f"  tar xzf hdmap.tar.gz")
            print(f"  tar xzf caption.tar.gz")
            print(f"  cat generation.tar.gz.part-* | tar xzf -")
    if download_real:
        print(f"Real-world data: {args.odir}/")


if __name__ == "__main__":
    main()
