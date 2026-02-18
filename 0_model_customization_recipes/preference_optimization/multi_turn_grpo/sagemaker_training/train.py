#!/usr/bin/env python3
"""SageMaker training entry point for MT-GRPO"""
import os
import sys
import subprocess
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--num_gpus", type=str, default="8")
    parser.add_argument("--vllm_port", type=str, default="8000")
    args, _ = parser.parse_known_args()
    
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_script = os.path.join(script_dir, "sm_mt_grpo_train.sh")
    
    # Build command
    cmd = [
        "bash",
        train_script,
        "--config", args.config,
        "--num_process", args.num_gpus,
        "--vllm_port", args.vllm_port,
    ]
    
    print(f"Launching training: {' '.join(cmd)}")
    sys.stdout.flush()
    
    # Execute training script
    result = subprocess.run(cmd, cwd=script_dir)
    sys.exit(result.returncode)
