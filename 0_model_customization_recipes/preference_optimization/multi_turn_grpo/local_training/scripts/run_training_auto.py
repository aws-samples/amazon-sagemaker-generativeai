#!/usr/bin/env python3
"""
Local P5 Multi-Turn GRPO Training - Automatic Mode
This script runs all setup and training steps automatically with default settings.
"""

import os
import sys
import subprocess
import yaml
from pathlib import Path

# Default configuration
DEFAULT_CONFIG = "hf_recipes/Qwen/Qwen3-0.6B--mt-grpo.yaml"
DEFAULT_NUM_GPUS = 8
DEFAULT_VLLM_PORT = 8000
RUN_IN_BACKGROUND = True  # Set to False to run in foreground

def log(message, level="INFO"):
    """Simple logging"""
    colors = {
        "INFO": "\033[0;34m",
        "SUCCESS": "\033[0;32m",
        "WARNING": "\033[1;33m",
        "ERROR": "\033[0;31m",
        "NC": "\033[0m"
    }
    color = colors.get(level, colors["INFO"])
    nc = colors["NC"]
    prefix = "✓" if level == "SUCCESS" else "✗" if level == "ERROR" else "•"
    print(f"{color}{prefix} {message}{nc}")

def run_cmd(cmd, check=True):
    """Run command and return success status"""
    try:
        subprocess.run(cmd, shell=True, check=check)
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    """Main execution"""
    print("\n" + "="*70)
    print("Local P5 Multi-Turn GRPO Training - Automatic Mode")
    print("="*70 + "\n")
    
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    log(f"Working directory: {os.getcwd()}")
    
    # Step 1: Install dependencies
    log("Installing dependencies...")
    if not run_cmd("pip install -q -r requirements_local.txt"):
        log("Failed to install dependencies", "ERROR")
        sys.exit(1)
    log("Dependencies installed", "SUCCESS")
    
    # Step 2: Set Java environment
    log("Setting up Java environment...")
    java_home = "/usr/lib/jvm/java-21-openjdk-amd64"
    os.environ['JAVA_HOME'] = java_home
    os.environ['PATH'] = f"{java_home}/bin:{os.environ['PATH']}"
    
    if run_cmd("java -version 2>&1 | head -1", check=False):
        log("Java configured", "SUCCESS")
    else:
        log("Java not found - install with: apt-get install -y openjdk-21-jdk", "ERROR")
        sys.exit(1)
    
    # Step 3: Check GPUs
    log("Checking GPU availability...")
    result = subprocess.run("nvidia-smi --list-gpus | wc -l", 
                          shell=True, capture_output=True, text=True)
    num_gpus = int(result.stdout.strip()) if result.returncode == 0 else 0
    
    if num_gpus < 2:
        log(f"Need at least 2 GPUs, found {num_gpus}", "ERROR")
        sys.exit(1)
    log(f"Found {num_gpus} GPUs", "SUCCESS")
    
    # Step 4: Check config file
    config_file = DEFAULT_CONFIG
    if not os.path.exists(config_file):
        log(f"Config file not found: {config_file}", "ERROR")
        sys.exit(1)
    
    # Step 5: Display configuration
    log(f"Configuration: {config_file}")
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        log(f"  Model: {config.get('model_name_or_path', 'N/A')}")
        log(f"  Max steps: {config.get('max_steps', 'N/A')}")
        log(f"  Learning rate: {config.get('learning_rate', 'N/A')}")
    except Exception as e:
        log(f"Could not read config: {e}", "WARNING")
    
    # Step 6: Launch training
    log(f"Launching training with {DEFAULT_NUM_GPUS} GPUs...")
    log(f"  Training GPUs: 0-{DEFAULT_NUM_GPUS-2}")
    log(f"  vLLM GPU: {DEFAULT_NUM_GPUS-1}")
    log(f"  vLLM Port: {DEFAULT_VLLM_PORT}")
    
    cmd = (f"bash local_mt_grpo_train.sh "
           f"--config {config_file} "
           f"--num_process {DEFAULT_NUM_GPUS} "
           f"--vllm_port {DEFAULT_VLLM_PORT}")
    
    if RUN_IN_BACKGROUND:
        cmd = f"nohup {cmd} > training.log 2>&1 &"
        log("Starting training in background...")
        run_cmd(cmd)
        log("Training started", "SUCCESS")
        print("\nMonitor progress:")
        print("  tail -f training.log")
        print(f"  curl http://localhost:{DEFAULT_VLLM_PORT}/health")
        print("  tail -f vllm_server.log")
    else:
        log("Starting training in foreground...")
        log("Press Ctrl+C to stop", "WARNING")
        run_cmd(cmd, check=False)
    
    print("\n" + "="*70)
    print("Training launched successfully!")
    print("="*70 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
