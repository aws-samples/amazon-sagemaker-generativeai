#!/usr/bin/env python3
"""
Local P5 Multi-Turn GRPO Training - Step by Step
This script runs all the setup and training steps interactively.
"""

import os
import sys
import subprocess
import time
import yaml
import glob
from pathlib import Path

# Color codes for pretty output
class Colors:
    BLUE = '\033[0;34m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    RED = '\033[0;31m'
    NC = '\033[0m'  # No Color

def print_step(step_num, title):
    """Print a step header"""
    print(f"\n{Colors.BLUE}{'='*70}{Colors.NC}")
    print(f"{Colors.BLUE}STEP {step_num}: {title}{Colors.NC}")
    print(f"{Colors.BLUE}{'='*70}{Colors.NC}\n")

def print_success(message):
    """Print success message"""
    print(f"{Colors.GREEN}✓ {message}{Colors.NC}")

def print_warning(message):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠ {message}{Colors.NC}")

def print_error(message):
    """Print error message"""
    print(f"{Colors.RED}✗ {message}{Colors.NC}")

def run_command(cmd, check=True, capture_output=False):
    """Run a shell command"""
    try:
        if capture_output:
            result = subprocess.run(cmd, shell=True, check=check, 
                                  capture_output=True, text=True)
            return result.stdout.strip()
        else:
            result = subprocess.run(cmd, shell=True, check=check)
            return result.returncode == 0
    except subprocess.CalledProcessError as e:
        if check:
            print_error(f"Command failed: {cmd}")
            print_error(f"Error: {e}")
            sys.exit(1)
        return False

def step1_install_dependencies():
    """Step 1: Install Dependencies"""
    print_step(1, "Install Dependencies")
    
    requirements_file = "requirements_local.txt"
    
    if not os.path.exists(requirements_file):
        print_error(f"Requirements file not found: {requirements_file}")
        sys.exit(1)
    
    print(f"Installing packages from {requirements_file}...")
    run_command(f"pip install -q -r {requirements_file}")
    print_success("Dependencies installed")

def step2_verify_java():
    """Step 2: Verify Java Installation"""
    print_step(2, "Verify Java Installation")
    
    # Set Java environment
    java_home = "/usr/lib/jvm/java-21-openjdk-amd64"
    os.environ['JAVA_HOME'] = java_home
    os.environ['PATH'] = f"{java_home}/bin:{os.environ['PATH']}"
    
    print(f"JAVA_HOME set to: {java_home}")
    
    # Check if Java is available
    java_version = run_command("java -version 2>&1 | head -1", capture_output=True)
    
    if java_version:
        print_success(f"Java found: {java_version}")
    else:
        print_error("Java not found!")
        print_warning("Please install Java 21:")
        print("  apt-get update && apt-get install -y openjdk-21-jdk")
        sys.exit(1)

def step3_check_gpus():
    """Step 3: Check GPU Availability"""
    print_step(3, "Check GPU Availability")
    
    gpu_list = run_command("nvidia-smi --list-gpus", capture_output=True)
    
    if gpu_list:
        print("Available GPUs:")
        print(gpu_list)
        
        num_gpus = len(gpu_list.strip().split('\n'))
        print_success(f"Found {num_gpus} GPUs")
        
        if num_gpus < 2:
            print_error("Need at least 2 GPUs (1 for vLLM, 1+ for training)")
            sys.exit(1)
        
        return num_gpus
    else:
        print_error("No GPUs found!")
        sys.exit(1)

def step4_download_pyserini_index():
    """Step 4: Pre-download Pyserini Index"""
    print_step(4, "Pre-download Pyserini Index (Optional)")
    
    response = input("Download Pyserini Wikipedia index (10GB)? This may take several minutes. [y/N]: ")
    
    if response.lower() == 'y':
        print("Downloading Pyserini Wikipedia index...")
        try:
            from pyserini.search.lucene import LuceneSearcher
            searcher = LuceneSearcher.from_prebuilt_index('wikipedia-kilt-doc')
            print_success("Pyserini index ready")
        except Exception as e:
            print_warning(f"Failed to download Pyserini index: {e}")
            print_warning("Index will be downloaded during training if needed")
    else:
        print("Skipping Pyserini index download")

def step5_configure_training():
    """Step 5: Configure Training"""
    print_step(5, "Configure Training")
    
    # List available configs
    config_dir = "hf_recipes/Qwen"
    if os.path.exists(config_dir):
        configs = sorted(glob.glob(f"{config_dir}/*.yaml"))
        
        if configs:
            print("Available configurations:")
            for i, config in enumerate(configs, 1):
                print(f"  {i}. {os.path.basename(config)}")
            
            # Let user choose
            while True:
                choice = input(f"\nSelect configuration [1-{len(configs)}] or press Enter for default (1): ").strip()
                
                if choice == "":
                    choice = "1"
                
                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(configs):
                        config_file = configs[idx]
                        break
                    else:
                        print_error(f"Please enter a number between 1 and {len(configs)}")
                except ValueError:
                    print_error("Please enter a valid number")
        else:
            print_error(f"No config files found in {config_dir}")
            sys.exit(1)
    else:
        print_error(f"Config directory not found: {config_dir}")
        sys.exit(1)
    
    print(f"\nSelected configuration: {config_file}")
    
    # Get number of GPUs
    num_gpus = input("Number of GPUs [8]: ").strip()
    num_gpus = int(num_gpus) if num_gpus else 8
    
    # Get vLLM port
    vllm_port = input("vLLM server port [8000]: ").strip()
    vllm_port = int(vllm_port) if vllm_port else 8000
    
    return config_file, num_gpus, vllm_port

def step6_view_config(config_file):
    """Step 6: View Training Configuration"""
    print_step(6, "View Training Configuration")
    
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        print("Training Configuration:")
        print(f"  Model: {config.get('model_name_or_path', 'N/A')}")
        print(f"  Max steps: {config.get('max_steps', 'N/A')}")
        print(f"  Learning rate: {config.get('learning_rate', 'N/A')}")
        print(f"  Batch size: {config.get('per_device_train_batch_size', 'N/A')}")
        print(f"  Num generations: {config.get('num_generations', 'N/A')}")
        print(f"  Max env steps: {config.get('max_env_steps', 'N/A')}")
        print(f"  Turn advantage coef: {config.get('turn_advantage_coef', 'N/A')}")
        
        return config
    except Exception as e:
        print_error(f"Failed to read config: {e}")
        sys.exit(1)

def step7_launch_training(config_file, num_gpus, vllm_port):
    """Step 7: Launch Training"""
    print_step(7, "Launch Training")
    
    print(f"Configuration:")
    print(f"  Config file: {config_file}")
    print(f"  Total GPUs: {num_gpus}")
    print(f"  Training GPUs: 0-{num_gpus-2}")
    print(f"  vLLM GPU: {num_gpus-1}")
    print(f"  vLLM Port: {vllm_port}")
    
    print("\nTraining options:")
    print("  1. Run in foreground (blocking, see output in real-time)")
    print("  2. Run in background (non-blocking, logs to training.log)")
    print("  3. Cancel")
    
    choice = input("\nSelect option [1]: ").strip()
    choice = choice if choice else "1"
    
    if choice == "1":
        # Run in foreground
        print("\nStarting training in foreground...")
        print("Press Ctrl+C to stop training\n")
        time.sleep(2)
        
        cmd = f"bash local_mt_grpo_train.sh --config {config_file} --num_process {num_gpus} --vllm_port {vllm_port}"
        run_command(cmd, check=False)
        
    elif choice == "2":
        # Run in background
        print("\nStarting training in background...")
        
        cmd = f"nohup bash local_mt_grpo_train.sh --config {config_file} --num_process {num_gpus} --vllm_port {vllm_port} > training.log 2>&1 &"
        run_command(cmd)
        
        print_success("Training started in background")
        print(f"\nMonitor progress with:")
        print(f"  tail -f training.log")
        print(f"\nCheck vLLM server:")
        print(f"  curl http://localhost:{vllm_port}/health")
        print(f"\nView vLLM logs:")
        print(f"  tail -f vllm_server.log")
        
    else:
        print("Training cancelled")
        sys.exit(0)

def step8_monitor_training(vllm_port):
    """Step 8: Monitor Training"""
    print_step(8, "Monitor Training (Optional)")
    
    print("Monitoring options:")
    print("  1. Check vLLM server status")
    print("  2. View vLLM server logs")
    print("  3. Check GPU usage")
    print("  4. View training logs (if running in background)")
    print("  5. Skip monitoring")
    
    choice = input("\nSelect option [5]: ").strip()
    choice = choice if choice else "5"
    
    if choice == "1":
        print("\nChecking vLLM server status...")
        try:
            import requests
            response = requests.get(f"http://localhost:{vllm_port}/health", timeout=5)
            print_success(f"vLLM server is running (status: {response.status_code})")
        except Exception as e:
            print_error(f"vLLM server not reachable: {e}")
    
    elif choice == "2":
        print("\nvLLM server logs (last 50 lines):")
        run_command("tail -50 vllm_server.log", check=False)
    
    elif choice == "3":
        print("\nGPU usage:")
        run_command("nvidia-smi")
    
    elif choice == "4":
        print("\nTraining logs (last 100 lines):")
        run_command("tail -100 training.log", check=False)
    
    else:
        print("Skipping monitoring")

def step9_check_outputs():
    """Step 9: Check Training Output"""
    print_step(9, "Check Training Output")
    
    output_dirs = glob.glob("outputs/*")
    
    if output_dirs:
        print("Training outputs:")
        for d in sorted(output_dirs):
            print(f"  {d}")
    else:
        print("No output directories found yet")
        print("Outputs will appear in the 'outputs/' directory as training progresses")

def main():
    """Main execution"""
    print(f"\n{Colors.GREEN}{'='*70}{Colors.NC}")
    print(f"{Colors.GREEN}Local P5 Multi-Turn GRPO Training - Step by Step{Colors.NC}")
    print(f"{Colors.GREEN}{'='*70}{Colors.NC}\n")
    
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    print(f"Working directory: {os.getcwd()}\n")
    
    try:
        # Step 1: Install dependencies
        step1_install_dependencies()
        
        # Step 2: Verify Java
        step2_verify_java()
        
        # Step 3: Check GPUs
        num_gpus = step3_check_gpus()
        
        # Step 4: Download Pyserini index (optional)
        step4_download_pyserini_index()
        
        # Step 5: Configure training
        config_file, num_gpus, vllm_port = step5_configure_training()
        
        # Step 6: View config
        config = step6_view_config(config_file)
        
        # Step 7: Launch training
        step7_launch_training(config_file, num_gpus, vllm_port)
        
        # Step 8: Monitor training (optional)
        step8_monitor_training(vllm_port)
        
        # Step 9: Check outputs
        step9_check_outputs()
        
        print(f"\n{Colors.GREEN}{'='*70}{Colors.NC}")
        print(f"{Colors.GREEN}Setup and training launch complete!{Colors.NC}")
        print(f"{Colors.GREEN}{'='*70}{Colors.NC}\n")
        
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Interrupted by user{Colors.NC}")
        sys.exit(0)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
