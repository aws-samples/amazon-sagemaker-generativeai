#!/usr/bin/env python3
"""
Test script to verify local P5 training setup
Run this before starting training to catch issues early
"""

import os
import sys
import subprocess
import importlib

def test(name, func):
    """Run a test and report result"""
    try:
        result = func()
        if result:
            print(f"✓ {name}")
            return True
        else:
            print(f"✗ {name}")
            return False
    except Exception as e:
        print(f"✗ {name}: {e}")
        return False

def check_java():
    """Check Java installation"""
    try:
        result = subprocess.run("java -version", shell=True, 
                              capture_output=True, text=True)
        return result.returncode == 0
    except:
        return False

def check_gpus():
    """Check GPU availability"""
    try:
        result = subprocess.run("nvidia-smi --list-gpus", shell=True,
                              capture_output=True, text=True)
        if result.returncode == 0:
            num_gpus = len(result.stdout.strip().split('\n'))
            print(f"  Found {num_gpus} GPUs")
            return num_gpus >= 2
        return False
    except:
        return False

def check_package(package_name):
    """Check if a Python package is installed"""
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False

def check_files():
    """Check if required files exist"""
    required_files = [
        "local_mt_grpo_train.sh",
        "mt_grpo_trainer.py",
        "requirements_local.txt",
        "configs/accelerate/ds_zero3.yaml",
        "hf_recipes/Qwen/Qwen3-0.6B--mt-grpo.yaml",
        "rewards/triviaqa_reward.py",
        "tools_funcs/wiki_search.py"
    ]
    
    missing = []
    for f in required_files:
        if not os.path.exists(f):
            missing.append(f)
    
    if missing:
        print(f"  Missing files: {', '.join(missing)}")
        return False
    return True

def check_port(port):
    """Check if a port is available"""
    try:
        result = subprocess.run(f"lsof -i :{port}", shell=True,
                              capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            print(f"  Port {port} is in use")
            return False
        return True
    except:
        return True

def main():
    print("\n" + "="*60)
    print("Local P5 Training Setup - Verification")
    print("="*60 + "\n")
    
    results = []
    
    # Test 1: Java
    results.append(test("Java 21 installed", check_java))
    
    # Test 2: GPUs
    results.append(test("GPU availability (need 2+)", check_gpus))
    
    # Test 3: Required files
    results.append(test("Required files present", check_files))
    
    # Test 4: Python packages
    print("\nPython Packages:")
    packages = [
        "torch",
        "transformers",
        "accelerate",
        "deepspeed",
        "trl",
        "vllm",
        "peft",
        "datasets"
    ]
    
    for pkg in packages:
        results.append(test(f"  {pkg}", lambda p=pkg: check_package(p)))
    
    # Test 5: Port availability
    print("\nPort Availability:")
    results.append(test("  Port 8000 (vLLM)", lambda: check_port(8000)))
    
    # Test 6: Pyserini (optional)
    print("\nOptional Components:")
    try:
        from pyserini.search.lucene import LuceneSearcher
        print("✓ Pyserini installed")
    except ImportError:
        print("⚠ Pyserini not installed (will be installed during training)")
    
    # Summary
    print("\n" + "="*60)
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✓ All tests passed ({passed}/{total})")
        print("\nYou're ready to start training!")
        print("Run: python run_training_auto.py")
    else:
        print(f"✗ Some tests failed ({passed}/{total} passed)")
        print("\nPlease fix the issues above before training")
        
        if not check_java():
            print("\nTo install Java:")
            print("  apt-get update && apt-get install -y openjdk-21-jdk")
        
        if not all([check_package(p) for p in packages]):
            print("\nTo install Python packages:")
            print("  pip install -r requirements_local.txt")
    
    print("="*60 + "\n")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())
