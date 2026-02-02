#!/usr/bin/env python3
"""
SageMaker training script for MT-GRPO
Wraps the existing verifiers training code for SageMaker compatibility
"""
import os
import sys
import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser()
    
    # SageMaker environment variables
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    parser.add_argument('--output_data_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output/data'))
    parser.add_argument('--num_gpus', type=int, default=int(os.environ.get('SM_NUM_GPUS', 1)))
    parser.add_argument('--hosts', type=str, default=os.environ.get('SM_HOSTS', '[]'))
    parser.add_argument('--current_host', type=str, default=os.environ.get('SM_CURRENT_HOST', 'algo-1'))
    
    # Training hyperparameters
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-7B')
    parser.add_argument('--learning_rate', type=float, default=1e-6)
    parser.add_argument('--num_generations', type=int, default=8)
    parser.add_argument('--per_device_train_batch_size', type=int, default=2)
    parser.add_argument('--grad_accum_steps', type=int, default=4)
    parser.add_argument('--num_iterations', type=int, default=2)
    parser.add_argument('--max_steps', type=int, default=300)
    parser.add_argument('--beta', type=float, default=0)
    parser.add_argument('--trainer', type=str, default='mt_grpo')
    parser.add_argument('--turn_advantage_coef', type=float, default=1)
    
    args = parser.parse_args()
    
    # Set HuggingFace cache to use /tmp which has more space
    os.environ['HF_DATASETS_CACHE'] = '/tmp/hf_cache'
    os.environ['TRANSFORMERS_CACHE'] = '/tmp/transformers_cache'
    
    # Install dependencies
    print("Installing dependencies...")
    subprocess.run([
        sys.executable, '-m', 'pip', 'install', '-r', '/opt/ml/code/requirements.txt'
    ], check=True)
    
    print("Installing verifiers package...")
    subprocess.run([
        sys.executable, '-m', 'pip', 'install', '-e', '/opt/ml/code', '--no-deps'
    ], check=True)
    
    # Download Wikipedia index
    print("Downloading Wikipedia search index...")
    subprocess.run([
        sys.executable, '/opt/ml/code/verifiers/tools/local_wiki_search.py'
    ], check=True)
    
    # Build accelerate launch command
    # Use num_gpus-1 processes for training, leaving last GPU dedicated for vLLM
    num_training_processes = args.num_gpus - 1
    cmd = [
        'accelerate', 'launch',
        '--config-file', '/opt/ml/code/configs/zero3.yaml',
        '--num-processes', str(num_training_processes),
        '/opt/ml/code/verifiers/examples/triviaqa_search.py',
        '--model_name', args.model_name,
        '--num_gpus', str(args.num_gpus),
        '--learning_rate', str(args.learning_rate),
        '--num_generations', str(args.num_generations),
        '--per_device_train_batch_size', str(args.per_device_train_batch_size),
        '--grad_accum_steps', str(args.grad_accum_steps),
        '--num_iterations', str(args.num_iterations),
        '--max_steps', str(args.max_steps),
        '--beta', str(args.beta),
        '--trainer', args.trainer,
        '--turn_advantage_coef', str(args.turn_advantage_coef),
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    
    # Copy outputs to model directory
    print(f"Copying outputs to {args.model_dir}")
    for output_dir in ['outputs', 'checkpoints', 'logs']:
        if os.path.exists(output_dir):
            subprocess.run(['cp', '-r', output_dir, args.model_dir], check=True)

if __name__ == '__main__':
    main()
