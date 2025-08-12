import os
import gc
import time
import argparse
import pandas as pd
import logging

import torch
import GPUtil
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration
)
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure GPU memory usage around loading quantized HuggingFace models"
    )
    parser.add_argument(
        "--quantized-model-path",
        type=str,
        required=True,
        help="Directory containing subdirectories of quantized models."
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=os.environ.get("SM_OUTPUT_DIR", "/opt/ml/output/gpu-metrics"),
        help="Path to save the GPU utilization report path."
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="cuda:0",
        help="HuggingFace transformers model load device"
    )
    return parser.parse_args()


def sample_gpu_memory(duration_sec: int, interval: float = 1.0) -> float:
    """
    Sample total GPU memory usage over all GPUs and return average used in MiB.
    """
    readings = []
    samples = int(duration_sec / interval)
    for _ in range(samples):
        gpus = GPUtil.getGPUs()
        total_used = sum(gpu.memoryUsed for gpu in gpus)
        readings.append(total_used)
        time.sleep(interval)
    return sum(readings) / len(readings) if readings else 0.0


def measure_and_load(model_path: str, output_path: str, device_map: str):
    """
    Measure GPU memory before and after loading the model, then load and cleanup.
    """
    # Get total memory across all GPUs (assumes identical capacity, can adjust if needed)
    gpus = GPUtil.getGPUs()
    total_mib = sum(gpu.memoryTotal for gpu in gpus)

    logger.info(f"Sampling GPU memory for 10s before loading {model_path!r}...")
    pre_avg = sample_gpu_memory(duration_sec=10, interval=1.0)

    # Load model and time it
    logger.info(f"Loading model from: {model_path}")
    start = time.time()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map
    )
    load_time = time.time() - start

    logger.info("Sampling GPU memory for 30s after load...")
    post_avg = sample_gpu_memory(duration_sec=30, interval=1.0)

    # Cleanup
    logger.info("Cleaning up model from GPU...")
    # model.to("cpu")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("Sleeping to ensure cache is cleared...")
    time.sleep(10)

    delta = post_avg - pre_avg

    # Prepare report row
    row = {
        "Model": os.path.basename(model_path),
        "GPU_Total_MiB": f"{total_mib:.2f}",
        "Used_Pre_MiB": f"{pre_avg:.2f}",
        "Used_Post_MiB": f"{post_avg:.2f}",
        "Delta_MiB": f"{delta:.2f}",
        "Load_Time_s": f"{load_time:.2f}"
    }

    # Write to CSV
    os.makedirs(output_path, exist_ok=True)
    output_csv = os.path.join(
        output_path,
        f"{os.path.basename(model_path)}-gpu-report.csv"
    )
    pd.DataFrame.from_dict([row]).to_csv(output_csv)

    # Print table
    print("\nGPU Utilization Report")
    print(f"{'Model':<30} {'Total(MiB)':>12} {'Pre(MiB)':>12} "
          f"{'Post(MiB)':>12} {'Δ(MiB)':>10} {'Load(s)':>10}")
    print("-" * 90)
    print(f"{row['Model']:<30} {row['GPU_Total_MiB']:>12} {row['Used_Pre_MiB']:>12} "
          f"{row['Used_Post_MiB']:>12} {row['Delta_MiB']:>10} {row['Load_Time_s']:>10}")


def main():
    args = parse_args()
    logger.info(f"Starting GPU measurement for models in {args.quantized_model_path!r}")
    measure_and_load(args.quantized_model_path, args.output_path, args.device_map)


if __name__ == "__main__":
    main()
