"""
Validation script for OpenVLA models.
Run this after training to evaluate your fine-tuned model.
"""
import argparse
import torch
import numpy as np
from pathlib import Path
from datasets import load_from_disk
from transformers import AutoProcessor, AutoModelForVision2Seq
from tqdm import tqdm
import json
import sys

script_dir = Path(__file__).parent.absolute()
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from utils.openvla_utils import ActionTokenizer, PurePromptBuilder, VicunaV15ChatPromptBuilder


def parse_args():
    parser = argparse.ArgumentParser(description="Validate OpenVLA model")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--output_file", type=str, default="validation_results.json")
    parser.add_argument("--save_predictions", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading model from {args.model_path}")
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
        attn_implementation="eager", trust_remote_code=True,
    ).to("cuda")
    model.eval()

    action_tokenizer = ActionTokenizer(processor.tokenizer)

    print(f"Loading validation dataset from {args.dataset_path}")
    dataset = load_from_disk(args.dataset_path)
    val_dataset = dataset["validation"] if "validation" in dataset else dataset["train"]

    num_samples = min(args.num_samples, len(val_dataset))
    indices = np.random.choice(len(val_dataset), num_samples, replace=False)

    print(f"Running validation on {num_samples} samples...")
    results = {"num_samples": num_samples, "predictions": [], "metrics": {}}
    l1_errors, l2_errors = [], []

    with torch.no_grad():
        for idx in tqdm(indices):
            sample = val_dataset[int(idx)]
            img_seq = sample["observation/image"]
            action_seq = sample["actions"]
            instruction = sample["language_instruction"]

            image = img_seq[0]
            action_gt = np.array(action_seq[0])

            prompt_builder_fn = PurePromptBuilder if "v01" not in args.model_path else VicunaV15ChatPromptBuilder
            prompt_text = prompt_builder_fn("openvla").build_prompt(instruction, "")

            inputs = processor(text=prompt_text, images=image, return_tensors="pt").to("cuda", dtype=torch.bfloat16)
            outputs = model(**inputs)

            action_logits = outputs.logits[0]
            action_token_ids = torch.argmax(action_logits[-7:], dim=-1).cpu().numpy()
            action_pred = action_tokenizer.decode_token_ids_to_actions(action_token_ids)

            l1_error = np.mean(np.abs(action_pred - action_gt))
            l2_error = np.sqrt(np.mean((action_pred - action_gt) ** 2))
            l1_errors.append(l1_error)
            l2_errors.append(l2_error)

            if args.save_predictions:
                results["predictions"].append({
                    "instruction": instruction,
                    "action_pred": action_pred.tolist(),
                    "action_gt": action_gt.tolist(),
                    "l1_error": float(l1_error),
                    "l2_error": float(l2_error),
                })

    results["metrics"] = {
        "mean_l1_error": float(np.mean(l1_errors)),
        "std_l1_error": float(np.std(l1_errors)),
        "median_l1_error": float(np.median(l1_errors)),
        "mean_l2_error": float(np.mean(l2_errors)),
        "std_l2_error": float(np.std(l2_errors)),
        "median_l2_error": float(np.median(l2_errors)),
    }

    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nL1 Error — Mean: {results['metrics']['mean_l1_error']:.4f}, "
          f"Median: {results['metrics']['median_l1_error']:.4f}")
    print(f"L2 Error — Mean: {results['metrics']['mean_l2_error']:.4f}, "
          f"Median: {results['metrics']['median_l2_error']:.4f}")
    print(f"Results saved to: {args.output_file}")


if __name__ == "__main__":
    main()
