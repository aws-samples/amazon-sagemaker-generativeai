#!/usr/bin/env python
"""
Client-side launcher — submit the WHOLE tool-calling workflow (prepare -> SFT ->
GRPO/RLVR -> evaluate) as ONE SageMaker training job, using ModelTrainer
(SageMaker Python SDK v3).

All phases run inside a single job on one multi-GPU instance; artifacts move
through S3 under --output-s3. Role/region are auto-detected; pass --role to override.

For the per-step pipeline (each phase its own right-sized job, cheaper), use the
notebooks in notebooks/ instead.

Role, region, the S3 output prefix (SageMaker default bucket), and your HF token are
all auto-detected — no required flags beyond the defaults.

Usage:
    pip install -r requirements-launch.txt
    python launch_job.py                                   # everything auto-detected
    python launch_job.py --base-model Qwen/Qwen3.5-9B --instance-type ml.p4de.24xlarge --wait
    python launch_job.py --output-s3 s3://my-bucket/toolcall/run1   # override the prefix

Provide --image-uri for a prebuilt image (recommended, esp. with vLLM for GRPO
rollouts). Otherwise a PyTorch DLC is resolved and src/requirements.txt is installed
at start.
"""
import argparse


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # AWS / SageMaker — optional, auto-detected when omitted
    p.add_argument("--role", default=None, help="execution role ARN (default: auto-detect)")
    p.add_argument("--region", default=None, help="AWS region (default: session region)")
    p.add_argument("--instance-type", default="ml.p4de.24xlarge",
                   help="Multi-GPU instance (8xA100 80GB). p5.48xlarge also works.")
    p.add_argument("--instance-count", type=int, default=1)
    p.add_argument("--volume-size", type=int, default=512, help="EBS GB for checkpoints/data")
    p.add_argument("--max-run", type=int, default=2 * 24 * 3600, help="max runtime seconds")
    p.add_argument("--keep-alive", type=int, default=3600,
                   help="warm-pool seconds after the job ends (3600 = API max; needs quota; "
                        "0 disables)")
    p.add_argument("--job-name", default="toolcall-sft-grpo", help="base job name")
    p.add_argument("--image-uri", default=None, help="prebuilt training image; else a DLC is resolved")
    p.add_argument("--pytorch-version", default="2.7", help="DLC version if no --image-uri")
    p.add_argument("--wait", action="store_true", help="block and stream logs")

    # Output — optional; defaults to the SageMaker default bucket + a timestamp
    p.add_argument("--output-s3", default=None,
                   help="s3://bucket/prefix base (default: auto — SageMaker default bucket "
                        "+ timestamped run). Substeps write data/, sft_model/, grpo_model/, "
                        "results/ underneath")
    p.add_argument("--skip-eval", type=int, default=0, choices=[0, 1])

    # Hugging Face access — auto-detected from your local token by default.
    p.add_argument("--hf-token", default=None,
                   help="HF token (default: auto — read from HF_TOKEN env or your "
                        "`huggingface-cli login`). Passed to the job environment.")

    # Pipeline hyperparameters (forwarded to src/entrypoint.py)
    p.add_argument("--base-model", default="Qwen/Qwen3.5-4B")
    p.add_argument("--xlam-dataset", default="Salesforce/xlam-function-calling-60k")
    p.add_argument("--n-train", type=int, default=8000)
    p.add_argument("--n-grpo", type=int, default=3000)
    p.add_argument("--n-eval", type=int, default=500)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--enable-thinking", type=int, default=0, choices=[0, 1])
    p.add_argument("--attn-impl", default="flash_attention_2",
                   choices=["flash_attention_2", "eager", "sdpa"])
    p.add_argument("--use-lora", type=int, default=1, choices=[0, 1])
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--sft-epochs", type=float, default=1.0)
    p.add_argument("--sft-lr", type=float, default=2e-4)
    p.add_argument("--sft-bs", type=int, default=2)
    p.add_argument("--sft-grad-accum", type=int, default=8)
    p.add_argument("--grpo-epochs", type=float, default=1.0)
    p.add_argument("--grpo-lr", type=float, default=1e-6)
    p.add_argument("--grpo-bs", type=int, default=8)
    p.add_argument("--grpo-grad-accum", type=int, default=4)
    p.add_argument("--num-generations", type=int, default=8)
    p.add_argument("--grpo-beta", type=float, default=0.04)
    p.add_argument("--grpo-temperature", type=float, default=1.0)
    p.add_argument("--max-completion-length", type=int, default=256)
    p.add_argument("--reward-weights", default="0.1,0.3,0.2,0.4")
    p.add_argument("--use-vllm", type=int, default=0, choices=[0, 1])
    p.add_argument("--eval-base", type=int, default=1, choices=[0, 1])
    p.add_argument("--eval-max-new-tokens", type=int, default=256)
    p.add_argument("--eval-batch-size", type=int, default=16)
    return p.parse_args()


def main():
    args = parse_args()

    from sagemaker.core import image_uris
    from sagemaker.core.helper.session_helper import Session, get_execution_role
    from sagemaker.train.configs import Compute, SourceCode, StoppingCondition
    from sagemaker.train.model_trainer import ModelTrainer

    sess = Session()
    role = args.role or get_execution_role()
    region = args.region or sess.boto_region_name

    # Auto-derive the output prefix from the SageMaker default bucket if not given.
    output_s3 = args.output_s3
    if not output_s3:
        import time

        output_s3 = f"s3://{sess.default_bucket()}/toolcall/{time.strftime('%Y%m%d-%H%M%S')}"
        print(f"[launch] --output-s3 not given; using {output_s3}")

    training_image = args.image_uri or image_uris.retrieve(
        framework="pytorch", region=region, version=args.pytorch_version,
        instance_type=args.instance_type, image_scope="training",
    )

    hyperparameters = {
        "output-s3": output_s3,
        "skip-eval": args.skip_eval,
        "base-model": args.base_model,
        "xlam-dataset": args.xlam_dataset,
        "n-train": args.n_train,
        "n-grpo": args.n_grpo,
        "n-eval": args.n_eval,
        "seq-len": args.seq_len,
        "seed": args.seed,
        "enable-thinking": args.enable_thinking,
        "attn-impl": args.attn_impl,
        "use-lora": args.use_lora,
        "lora-r": args.lora_r,
        "lora-alpha": args.lora_alpha,
        "sft-epochs": args.sft_epochs,
        "sft-lr": args.sft_lr,
        "sft-bs": args.sft_bs,
        "sft-grad-accum": args.sft_grad_accum,
        "grpo-epochs": args.grpo_epochs,
        "grpo-lr": args.grpo_lr,
        "grpo-bs": args.grpo_bs,
        "grpo-grad-accum": args.grpo_grad_accum,
        "num-generations": args.num_generations,
        "grpo-beta": args.grpo_beta,
        "grpo-temperature": args.grpo_temperature,
        "max-completion-length": args.max_completion_length,
        "reward-weights": args.reward_weights,
        "use-vllm": args.use_vllm,
        "eval-base": args.eval_base,
        "eval-max-new-tokens": args.eval_max_new_tokens,
        "eval-batch-size": args.eval_batch_size,
    }

    environment = {
        "HF_HOME": "/opt/ml/hf_home",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    # HF token: explicit --hf-token, else local HF_TOKEN env / `huggingface-cli login`.
    tok = args.hf_token
    if not tok:
        import os

        tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if not tok:
            try:
                from huggingface_hub import get_token

                tok = get_token()
            except Exception:  # noqa: BLE001
                tok = None
    if tok:
        environment["HF_TOKEN"] = tok
        environment["HUGGING_FACE_HUB_TOKEN"] = tok
        print("[launch] HF token found and passed to the job environment.")
    else:
        print("[launch] no HF token found (set HF_TOKEN, run `huggingface-cli login`, or "
              "pass --hf-token). Public models will still work.")

    source_code = SourceCode(
        source_dir="src",
        requirements="requirements.txt",
        entry_script="entrypoint.py",
    )
    compute = Compute(
        instance_type=args.instance_type,
        instance_count=args.instance_count,
        volume_size_in_gb=args.volume_size,
        keep_alive_period_in_seconds=args.keep_alive,
    )

    trainer = ModelTrainer(
        sagemaker_session=sess,
        role=role,
        training_image=training_image,
        source_code=source_code,
        compute=compute,
        hyperparameters=hyperparameters,
        environment=environment,
        base_job_name=args.job_name,
        stopping_condition=StoppingCondition(max_runtime_in_seconds=args.max_run),
    )

    print(f"[launch] region={region}")
    print(f"[launch] role={role}")
    print(f"[launch] instance={args.instance_type} x{args.instance_count}")
    print(f"[launch] image={training_image}")
    print(f"[launch] base_model={args.base_model}  output={output_s3}")

    trainer.train(wait=args.wait)
    print("[launch] submitted.")
    if not args.wait:
        print("[launch] tip: re-run with --wait to stream logs, or watch the SageMaker console.")


if __name__ == "__main__":
    main()
