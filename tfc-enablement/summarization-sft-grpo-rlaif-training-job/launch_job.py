#!/usr/bin/env python
"""
Client-side launcher — submit the WHOLE summarization workflow (prepare -> SFT ->
GRPO/RLAIF -> evaluate) as ONE SageMaker training job, using ModelTrainer
(SageMaker Python SDK v3).

All phases run inside a single job on one multi-GPU instance; artifacts move
through S3 under --output-s3. Role/region are auto-detected; pass --role to override.

The GRPO reward and the evaluation both call an LLM judge. With --judge-backend
bedrock (default), the JOB ROLE must allow bedrock:InvokeModel on the judge models.
The eval judge is intentionally a different model from the train judge.

For the per-step pipeline (each phase its own right-sized job, cheaper), use the
notebooks in notebooks/ instead.

Usage:
    pip install -r requirements-launch.txt
    python launch_job.py                                   # everything auto-detected
    python launch_job.py --base-model Qwen/Qwen3.5-0.8B    # more GRPO headroom
    python launch_job.py --judge-backend mock              # offline smoke (no Bedrock)
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
    p.add_argument("--job-name", default="summ-sft-grpo-rlaif", help="base job name")
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
    p.add_argument("--tldr-dataset", default="CarperAI/openai_summarize_tldr")
    p.add_argument("--n-train", type=int, default=8000)
    p.add_argument("--n-grpo", type=int, default=2000)
    p.add_argument("--n-eval", type=int, default=400)
    p.add_argument("--n-pairwise", type=int, default=200)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--enable-thinking", type=int, default=0, choices=[0, 1])
    p.add_argument("--attn-impl", default="flash_attention_2",
                   choices=["flash_attention_2", "eager", "sdpa"])
    p.add_argument("--use-lora", type=int, default=1, choices=[0, 1])
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--sft-epochs", type=float, default=1.0)
    p.add_argument("--sft-lr", type=float, default=2e-4)
    p.add_argument("--sft-bs", type=int, default=4)
    p.add_argument("--sft-grad-accum", type=int, default=4)
    p.add_argument("--grpo-epochs", type=float, default=1.0)
    p.add_argument("--grpo-lr", type=float, default=5e-6)
    p.add_argument("--grpo-bs", type=int, default=8)
    p.add_argument("--grpo-grad-accum", type=int, default=4)
    p.add_argument("--num-generations", type=int, default=8)
    p.add_argument("--grpo-beta", type=float, default=0.01)
    p.add_argument("--grpo-temperature", type=float, default=1.0)
    p.add_argument("--max-completion-length", type=int, default=128)
    p.add_argument("--judge-backend", default="bedrock", choices=["bedrock", "local", "mock"])
    p.add_argument("--train-judge-model", default="us.anthropic.claude-sonnet-4-5-20250929-v1:0")
    p.add_argument("--eval-judge-model", default="openai.gpt-oss-120b-1:0")
    p.add_argument("--judge-region", default=None)
    p.add_argument("--judge-max-concurrency", type=int, default=8)
    p.add_argument("--reward-weights", default="0.30,0.40,0.15,0.15")
    p.add_argument("--reward-mode", default="guarded", choices=["rubric", "guarded"])
    p.add_argument("--guard-weights", default="0.45,0.30,0.25")
    p.add_argument("--faith-gate", type=float, default=0.40)
    p.add_argument("--gate-mult", type=float, default=0.30)
    p.add_argument("--compare-against", default="sft", choices=["sft", "reference"])
    p.add_argument("--eval-base", type=int, default=1, choices=[0, 1])
    p.add_argument("--eval-max-new-tokens", type=int, default=128)
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

    output_s3 = args.output_s3
    if not output_s3:
        import time

        output_s3 = f"s3://{sess.default_bucket()}/summ-rlaif/{time.strftime('%Y%m%d-%H%M%S')}"
        print(f"[launch] --output-s3 not given; using {output_s3}")

    if args.eval_judge_model == args.train_judge_model and args.judge_backend != "mock":
        print("[launch] WARNING: eval judge == train judge. GRPO can game its own judge; "
              "set a different --eval-judge-model for an honest win-rate.")

    training_image = args.image_uri or image_uris.retrieve(
        framework="pytorch", region=region, version=args.pytorch_version,
        instance_type=args.instance_type, image_scope="training",
    )

    hyperparameters = {
        "output-s3": output_s3,
        "skip-eval": args.skip_eval,
        "base-model": args.base_model,
        "tldr-dataset": args.tldr_dataset,
        "n-train": args.n_train,
        "n-grpo": args.n_grpo,
        "n-eval": args.n_eval,
        "n-pairwise": args.n_pairwise,
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
        "judge-backend": args.judge_backend,
        "train-judge-model": args.train_judge_model,
        "eval-judge-model": args.eval_judge_model,
        "judge-max-concurrency": args.judge_max_concurrency,
        "reward-weights": args.reward_weights,
        "reward-mode": args.reward_mode,
        "guard-weights": args.guard_weights,
        "faith-gate": args.faith_gate,
        "gate-mult": args.gate_mult,
        "compare-against": args.compare_against,
        "eval-base": args.eval_base,
        "eval-max-new-tokens": args.eval_max_new_tokens,
        "eval-batch-size": args.eval_batch_size,
    }
    if args.judge_region:
        hyperparameters["judge-region"] = args.judge_region

    environment = {
        "HF_HOME": "/opt/ml/hf_home",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
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
    print(f"[launch] judge: train={args.train_judge_model}  eval={args.eval_judge_model}")

    trainer.train(wait=args.wait)
    print("[launch] submitted.")
    if not args.wait:
        print("[launch] tip: re-run with --wait to stream logs, or watch the SageMaker console.")


if __name__ == "__main__":
    main()
