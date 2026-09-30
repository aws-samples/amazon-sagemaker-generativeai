"""SeqKD SFT for Qwen3.5-4B on ShoppingBench, via HF TRL — run by SageMaker ModelTrainer.

ModelTrainer passes the `hyperparameters` dict as `--key value` CLI args and mounts each
InputData channel at /opt/ml/input/data/<channel>. We read the `train` (and optional `val`)
channel, SFT with TRL on the pre-rendered {prompt, completion} rows (loss falls on the
completion only — TRL auto-detects prompt/completion format), and save to /opt/ml/model.

Because WE own this script, `--max-length` is honored exactly (no managed-recipe reduction):
pass 4096 to replicate the report, or up to 16384 to use the full context that the managed
serverless SFTTrainer silently caps at 4096. That control is the reason to use ModelTrainer.

Kernel note (Qwen3.5 is a linear-attn/conv1d hybrid): the linear-attn path needs
flash-linear-attention, whose triton>=3.7.1 requirement collides with torch 2.10's triton==3.6
pin under pip's resolver. `--install-kernels` (default on) force-installs the compatible pair
with --no-deps at job start; see the block below.
"""
import argparse, os, glob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--train", default="/opt/ml/input/data/train")   # channel dir
    ap.add_argument("--val",   default="/opt/ml/input/data/val")     # optional channel dir
    ap.add_argument("--output", default="/opt/ml/model")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=16384)         # honored exactly (no 4096 cap here)
    ap.add_argument("--max-steps", type=int, default=-1)             # >0 = stop early (path-verification runs)
    ap.add_argument("--max-train-samples", type=int, default=-1)     # >0 = subset train BEFORE tokenizing (fast verify)
    ap.add_argument("--max-val-samples", type=int, default=-1)       # >0 = subset val too
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--fft", type=str, default="false")   # "true"/"false" (ModelTrainer passes --key value)
    # Qwen3.5 is a linear-attn/conv1d HYBRID: FSDP full_shard crashes those custom params
    # (cudaErrorIllegalAddress) — use shard_grad_op (ZeRO-2, params unsharded) for FFT.
    ap.add_argument("--fsdp", default="")
    ap.add_argument("--install-kernels", type=str, default="true")   # force-install fla/triton kernels at job start
    ap.add_argument("--attn", type=str, default="sdpa")   # sdpa is safe everywhere. NOTE: flash_attention_2
    # CUDA-IMAs at step 0 on the managed SMTJ p4de/p5 driver (loads, then crashes in the forward with no
    # fallback), so keep sdpa on SageMaker; sdpa is numerically the same attention, just slower.
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    FFT = a.fft.lower() == "true"

    # Install the Qwen3.5-hybrid linear-attn kernels. This MUST use --no-deps and MUST run before
    # importing torch/transformers below. Why --no-deps: fla 0.5.2 needs triton>=3.7.1, but the verl
    # image's torch==2.10 pins triton==3.6 -> a normal `pip install` either dead-ends (ResolutionImpossible)
    # or bumps torch->2.14 and breaks transformers. --no-deps force-installs triton 3.8 + fla 0.5.2 OVER
    # torch's triton pin (triton 3.8 is runtime-compatible with torch 2.10), exactly like the pod's trlenv2.
    # The baked fla (old) CUDA-IMAs even on a managed A100/H100, so this upgrade is required, not optional.
    # Any failure is non-fatal: the linear-attn path falls back to a (slower) torch implementation.
    # Install ONCE on local rank 0 only: all ranks share this single node's site-packages, so letting
    # every rank install in parallel makes 8 pip processes race — harmless for the fla/triton wheels but
    # it collides the causal-conv1d source compile (8 concurrent nvcc builds). Rank 0 installs; the other
    # ranks wait for the done-marker, then import the now-present kernels.
    if a.install_kernels.lower() == "true":
        import subprocess, sys, time
        rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
        done = "/tmp/.seqkd_kernels_installed"
        if rank == 0:
            os.environ.setdefault("MAX_JOBS", "4")   # cap nvcc jobs (causal-conv1d) to avoid host-RAM OOM
            def _pip(*args):
                subprocess.check_call([sys.executable, "-m", "pip", "install", *args])
            try:
                _pip("--no-deps", "triton==3.8.0", "flash-linear-attention==0.5.2")
                print("[kernels] triton 3.8.0 + flash-linear-attention 0.5.2 (--no-deps)", flush=True)
            except Exception as e:
                print("[kernels] fla/triton FAILED ->", repr(e), "(continuing; torch linear-attn fallback)", flush=True)
            try:
                _pip("causal-conv1d>=1.4.0", "--no-build-isolation")   # Mamba conv fast path (speed only)
                print("[kernels] causal-conv1d installed", flush=True)
            except Exception as e:
                print("[kernels] causal-conv1d skipped ->", repr(e), flush=True)
            open(done, "w").close()
        else:
            for _ in range(180):   # wait up to ~15 min for rank 0's causal-conv1d compile
                if os.path.exists(done):
                    break
                time.sleep(5)

    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForImageTextToText, set_seed
    from trl import SFTTrainer, SFTConfig
    from peft import LoraConfig
    set_seed(a.seed)

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def _load(dir_):
        files = sorted(glob.glob(os.path.join(dir_, "*.jsonl")))
        return load_dataset("json", data_files=files, split="train") if files else None
    ds = _load(a.train)
    eval_ds = _load(a.val)
    # Subset BEFORE tokenizing so a verify run doesn't tokenize the whole 58k (max_steps only caps steps).
    if a.max_train_samples > 0 and ds is not None:
        ds = ds.select(range(min(a.max_train_samples, len(ds))))
    if a.max_val_samples > 0 and eval_ds is not None:
        eval_ds = eval_ds.select(range(min(a.max_val_samples, len(eval_ds))))
    print(f"train rows {len(ds)}  eval rows {0 if eval_ds is None else len(eval_ds)}  max_length {a.max_length}", flush=True)

    # Qwen3.5-4B is a Text+Image VLM class; we SFT text-only (vision path unused).
    # --attn picks the FULL-attn kernel: flash_attention_2 (H100, fast) or sdpa (A10G/g5, where
    # flash-attn CUDA-IMAs at step 0). fla handles the linear-attn path separately. FA2 LOADS on A10G
    # (no exception) but crashes in the forward, so honor --attn explicitly rather than try/except.
    def _load(impl):
        return AutoModelForImageTextToText.from_pretrained(
            a.model, torch_dtype=torch.bfloat16, trust_remote_code=True, attn_implementation=impl)
    try:
        model = _load(a.attn); print(f"[attn] {a.attn}", flush=True)
    except Exception as e:
        print(f"[attn] {a.attn} unavailable -> sdpa:", repr(e), flush=True)
        model = _load("sdpa")

    peft = None if FFT else LoraConfig(
        r=a.lora_r, lora_alpha=a.lora_r * 2, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM", target_modules="all-linear")

    cfg = SFTConfig(
        output_dir=a.output, num_train_epochs=a.epochs, learning_rate=a.lr,
        per_device_train_batch_size=a.batch_size, gradient_accumulation_steps=a.grad_accum,
        max_length=a.max_length, bf16=True, gradient_checkpointing=True,
        max_steps=a.max_steps,
        logging_steps=10, logging_first_step=True, packing=False,
        save_strategy="epoch", save_total_limit=1,
        eval_strategy=("epoch" if eval_ds is not None else "no"),
        fsdp=a.fsdp,
        gradient_checkpointing_kwargs=({"use_reentrant": False} if a.fsdp else None),
        seed=a.seed, report_to=[])

    tr = SFTTrainer(model=model, args=cfg, train_dataset=ds, eval_dataset=eval_ds,
                    processing_class=tok, peft_config=peft)
    if hasattr(tr.model, "print_trainable_parameters"):
        tr.model.print_trainable_parameters()
    tr.train()
    tr.save_model(a.output)                 # rank-0 safe; LoRA adapter or full (FFT) weights
    if tr.is_world_process_zero():
        tok.save_pretrained(a.output)
    print("saved ->", a.output, flush=True)
    # NOTE: LoRA runs save the ADAPTER. Merge for serving with a separate step:
    #   AutoModelForImageTextToText.from_pretrained(base) + PeftModel.from_pretrained(...).merge_and_unload()
    #   then copy preprocessor_config.json + processor_config.json into the merged dir before vLLM serve.


if __name__ == "__main__":
    main()
