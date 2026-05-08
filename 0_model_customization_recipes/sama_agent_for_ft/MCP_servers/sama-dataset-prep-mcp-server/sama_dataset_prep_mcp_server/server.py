#!/usr/bin/env python3
"""
SAMA Dataset Prep MCP Server
Bounded dataset preparation: 4 approved sample datasets + BYOD validation + S3 upload.
Each sample dataset has a dedicated function matching the exact notebook logic.
"""

import base64
import io as _io
import json
import logging
import os
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)

from fastmcp import FastMCP

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

mcp = FastMCP("sama-dataset-prep-mcp-server")

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
OUTPUT_DIR = os.path.join(os.getcwd(), "tmp_cache_local_dataset")


# ===================================================================
# Tool 1: BYOD Validation
# ===================================================================

@mcp.tool()
def validate_byod_dataset(
    file_path: str,
    num_samples_to_check: int = 5,
) -> Dict[str, Any]:
    """
    Validate a user-provided JSONL dataset for SFT/GRPO training format.

    Checks:
    - File is valid JSONL
    - Each row has 'messages' (SFT) or 'prompt'+'answer' (GRPO)
    - Messages have 'role' and 'content' fields
    - Image content has valid base64 or path in image_url blocks
    - Audio content has audio_url blocks
    - Reasoning content has optional 'thinking' field on assistant messages

    Args:
        file_path: Local path to the JSONL file to validate.
        num_samples_to_check: Number of rows to inspect (default 5).

    Returns:
        Validation result with format type, sample count, and any errors found.
    """
    try:
        if not os.path.isfile(file_path):
            return {"status": "error", "message": f"File not found: {file_path}"}

        errors = []
        format_type = None
        total_lines = 0
        has_thinking = False
        has_images = False
        has_audio = False

        with open(file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                total_lines += 1
                line = line.strip()
                if not line:
                    continue

                try:
                    row = json.loads(line)
                except json.JSONDecodeError as e:
                    if i < num_samples_to_check:
                        errors.append(f"Line {i+1}: Invalid JSON — {e}")
                    continue

                # Detect format
                if "messages" in row:
                    if format_type is None:
                        format_type = "sft"
                    if i < num_samples_to_check:
                        msgs = row["messages"]
                        if not isinstance(msgs, list) or len(msgs) < 2:
                            errors.append(f"Line {i+1}: 'messages' must be a list with at least 2 entries")
                            continue
                        for j, msg in enumerate(msgs):
                            if "role" not in msg:
                                errors.append(f"Line {i+1}, message {j}: missing 'role'")
                            if "content" not in msg:
                                errors.append(f"Line {i+1}, message {j}: missing 'content'")
                            if msg.get("thinking"):
                                has_thinking = True
                            content = msg.get("content", "")
                            if isinstance(content, list):
                                for block in content:
                                    if isinstance(block, dict):
                                        if block.get("type") == "image_url":
                                            has_images = True
                                        if block.get("type") == "audio":
                                            has_audio = True

                elif "prompt" in row and "answer" in row:
                    if format_type is None:
                        format_type = "grpo"
                    if i < num_samples_to_check:
                        prompt = row["prompt"]
                        if not isinstance(prompt, list):
                            errors.append(f"Line {i+1}: 'prompt' must be a list of message dicts")
                else:
                    if i < num_samples_to_check:
                        errors.append(f"Line {i+1}: Row must have 'messages' (SFT) or 'prompt'+'answer' (GRPO)")

        modality = "text"
        if has_thinking:
            modality = "text_reasoning"
        elif has_images:
            modality = "image"
        elif has_audio:
            modality = "audio"

        if errors:
            return {
                "status": "error",
                "message": f"Validation found {len(errors)} issue(s) in first {num_samples_to_check} rows.",
                "errors": errors,
                "total_rows": total_lines,
                "detected_format": format_type,
                "detected_modality": modality,
            }

        return {
            "status": "success",
            "message": f"Dataset validated — {total_lines} rows, {format_type} format, {modality} modality.",
            "total_rows": total_lines,
            "detected_format": format_type,
            "detected_modality": modality,
            "has_thinking": has_thinking,
            "has_images": has_images,
            "has_audio": has_audio,
            "file_path": file_path,
            "next_step": "Dataset is valid. Ask the user for an S3 URI and call upload_to_s3.",
        }
    except Exception as e:
        return {"status": "error", "message": f"Validation failed: {e}"}


# ===================================================================
# Tool 2: Prepare Finance-Instruct-500k (text)
# Exact logic from: finetune--meta-llama--Llama-3.2-3B-Instruct.ipynb
# ===================================================================

@mcp.tool()
def prepare_finance_instruct_500k(
    max_samples: Optional[int] = 1000,
    system_prompt: str = "You are a financial reasoning assistant. Read the user's query, restate the key data, and solve step by step. Show calculations clearly, explain any rounding or adjustments, and present the final answer in a concise and professional manner.",
) -> Dict[str, Any]:
    """
    Prepare Josephgflowers/Finance-Instruct-500k as messages-format JSONL.

    This is a TEXT modality dataset with system/user/assistant columns.
    Columns: system, user, assistant.

    Args:
        max_samples: Max samples to process (None = all, default 1000).
        system_prompt: System message content.

    Returns:
        Output file path and sample count.
    """
    try:
        from datasets import load_dataset

        dataset_name = "Josephgflowers/Finance-Instruct-500k"
        split = f"train[:{max_samples}]" if max_samples else "train"
        ds = load_dataset(dataset_name, split=split)
        logger.info(f"Loaded {len(ds)} samples from {dataset_name}")

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_file = os.path.join(OUTPUT_DIR, "Josephgflowers--Finance-Instruct-500k.jsonl")

        count = 0
        with open(out_file, "w", encoding="utf-8") as f:
            for row in ds:
                user_content = row.get("user", "")
                assistant_content = row.get("assistant", "")
                if not user_content or not assistant_content:
                    continue
                record = {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": assistant_content},
                    ]
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1

        return {
            "status": "success",
            "dataset": dataset_name,
            "modality": "text",
            "samples_processed": count,
            "output_path": out_file,
            "next_step": "Ask the user for an S3 URI and call upload_to_s3.",
        }
    except Exception as e:
        return {"status": "error", "message": f"Failed to prepare Finance-Instruct-500k: {e}"}


# ===================================================================
# Tool 3: Prepare hermes_reasoning_tool_use (text_reasoning)
# Exact logic from: finetune--openai--gpt-oss-20b.ipynb
# ===================================================================

@mcp.tool()
def prepare_hermes_reasoning_tool_use(
    max_samples: Optional[int] = 1000,
    gpt_oss: bool = False,
) -> Dict[str, Any]:
    """
    Prepare interstellarninja/hermes_reasoning_tool_use as messages-format JSONL.

    Two output formats depending on the target model:

    gpt_oss=True (OpenAI GPT-oss models):
      - 'thinking' is a SEPARATE FIELD on each message dict (None for system/user,
        extracted reasoning string for assistant)
      - assistant 'content' contains only the <tool_call> blocks
      - This matches the GPT-oss harmony response format

    gpt_oss=False (Qwen, DeepSeek, Llama, all other models — DEFAULT):
      - Thinking is INLINE in assistant content as <think>...</think>
      - No separate 'thinking' field
      - assistant content = <think>reasoning</think>\n<tool_call>...</tool_call>
      - This is the standard open-source reasoning format

    Args:
        max_samples: Max samples to process (None = all, default 1000).
        gpt_oss: If True, use GPT-oss format (thinking as separate field).
            If False (default), use standard format (thinking inline as <think> tags).

    Returns:
        Output file path, format used, and sample count.
    """
    try:
        from datasets import load_dataset

        dataset_name = "interstellarninja/hermes_reasoning_tool_use"
        split = f"train[:{max_samples}]" if max_samples else "train"
        ds = load_dataset(dataset_name, split=split)
        logger.info(f"Loaded {len(ds)} samples from {dataset_name}")

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        fmt_tag = "gpt-oss" if gpt_oss else "standard"
        out_file = os.path.join(
            OUTPUT_DIR,
            f"interstellarninja--hermes_reasoning_tool_use--{fmt_tag}.jsonl",
        )

        count = 0
        with open(out_file, "w", encoding="utf-8") as f:
            for row in ds:
                conversations = row.get("conversations", [])
                if len(conversations) < 3:
                    continue

                system_content = conversations[0]["value"]
                user_content = conversations[1]["value"]
                assistant_text = conversations[2]["value"]

                # Extract reasoning and tool calls from raw text
                think_match = THINK_RE.search(assistant_text)
                reasoning_content = think_match.group(1).strip() if think_match else ""
                tool_payloads = TOOL_CALL_RE.findall(assistant_text)
                tool_call_str = "\n".join(
                    f"<tool_call>{t}</tool_call>" for t in tool_payloads
                )

                if gpt_oss:
                    # GPT-oss format: thinking as separate field, content = tool calls only
                    record = {
                        "messages": [
                            {"role": "system", "content": system_content, "thinking": None},
                            {"role": "user", "content": user_content, "thinking": None},
                            {"role": "assistant", "content": tool_call_str, "thinking": reasoning_content},
                        ]
                    }
                else:
                    # Standard format: thinking inline as <think> tags in content
                    if reasoning_content:
                        assistant_content = f"<think>{reasoning_content}</think>\n{tool_call_str}"
                    else:
                        assistant_content = tool_call_str
                    record = {
                        "messages": [
                            {"role": "system", "content": system_content},
                            {"role": "user", "content": user_content},
                            {"role": "assistant", "content": assistant_content},
                        ]
                    }

                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1

        return {
            "status": "success",
            "dataset": dataset_name,
            "modality": "text_reasoning",
            "format": "gpt_oss" if gpt_oss else "standard_think_tags",
            "samples_processed": count,
            "output_path": out_file,
            "next_step": "Ask the user for an S3 URI and call upload_to_s3.",
        }
    except Exception as e:
        return {"status": "error", "message": f"Failed to prepare hermes_reasoning_tool_use: {e}"}


# ===================================================================
# Tool 4: Prepare Visual-TableQA (image)
# Exact logic from: finetune--Qwen--Qwen3-VL-2B-Instruct.ipynb
# ===================================================================

def _pil_to_base64(pil_img, resize_perc: float = 0.2) -> str:
    """Convert a PIL image to base64-encoded PNG string."""
    from PIL import Image
    if not isinstance(pil_img, Image.Image):
        raise ValueError("Input must be a PIL Image.")
    new_size = [int(resize_perc * s) for s in pil_img.size]
    pil_img = pil_img.resize(new_size)
    buffer = _io.BytesIO()
    pil_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


@mcp.tool()
def prepare_visual_tableqa(
    max_samples: Optional[int] = 200,
    system_prompt: str = "You are a multimodal reasoning assistant. Given a table (and its image if present) and a question, provide a clear, concise answer followed by a brief explanation of how the table supports your conclusion. Keep the reasoning grounded in the data and avoid speculation.",
) -> Dict[str, Any]:
    """
    Prepare AI-4-Everyone/Visual-TableQA as messages-format JSONL with base64 images.

    This is an IMAGE modality dataset. Converts PIL images to base64 PNG and
    builds image_url content blocks.
    Columns: question, answer, image (PIL).

    Args:
        max_samples: Max samples to process (None = all, default 200).
        system_prompt: System message content.

    Returns:
        Output file path and sample count.
    """
    try:
        from datasets import load_dataset
        from PIL import Image

        dataset_name = "AI-4-Everyone/Visual-TableQA"
        split = f"train[:{max_samples}]" if max_samples else "train"
        ds = load_dataset(dataset_name, split=split)
        logger.info(f"Loaded {len(ds)} samples from {dataset_name}")

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_file = os.path.join(OUTPUT_DIR, "AI-4-Everyone--Visual-TableQA.jsonl")

        count = 0
        with open(out_file, "w", encoding="utf-8") as f:
            for row in ds:
                user_content = row.get("question", "")
                assistant_content = row.get("answer", "")
                image_content = row.get("image")

                if not user_content:
                    continue

                images = []
                if image_content is not None:
                    if isinstance(image_content, list):
                        for img in image_content:
                            if hasattr(img, "save"):
                                b64 = _pil_to_base64(img)
                                images.append({
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                                })
                    elif hasattr(image_content, "save"):
                        b64 = _pil_to_base64(image_content)
                        images.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        })

                record = {
                    "messages": [
                        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                        {"role": "user", "content": images + [{"type": "text", "text": user_content}]},
                        {"role": "assistant", "content": [{"type": "text", "text": assistant_content}]},
                    ]
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1

        return {
            "status": "success",
            "dataset": dataset_name,
            "modality": "image",
            "samples_processed": count,
            "output_path": out_file,
            "next_step": "Ask the user for an S3 URI and call upload_to_s3.",
        }
    except Exception as e:
        return {"status": "error", "message": f"Failed to prepare Visual-TableQA: {e}"}


# ===================================================================
# Tool 5: Prepare AudioSet-Audio-Instructions (audio)
# Exact logic from: finetune--Qwen--Qwen2-Audio-7B-Instruct.ipynb
# ===================================================================

@mcp.tool()
def prepare_audioset_instructions(
    max_samples: Optional[int] = 200,
    system_prompt: str = "You are an audio understanding assistant. Listen carefully to the audio clip, analyze the sounds, and provide a clear and concise description.",
) -> Dict[str, Any]:
    """
    Prepare mesolitica/AudioSet-Audio-Instructions as messages-format JSONL with audio files.

    This is an AUDIO modality dataset. Decodes audio arrays to WAV files and
    builds audio_url content blocks with file:// paths.
    Columns: audio_filename (Audio), question, answer.

    IMPORTANT: This also saves WAV files under tmp_cache_local_dataset/audio_files/.
    Both the JSONL and the audio_files directory must be uploaded to S3.

    Args:
        max_samples: Max samples to process (None = all, default 200).
        system_prompt: System message content.

    Returns:
        Output file path, audio directory path, and sample count.
    """
    try:
        import soundfile as sf
        from datasets import load_dataset, Audio

        dataset_name = "mesolitica/AudioSet-Audio-Instructions"
        split = f"500k_part1_speech[:{max_samples}]" if max_samples else "500k_part1_speech"
        ds = load_dataset(dataset_name, split=split)
        ds = ds.cast_column("audio_filename", Audio(sampling_rate=16000))
        logger.info(f"Loaded {len(ds)} samples from {dataset_name}")

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        audio_dir = os.path.join(OUTPUT_DIR, "audio_files")
        os.makedirs(audio_dir, exist_ok=True)
        out_file = os.path.join(OUTPUT_DIR, "mesolitica--AudioSet-Audio-Instructions.jsonl")

        count = 0
        with open(out_file, "w", encoding="utf-8") as f:
            for idx, row in enumerate(ds):
                audio = row.get("audio_filename")
                if audio is None:
                    continue

                array = audio["array"]
                sr = audio["sampling_rate"]

                filename = f"sample-{idx:06d}.wav"
                filepath = os.path.join(audio_dir, filename)
                sf.write(filepath, array, sr)

                rel_path = os.path.relpath(filepath, OUTPUT_DIR)
                question = row.get("question", "")
                answer = row.get("answer", "")

                record = {
                    "messages": [
                        {
                            "role": "system",
                            "content": [{"type": "text", "text": system_prompt}],
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "audio", "audio_url": f"file:///opt/ml/input/data/training/{rel_path}"},
                                {"type": "text", "text": question},
                            ],
                        },
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": answer}],
                        },
                    ]
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1

        return {
            "status": "success",
            "dataset": dataset_name,
            "modality": "audio",
            "samples_processed": count,
            "output_path": out_file,
            "audio_dir": audio_dir,
            "note": "You must upload BOTH the JSONL file AND the audio_files directory to S3.",
            "next_step": "Ask the user for an S3 URI. Upload the JSONL with upload_to_s3, then also upload the audio_files directory.",
        }
    except Exception as e:
        return {"status": "error", "message": f"Failed to prepare AudioSet-Audio-Instructions: {e}"}


# ===================================================================
# Tool 6: S3 Upload
# ===================================================================

@mcp.tool()
def upload_to_s3(
    local_path: str,
    s3_uri: str,
) -> Dict[str, Any]:
    """
    Upload a prepared dataset file or directory to S3 for SageMaker training.

    IMPORTANT: You MUST ask the user for the s3_uri before calling this tool.
    Do NOT use a default.

    The returned training_data_s3_uri MUST be passed when calling
    launch_sft_training_job or launch_grpo_training_job.

    Args:
        local_path: Local path to the JSONL file or directory to upload.
        s3_uri: Full S3 URI destination (e.g. 's3://my-bucket/datasets/training').
            MUST be provided by the user.

    Returns:
        S3 URI of the uploaded file.
    """
    try:
        import _io as _sio

        _orig = sys.stdout
        sys.stdout = _sio.StringIO()
        try:
            import sagemaker
            from sagemaker.s3 import S3Uploader
        finally:
            sys.stdout = _orig

        if not os.path.exists(local_path):
            return {"status": "error", "message": f"Path not found: {local_path}"}

        if not s3_uri.startswith("s3://"):
            return {"status": "error", "message": f"Invalid S3 URI: '{s3_uri}'. Must start with 's3://'."}

        uploaded = S3Uploader.upload(local_path=local_path, desired_s3_uri=s3_uri)
        logger.info(f"Uploaded {local_path} to {uploaded}")

        return {
            "status": "success",
            "local_path": local_path,
            "s3_uri": uploaded,
            "training_data_s3_uri": uploaded,
            "next_step": f"Dataset uploaded to {uploaded}. Pass this as training_data_s3_uri when calling launch_sft_training_job or launch_grpo_training_job.",
        }
    except Exception as e:
        return {"status": "error", "message": f"S3 upload failed: {e}"}


# ===================================================================
# Entry point
# ===================================================================

def main():
    mcp.run(show_banner=False)

if __name__ == "__main__":
    main()
