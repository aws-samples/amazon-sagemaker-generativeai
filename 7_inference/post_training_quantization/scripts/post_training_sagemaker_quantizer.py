import subprocess

package_name = "flash-attn"

try:
    subprocess.check_call(["pip", "uninstall", package_name, "-y"])
    print(f"Successfully uninstalled {package_name}")
except subprocess.CalledProcessError as e:
    print(f"Failed to uninstall {package_name}: {e}")

import base64
from io import BytesIO
import argparse
import logging
import os
import torch
from typing import Dict, Any
from datasets import load_dataset
from qwen_vl_utils import process_vision_info
import transformers
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    AutoProcessor
)
from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.modifiers.quantization import GPTQModifier


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


SUPPORTED_AWQ_QUANT_SCHEMES = ["W4A16_ASYM", "W4A16"]
SUPPORTED_GPTQ_QUANT_SCHEMES = ["W4A16", "W4A16_ASYM", "W8A8", "W8A16"]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for model quantization.

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="Quantize a language model using AWQ or GPTQ on Amazon SageMakerAI")

    # Model and dataset parameters
    parser.add_argument(
        "--model-id",
        type=str,
        required=True,
        help="Hugging Face model ID"
    )
    parser.add_argument(
        "--sequential-loading",
        type=bool,
        default="store_false",
        help="If the quantization model size GPU set this param to true to run sequential loading to optimize on a single GPU"
    )

    # Dataset used for quantization params
    parser.add_argument(
        "--dataset-id",
        type=str,
        required=True,
        help="Hugging Face dataset ID"
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="test",
        help="Dataset split to use for calibration"
    )
    parser.add_argument(
        "--dataset-seed",
        type=int,
        default=42,
        help="Deterministic dataset seed"
    )
    parser.add_argument(
        "--num-calibration-samples",
        type=int,
        default=256,
        help="Number of samples for calibration, larger value <> better quantized model"
    )
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=512,
        help="Maximum sequence length for tokenization"
    )
    parser.add_argument(
        "--vision-enabled",
        action="store_true",
        help="Weather to use images during quanitzation with vision models"
    )

    parser.add_argument(
        "--transformer-model-name",
        type=str,
        default=None,
        help="Need a dynamic transformer import mechanism for varying types"
    )

    parser.add_argument(
        "--vision-sequential-targets",
        type=str,
        default=None,
        help="Vision model sequential targets"
    )

    # Quantization type
    parser.add_argument(
        "--algorithm",
        type=str,
        default="awq",
        choices=["awq", "gptq"],
        help="Quantization Algorithm to use"
    )

    parser.add_argument(
        "--ignore-layers",
        type=str,
        default="lm_head",
        help="Ignore layers to quantize, comma separated"
    )
    parser.add_argument(
        "--include-targets",
        type=str,
        default="Linear",
        help="Targets to quantize including, comma separated"
    )

    # Quantization parameters for AWQ
    parser.add_argument(
        "--awq-quantization-scheme",
        type=str,
        default="W4A16",
        choices=SUPPORTED_AWQ_QUANT_SCHEMES,
        help="AWQ Param: Quantization scheme to use"
    )

    # Quantization parameters for AWQ
    parser.add_argument(
        "--gptq-quantization-scheme",
        type=str,
        default="W4A16",
        choices=SUPPORTED_GPTQ_QUANT_SCHEMES,
        help="GPTQ Param: Quantization scheme to use"
    )

    # SageMaker specific
    parser.add_argument(
        "--sm-model-dir",
        type=str,
        default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"),
        help="Directory to save quantized model"
    )

    return parser.parse_args()


def preprocess_data(
    dataset: Any,
    tokenizer: AutoTokenizer,
    max_sequence_length: int
) -> Any:
    """Preprocess and tokenize dataset for quantization.

    Args:
        dataset: Hugging Face dataset
        tokenizer: Model tokenizer
        max_sequence_length: Maximum sequence length for tokenization

    Returns:
        Any: Processed dataset
    """
    def preprocess(example):
        return {
            "text": tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
            )
        }

    def tokenize(sample: Dict) -> Dict:
        return tokenizer(
            sample["text"],
            padding=False,
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=False,
        )

    dataset = dataset.map(preprocess)
    dataset = dataset.map(tokenize,  remove_columns=dataset.column_names)
    return dataset


def data_collator(batch):
    assert len(batch) == 1
    return {key: torch.tensor(value) for key, value in batch[0].items()}


def preprocess_data_vision(
    dataset: Any,
    processor: AutoProcessor,
    max_sequence_length: int
) -> Any:
    """Preprocess and tokenize dataset for quantization.

    Args:
        dataset: Hugging Face dataset
        tokenizer: Model tokenizer
        max_sequence_length: Maximum sequence length for tokenization

    Returns:
        Any: Processed dataset
    """
    def preprocess_and_tokenize(example):
        # preprocess
        buffered = BytesIO()
        example["image"].save(buffered, format="PNG")
        encoded_image = base64.b64encode(buffered.getvalue())
        encoded_image_text = encoded_image.decode("utf-8")
        base64_qwen = f"data:image;base64,{encoded_image_text}"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": base64_qwen},
                    {"type": "text", "text": "What does the image show?"},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, 
            add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        # tokenize
        return processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=False,
            max_length=max_sequence_length,
            truncation=True,
        )

    dataset = dataset.map(preprocess_and_tokenize,  remove_columns=dataset.column_names)
    return dataset


def quantize_model(
    args: argparse.Namespace
) -> None:
    """Quantize the model using AWQ/GPTQ and save it to disk using oneshot

    Args:
        args: Parsed command line arguments
    """
    try:

        if args.vision_enabled:
            logger.info(f"Loading model: {args.model_id} with vision-text-to-text!")
            
            assert args.transformer_model_name is not None, f"vision_enabled: {args.vision_enabled}, {args.transformer_model_name} cannot be none!"
            assert args.vision_sequential_targets is not None, f"vision_enabled: {args.vision_enabled}, {args.vision_sequential_targets} cannot be none!"
            # dynamic model loading
            ModelClass = getattr(transformers, f"{args.transformer_model_name}")
            # load from pretrained
            model = ModelClass.from_pretrained(
                args.model_id,
                torch_dtype="auto",
                device_map=None,
                trust_remote_code=True
            )
            # load processor
            tokenizer_or_processor = AutoProcessor.from_pretrained(
                args.model_id,
                trust_remote_code=True
            )
        else:
            logger.info(f"Loading model: {args.model_id} with text-to-text only!")
            
            # load model
            model = AutoModelForCausalLM.from_pretrained(
                args.model_id,
                torch_dtype="auto",
                device_map=None,
                trust_remote_code=True
            )
            # load tokenizer
            tokenizer_or_processor = AutoTokenizer.from_pretrained(
                args.model_id,
                trust_remote_code=True
            )

        # save the model first
        base_model_path = os.path.join(
            args.sm_model_dir, 
            args.model_id.rstrip("/").split("/")[-1]
        )
        if not os.path.exists(base_model_path):
            logger.info(f"saving base model to disk...")
            model.save_pretrained(
                base_model_path
            )
            tokenizer_or_processor.save_pretrained(
                base_model_path
            )
        else:
            logger.info(f"skipping base model to disk...")

        # load dataset to calibrate quantization algorithm
        logger.info(f"Loading dataset: {args.dataset_id}")
        dataset = load_dataset(
            args.dataset_id,
            split=f"{args.dataset_split}[:{args.num_calibration_samples}]"
        )
        # shuffle before applying quantization
        dataset = dataset.shuffle(seed=args.dataset_seed)

        logger.info(f"Preprocessing dataset with sequence length: {args.max_sequence_length}")

        if args.vision_enabled:
            processed_dataset = preprocess_data_vision(
                dataset,
                tokenizer_or_processor,
                args.max_sequence_length
            )
        else:
            processed_dataset = preprocess_data(
                dataset,
                tokenizer_or_processor,
                args.max_sequence_length
            )

        ##########
        #   AWQ  #
        ##########
        logger.info(f"Configuring {args.algorithm.upper()} quantization")
        if args.algorithm == "awq":

            quant_scheme = args.awq_quantization_scheme
            recipe = [
                AWQModifier(
                    ignore=[val.rstrip() for val in args.ignore_layers.split(',')],
                    scheme=args.awq_quantization_scheme,
                    targets=[val.rstrip() for val in args.include_targets.split(',')]
                )
            ]

        ###########
        #   GPTQ  #
        ###########
        elif args.algorithm == "gptq":

            quant_scheme = args.gptq_quantization_scheme
            recipe = [
                GPTQModifier(
                    ignore=[val.rstrip() for val in args.ignore_layers.split(',')],
                    scheme=args.gptq_quantization_scheme,
                    targets=[val.rstrip() for val in args.include_targets.split(',')]
                )
            ]
        # Create output directory
        save_dir = os.path.join(
            args.sm_model_dir,
            args.model_id.rstrip("/").split("/")[-1] + f"-{args.algorithm.upper()}-{quant_scheme}"
        )
        os.makedirs(save_dir, exist_ok=True)
        logger.info(f"Applying quantization and saving model to: {save_dir} with vars: {vars(recipe[0])}")

        if args.vision_enabled:
            logger.info(f"selecting targets: {args.vision_sequential_targets.split(',')}")
            oneshot(
                model=model,
                dataset=processed_dataset,
                recipe=recipe,
                max_seq_length=args.max_sequence_length,
                num_calibration_samples=args.num_calibration_samples,
                output_dir=save_dir,
                data_collator=data_collator,
                sequential_targets=args.vision_sequential_targets.split(','),
                trust_remote_code_model=True,
            )
        else:
            oneshot(
                model=model,
                dataset=processed_dataset,
                recipe=recipe,
                max_seq_length=args.max_sequence_length,
                num_calibration_samples=args.num_calibration_samples,
                output_dir=save_dir,
                trust_remote_code_model=True
            )

    except Exception as e:
        logger.error(f"Error during quantization: {str(e)}")
        raise


def main():
    """Main function to run model quantization."""
    args = parse_args()

    logger.info("="*35)
    logger.info(f"runtime arguments: {args}")
    logger.info("="*35)
    quantize_model(args)


if __name__ == "__main__":
    main()
