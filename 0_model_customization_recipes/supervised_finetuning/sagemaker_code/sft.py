"""
Supervised Fine-Tuning (SFT) script for language models using TRL and Transformers.

This script supports:
- Full fine-tuning and PEFT (LoRA) training
- 4-bit quantization with BitsAndBytesConfig and MXFP4
- Spectrum parameter selection for selective fine-tuning
- Distributed training with DeepSpeed and Accelerate
- Model merging and saving for deployment
"""

import logging
import os
import re
import soundfile as sf
import json
from dataclasses import dataclass
from datetime import datetime
from distutils.util import strtobool
from typing import Optional, Tuple, Dict, Any, List

import torch
from datasets import load_dataset, Dataset
from peft import AutoPeftModelForCausalLM, PeftModel, PeftConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
    BitsAndBytesConfig,
    Mxfp4Config,
    PreTrainedModel,
    PreTrainedTokenizer,
    set_seed,
    Qwen2AudioForConditionalGeneration,
    GenerationConfig
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import is_liger_kernel_available
from trl import (
    SFTTrainer,
    TrlParser,
    ModelConfig,
    SFTConfig,
    get_peft_config
)

if is_liger_kernel_available():
    from liger_kernel.transformers import AutoLigerKernelForCausalLM
import base64
import io
from PIL import Image


# here's a list of models that needs its own import from transformers
EXCEPTION_MODEL_LIST = ["Qwen/Qwen2-Audio-7B-Instruct"]


def process_vision_info(messages: list[dict]) -> list[Image.Image]:
    image_inputs = []
    for msg in messages:
        content = msg.get("content", [])
        if not isinstance(content, list):
            content = [content]

        for element in content:
            # We only care about TRL-style multimodal entries
            if isinstance(element, dict) and element.get("type") == "image_url":
                url = element.get("image_url", {}).get("url", None)
                if url and url.startswith("data:image"):
                    # strip the prefix "data:image/png;base64,"
                    b64_data = url.split(",")[1]
                    img_bytes = base64.b64decode(b64_data)
                    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    image_inputs.append(image)
    return image_inputs


def process_audio_info(messages):
    """
    Extract audio paths from messages and decode into waveform dicts
    that the processor can consume.
    """
    audio_inputs = []
    for msg in messages:
        for element in msg.get("content", []):
            if isinstance(element, dict) and element.get("type") == "audio":
                # Support either audio_url or audio.path
                audio_url = element.get("audio_url") or element.get("audio", {}).get("path")
                if audio_url:
                    # Strip file:// prefix if present
                    path = audio_url.replace("file://", "")
                    array, sr = sf.read(path)
                    audio_inputs.append({"array": array, "sampling_rate": sr})
    return audio_inputs


# Configure logging
def setup_logging() -> logging.Logger:
    """Set up logging configuration for the training script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)


logger = setup_logging()


@dataclass
class ScriptArguments:
    """Custom arguments for the SFT training script."""
    
    dataset_id_or_path: str
    """Path to dataset file (.jsonl) or HuggingFace dataset identifier."""
    
    dataset_splits: str = "train"
    """Dataset splits to use for training."""
    
    tokenizer_name_or_path: Optional[str] = None
    """Path to tokenizer or HuggingFace tokenizer identifier. If None, uses model tokenizer."""

    processor_name_or_path: Optional[str] = None
    """Path to processor or HuggingFace processor identifier. If None, uses model processor."""
    
    spectrum_config_path: Optional[str] = None
    """Path to YAML config file specifying which parameters to unfreeze for Spectrum training."""
    
    max_seq_length: int = 2048
    """Maximum sequence length for tokenization."""
    
    mxfp4: bool = False
    """Whether to use MXFP4 quantization instead of standard 4-bit quantization."""

    use_liger: bool = False
    """Whether to use LigerKernel over AutoClass for loading model."""

    modality_type: Optional[str] = "text"
    """Type of modality to use during training "video", "image", "audio" or "text" """
    
    run_evaluation: bool = True
    """Whether to run post-training evaluation comparing base and fine-tuned models."""
    
    eval_max_samples: int = 100
    """Maximum number of samples to use for evaluation (for efficiency)."""
    
    eval_max_new_tokens: int = 512
    """Maximum number of new tokens to generate during evaluation."""

def get_checkpoint_path(training_args: SFTConfig) -> Optional[str]:
    """
    Get the path to the last checkpoint if it exists.
    
    Args:
        training_args: Training configuration containing output directory
        
    Returns:
        Path to last checkpoint or None if no checkpoint exists
    """
    if os.path.isdir(training_args.output_dir):
        return get_last_checkpoint(training_args.output_dir)
    return None


def setup_model_for_spectrum(model: PreTrainedModel, spectrum_config_path: str) -> PreTrainedModel:
    """
    Configure model for Spectrum training by selectively unfreezing parameters.
    
    Args:
        model: The pretrained model to configure
        spectrum_config_path: Path to YAML file containing parameter patterns to unfreeze
        
    Returns:
        Model with appropriate parameters frozen/unfrozen for Spectrum training
        
    Raises:
        FileNotFoundError: If spectrum config file doesn't exist
        ValueError: If spectrum config file is malformed
    """
    if not os.path.exists(spectrum_config_path):
        raise FileNotFoundError(f"Spectrum config file not found: {spectrum_config_path}")
    
    try:
        with open(spectrum_config_path, "r", encoding="utf-8") as fin:
            yaml_content = fin.read()
    except Exception as e:
        raise ValueError(f"Failed to read spectrum config file: {e}")

    # Extract parameter patterns from YAML
    unfrozen_patterns = []
    for line in yaml_content.splitlines():
        line = line.strip()
        if line.startswith("- "):
            pattern = line[2:].strip()  # Remove "- " prefix
            if pattern:
                unfrozen_patterns.append(pattern)

    if not unfrozen_patterns:
        logger.warning("No parameter patterns found in spectrum config file")

    # Freeze all parameters initially
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze parameters matching the patterns
    unfrozen_count = 0
    for name, param in model.named_parameters():
        if any(re.match(pattern, name) for pattern in unfrozen_patterns):
            param.requires_grad = True
            unfrozen_count += 1
            logger.debug(f"Unfrozen parameter: {name}")

    logger.info(f"Spectrum training: {unfrozen_count} parameters unfrozen using {len(unfrozen_patterns)} patterns")
    return model


def load_datasets(script_args: ScriptArguments) -> Tuple[Dataset, Dataset]:
    """
    Load training and evaluation datasets based on script arguments.
    
    Args:
        script_args: Script arguments containing dataset configuration
        
    Returns:
        Tuple of (train_dataset, eval_dataset)
        
    Raises:
        ValueError: If dataset loading fails or required attributes are missing
    """
    dataset_path = script_args.dataset_id_or_path
    
    try:
        if dataset_path.endswith('.jsonl'):
            # Load local JSONL file
            logger.info(f"Loading JSONL dataset from {dataset_path}")
            dataset = load_dataset('json', data_files=dataset_path, split='train')
            
            # Split dataset (hardcoded split for JSONL files)
            total_samples = len(dataset)
            logger.warning(f"Dataset has only {total_samples} samples, using 90/10 split")
            split_idx = int(0.9 * total_samples)
            train_dataset = dataset.select(range(split_idx))
            eval_dataset = dataset.select(range(split_idx, total_samples))
        else:
            # Load HuggingFace dataset
            logger.info(f"Loading HuggingFace dataset: {dataset_path}")
            
            # Check if we have the required split attributes
            if not hasattr(script_args, 'dataset_train_split'):
                raise ValueError("dataset_train_split not found in script_args for HuggingFace dataset")
            if not hasattr(script_args, 'dataset_test_split'):
                raise ValueError("dataset_test_split not found in script_args for HuggingFace dataset")
            
            config = getattr(script_args, 'config', None)
            if config is not None:
                train_dataset = load_dataset(
                    dataset_path, config, split=script_args.dataset_train_split
                )
                eval_dataset = load_dataset(
                    dataset_path, config, split=script_args.dataset_test_split
                )
            else:
                train_dataset = load_dataset(
                    dataset_path, split=script_args.dataset_train_split
                )
                eval_dataset = load_dataset(
                    dataset_path, split=script_args.dataset_test_split
                )
        
        logger.info(f"Loaded training dataset: {len(train_dataset)} samples, features: {train_dataset.features}")
        logger.info(f"Loaded evaluation dataset: {len(eval_dataset)} samples, features: {eval_dataset.features}")
        
        # Log first sample for debugging
        if len(train_dataset) > 0:
            logger.debug(f"First training sample: {train_dataset[0]}")
        
        return train_dataset, eval_dataset
        
    except Exception as e:
        logger.error(f"Failed to load datasets: {e}")
        raise


def setup_tokenizer(script_args: ScriptArguments, model_args: ModelConfig) -> PreTrainedTokenizer:
    """
    Load and configure the tokenizer.
    
    Args:
        script_args: Script arguments containing tokenizer configuration
        model_args: Model arguments containing model configuration
        
    Returns:
        Configured tokenizer
    """
    tokenizer_name = script_args.tokenizer_name_or_path or model_args.model_name_or_path
    
    logger.info(f"Loading tokenizer from {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )
    
    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token to eos_token")
    
    return tokenizer


def setup_processor(script_args: ScriptArguments, model_args: ModelConfig):
    """
    Load and configure the processors.
    
    Args:
        script_args: Script arguments containing tokenizer configuration
        model_args: Model arguments containing model configuration
        
    Returns:
        Configured processor
    """
    processor_name = script_args.processor_name_or_path or model_args.model_name_or_path
    
    logger.info(f"Loading processor from {processor_name}")
    processor = AutoProcessor.from_pretrained(
        processor_name,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )
    
    return processor


def create_model_kwargs(model_args: ModelConfig, training_args: SFTConfig, script_args: ScriptArguments) -> Dict[str, Any]:
    """
    Create model loading arguments based on configuration.
    
    Args:
        model_args: Model configuration
        training_args: Training configuration  
        script_args: Script arguments
        
    Returns:
        Dictionary of model loading arguments
    """
    # Determine torch dtype
    if model_args.torch_dtype in ['auto', None]:
        torch_dtype = model_args.torch_dtype
    else:
        torch_dtype = getattr(torch, model_args.torch_dtype)
    
    model_kwargs = {
        'revision': model_args.model_revision,
        'trust_remote_code': model_args.trust_remote_code,
        'attn_implementation': model_args.attn_implementation,
        'torch_dtype': torch_dtype,
        # 'use_cache': not training_args.gradient_checkpointing,
    }
    
    # Set low_cpu_mem_usage based on DeepSpeed usage
    use_deepspeed = strtobool(os.environ.get("ACCELERATE_USE_DEEPSPEED", "false"))
    if not use_deepspeed:
        model_kwargs['low_cpu_mem_usage'] = True
    
    # Configure quantization
    if model_args.load_in_4bit:
        if script_args.mxfp4:
            logger.info("Using MXFP4 quantization")
            model_kwargs['quantization_config'] = Mxfp4Config(dequantize=True)
        else:
            logger.info("Using BitsAndBytes 4-bit quantization")
            model_kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_quant_storage=torch_dtype,
            )
    
    return model_kwargs


def load_model(model_args: ModelConfig, training_args: SFTConfig, script_args: ScriptArguments, model_kwargs: Dict[str, Any]) -> PreTrainedModel:
    """
    Load the pretrained model with appropriate configuration.
    
    Args:
        model_args: Model configuration
        training_args: Training configuration
        script_args: Script arguments
        model_kwargs: Model loading arguments
        
    Returns:
        Loaded model
        
    Raises:
        ValueError: If MXFP4 is used with unsupported configurations
    """
    model_name = model_args.model_name_or_path
    
    if script_args.mxfp4:
        logger.info("🌱 Loading model with MXFP4 - skipping Liger kernel")
        # MXFP4 doesn't support Liger kernel yet
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    else:
        # Use Liger kernel if available and requested
        if script_args.use_liger and is_liger_kernel_available():
            logger.info("🐯 Loading model with Liger kernel optimization")
            model = AutoLigerKernelForCausalLM.from_pretrained(model_name, **model_kwargs)
        else:
            logger.info("↔️ Loading standard model")
            if model_name in EXCEPTION_MODEL_LIST:
                if model_name == "Qwen/Qwen2-Audio-7B-Instruct":
                    model = Qwen2AudioForConditionalGeneration.from_pretrained(model_name, **model_kwargs)
                else:
                    raise AssertionError(f"model {model_name} not supported")
            else:
                print(model_name, model_kwargs)
                model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    
    # Wait for all processes in distributed training
    if hasattr(training_args, 'distributed_state'):
        training_args.distributed_state.wait_for_everyone()
    
    return model


def configure_model_for_training(model: PreTrainedModel, script_args: ScriptArguments) -> PreTrainedModel:
    """
    Configure model for specific training requirements (e.g., Spectrum).
    
    Args:
        model: The loaded model
        script_args: Script arguments
        
    Returns:
        Configured model
        
    Raises:
        AssertionError: If Spectrum config is required but not provided for non-MXFP4 training
    """
    if script_args.spectrum_config_path and not script_args.mxfp4:
        logger.info(f"✅ Configuring model for Spectrum training with config: {script_args.spectrum_config_path}")
        return setup_model_for_spectrum(model, script_args.spectrum_config_path)
    elif not script_args.spectrum_config_path and not script_args.mxfp4:
        # This seems to be a bug in the original code - it always raises an error
        # Let's make it more reasonable by only requiring spectrum config when explicitly needed
        logger.warning("🤖 No Spectrum config provided - using standard training")
        return model
    else:
        return model


def get_model_save_directory(model_name: str) -> str:
    """
    Get the directory path for saving the final model.
    
    Args:
        model_name: Name/path of the model
        
    Returns:
        Path to save directory
    """
    if "SM_MODEL_DIR" in os.environ:
        base_dir = os.environ["SM_MODEL_DIR"]
    else:
        base_dir = "/opt/ml/model"
    
    return os.path.join(base_dir, model_name)


def save_peft_model(trainer: SFTTrainer, training_args: SFTConfig, model_args: ModelConfig) -> None:
    """
    Save PEFT model, merge with base model, and save final merged model.
    
    Args:
        trainer: The SFT trainer instance
        training_args: Training configuration
        model_args: Model configuration
    """
    final_model_dir = get_model_save_directory(model_args.model_name_or_path)
    final_model_dir = os.path.join(final_model_dir, "peft")

    logger.info("Saving PEFT model")
    
    # Save adapter to final model dir directory
    trainer.save_model(final_model_dir)
    logger.info(f"PEFT adapter saved to {final_model_dir}")
    
    # Wait for all processes
    if hasattr(training_args, 'distributed_state'):
        training_args.distributed_state.wait_for_everyone()
    
    # Save tokenizer
    trainer.tokenizer.save_pretrained(final_model_dir)
    logger.info(f"Tokenizer saved to {final_model_dir}")


def save_full_model(trainer: SFTTrainer, training_args: SFTConfig, model_args: ModelConfig) -> None:
    """
    Save full fine-tuned model (non-PEFT).
    
    Args:
        trainer: The SFT trainer instance
        training_args: Training configuration
        model_args: Model configuration
    """
    logger.info("Saving full fine-tuned model")
    
    # Save model to final directory
    final_model_dir = get_model_save_directory(model_args.model_name_or_path)
    trainer.save_model(final_model_dir)
    logger.info(f"Model saved to {final_model_dir}")
    
    # Wait for all processes
    if hasattr(training_args, 'distributed_state'):
        training_args.distributed_state.wait_for_everyone()
    
    # Save tokenizer (fix bug: was saving to wrong directory)
    trainer.tokenizer.save_pretrained(final_model_dir)
    logger.info(f"Tokenizer saved to {final_model_dir}")


def load_model_for_evaluation(model_path: str, model_args: ModelConfig, script_args: ScriptArguments, is_peft: bool = False) -> Tuple[PreTrainedModel, Any]:
    """
    Load a model for evaluation (either base model or fine-tuned model).
    
    Args:
        model_path: Path to the model
        model_args: Model configuration
        script_args: Script arguments
        is_peft: Whether this is a PEFT model
        
    Returns:
        Tuple of (model, tokenizer_or_processor)
    """
    logger.info(f"Loading model for evaluation from: {model_path}")
    
    # Setup tokenizer/processor
    if script_args.processor_name_or_path or script_args.modality_type in ["image", "audio"]:
        # Use processor for multimodal models
        processor_path = script_args.processor_name_or_path or model_path
        tokenizer_or_processor = AutoProcessor.from_pretrained(
            processor_path,
            revision=model_args.model_revision,
            trust_remote_code=model_args.trust_remote_code,
        )
    else:
        # Use tokenizer for text-only models
        tokenizer_path = script_args.tokenizer_name_or_path or model_path
        tokenizer_or_processor = AutoTokenizer.from_pretrained(
            tokenizer_path,
            revision=model_args.model_revision,
            trust_remote_code=model_args.trust_remote_code,
        )
        if tokenizer_or_processor.pad_token is None:
            tokenizer_or_processor.pad_token = tokenizer_or_processor.eos_token
    
    # Create model kwargs for evaluation (no quantization for cleaner inference)
    eval_model_kwargs = {
        'revision': model_args.model_revision,
        'trust_remote_code': model_args.trust_remote_code,
        'torch_dtype': torch.bfloat16,  # Use bfloat16 for evaluation
        'device_map': 'auto',
        'low_cpu_mem_usage': True,
    }
    
    # Load model
    if is_peft:
        # Load PEFT model
        logger.info("Loading PEFT model for evaluation")
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_path,
            **eval_model_kwargs
        )
    else:
        # Load base or full fine-tuned model
        if model_args.model_name_or_path in EXCEPTION_MODEL_LIST:
            if model_args.model_name_or_path == "Qwen/Qwen2-Audio-7B-Instruct":
                model = Qwen2AudioForConditionalGeneration.from_pretrained(model_path, **eval_model_kwargs)
            else:
                raise AssertionError(f"model {model_args.model_name_or_path} not supported")
        else:
            model = AutoModelForCausalLM.from_pretrained(model_path, **eval_model_kwargs)
    
    model.eval()  # Set to evaluation mode
    return model, tokenizer_or_processor


def prepare_messages_for_generation(messages: List[Dict]) -> List[Dict]:
    """
    Prepare messages for generation by removing the last assistant response.
    This allows us to generate a new response for the last user message.
    
    Args:
        messages: List of message dictionaries
        
    Returns:
        Messages without the last assistant response
    """
    # Find the last assistant message and remove it
    generation_messages = []
    last_assistant_idx = -1
    
    for i, msg in enumerate(messages):
        if msg["role"] == "assistant":
            last_assistant_idx = i
    
    # Include all messages except the last assistant response
    for i, msg in enumerate(messages):
        if i != last_assistant_idx:
            generation_messages.append(msg)
    
    return generation_messages


def generate_response(model: PreTrainedModel, tokenizer_or_processor: Any, messages: List[Dict], script_args: ScriptArguments, max_new_tokens: int = 512) -> str:
    """
    Generate a response from the model given input messages.
    
    Args:
        model: The model to use for generation
        tokenizer_or_processor: Tokenizer or processor
        messages: List of message dictionaries
        script_args: Script arguments
        max_new_tokens: Maximum number of new tokens to generate
        
    Returns:
        Generated response text
    """
    try:
        # Prepare messages for generation (remove last assistant response)
        generation_messages = prepare_messages_for_generation(messages)
        
        # Handle different modalities
        if script_args.modality_type == "text":
            # Text-only generation
            prompt = tokenizer_or_processor.apply_chat_template(
                generation_messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer_or_processor(prompt, return_tensors="pt", padding=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
        elif script_args.modality_type == "image":
            # Vision-language generation
            prompt = tokenizer_or_processor.apply_chat_template(
                generation_messages, tokenize=False, add_generation_prompt=True
            )
            images = process_vision_info(generation_messages)
            inputs = tokenizer_or_processor(
                images=images if images else None,
                text=prompt,
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
        elif script_args.modality_type == "audio":
            # Audio-language generation
            prompt = tokenizer_or_processor.apply_chat_template(
                generation_messages, tokenize=False, add_generation_prompt=True
            )
            audios = process_audio_info(generation_messages)
            inputs = tokenizer_or_processor(
                audios=audios if audios else None,
                text=prompt,
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
        else:
            raise ValueError(f"Unsupported modality type: {script_args.modality_type}")
        
        # Generate response
        # Get pad and eos token IDs safely
        if hasattr(tokenizer_or_processor, 'pad_token_id'):
            pad_token_id = tokenizer_or_processor.pad_token_id
            eos_token_id = tokenizer_or_processor.eos_token_id
        else:
            pad_token_id = tokenizer_or_processor.tokenizer.pad_token_id
            eos_token_id = tokenizer_or_processor.tokenizer.eos_token_id
        
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                generation_config=generation_config,
            )
        
        # Decode response (only the new tokens)
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        
        if hasattr(tokenizer_or_processor, 'decode'):
            response = tokenizer_or_processor.decode(generated_tokens, skip_special_tokens=True)
        else:
            response = tokenizer_or_processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return response.strip()
        
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return f"[ERROR: {str(e)}]"


def run_evaluation(
    eval_dataset: Dataset,
    model_args: ModelConfig,
    script_args: ScriptArguments,
    training_args: SFTConfig,
    max_samples: int = 100
) -> None:
    """
    Run evaluation on both base model and fine-tuned model, save results to JSONL files.
    
    Args:
        eval_dataset: Evaluation dataset
        model_args: Model configuration
        script_args: Script arguments
        training_args: Training configuration
        max_samples: Maximum number of samples to evaluate (for efficiency)
    """
    logger.info("=" * 50)
    logger.info("Starting Model Evaluation")
    logger.info("=" * 50)
    
    # Create evaluation output directory
    eval_output_dir = os.path.join(training_args.output_dir, "evaluation_output")
    os.makedirs(eval_output_dir, exist_ok=True)
    
    # Limit evaluation samples for efficiency
    eval_samples = min(len(eval_dataset), max_samples)
    eval_subset = eval_dataset.select(range(eval_samples))
    logger.info(f"Evaluating on {eval_samples} samples")
    
    # Determine model paths and types
    base_model_path = model_args.model_name_or_path
    
    if model_args.use_peft:
        # PEFT model
        fine_tuned_model_path = os.path.join(get_model_save_directory(model_args.model_name_or_path), "peft")
        is_peft = True
    else:
        # Full fine-tuned model
        fine_tuned_model_path = get_model_save_directory(model_args.model_name_or_path)
        is_peft = False
    
    # Evaluate base model
    logger.info(f"Evaluating base model from: {base_model_path}")
    try:
        base_model, base_tokenizer = load_model_for_evaluation(
            base_model_path, model_args, script_args, is_peft=False
        )
        logger.info("Base model loaded successfully")
        
        base_results = []
        for i, sample in enumerate(eval_subset):
            logger.info(f"Base model - Processing sample {i+1}/{eval_samples}")
            
            messages = sample.get("messages", [])
            if not messages:
                logger.warning(f"Sample {i} has no messages, skipping")
                continue
            
            # Generate response
            response = generate_response(
                base_model, base_tokenizer, messages, script_args, 
                max_new_tokens=script_args.eval_max_new_tokens
            )
            
            # Create messages format result
            # Extract ground truth (original assistant responses)
            ground_truth = []
            messages_with_prediction = []
            assistant_count = 0
            
            for msg in messages:
                if msg["role"] == "assistant":
                    # This is the ground truth
                    ground_truth.append({
                        "role": "assistant",
                        "content": msg["content"],
                        "thinking": msg.get("thinking", None)
                    })
                    
                    # For the first assistant message, replace with model prediction
                    # For subsequent ones, keep original (multi-turn conversations)
                    if assistant_count == 0:
                        messages_with_prediction.append({
                            "role": "assistant", 
                            "content": response,
                            "thinking": None
                        })
                    else:
                        messages_with_prediction.append({
                            "role": "assistant",
                            "content": msg["content"],
                            "thinking": msg.get("thinking", None)
                        })
                    assistant_count += 1
                else:
                    # Keep system and user messages as-is
                    messages_with_prediction.append({
                        "role": msg["role"],
                        "content": msg["content"],
                        "thinking": msg.get("thinking", None)
                    })
            
            # Store result in messages format
            result = {
                "messages": messages_with_prediction,
                "ground_truth": ground_truth
            }
            base_results.append(result)
        
        # Save base model results
        base_results_path = os.path.join(eval_output_dir, "basemodel.jsonl")
        with open(base_results_path, 'w', encoding='utf-8') as f:
            for result in base_results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        logger.info(f"Base model results saved to: {base_results_path}")
        
        # Clean up base model
        del base_model, base_tokenizer
        torch.cuda.empty_cache()
        
    except Exception as e:
        logger.error(f"Error evaluating base model: {e}")
        base_results = []
    
    # Evaluate fine-tuned model
    logger.info(f"Evaluating fine-tuned model from: {fine_tuned_model_path}")
    logger.info(f"Model type: {'PEFT' if is_peft else 'Full fine-tuned'}")
    
    # Check if fine-tuned model path exists
    if not os.path.exists(fine_tuned_model_path):
        logger.error(f"Fine-tuned model path does not exist: {fine_tuned_model_path}")
        fine_tuned_results = []
    else:
        try:
            fine_tuned_model, fine_tuned_tokenizer = load_model_for_evaluation(
                fine_tuned_model_path, model_args, script_args, is_peft=is_peft
            )
            logger.info("Fine-tuned model loaded successfully")
            
            fine_tuned_results = []
            for i, sample in enumerate(eval_subset):
                logger.info(f"Fine-tuned model - Processing sample {i+1}/{eval_samples}")
                
                messages = sample.get("messages", [])
                if not messages:
                    continue
                
                # Generate response
                response = generate_response(
                    fine_tuned_model, fine_tuned_tokenizer, messages, script_args,
                    max_new_tokens=script_args.eval_max_new_tokens
                )
                
                # Create messages format result
                # Extract ground truth (original assistant responses)
                ground_truth = []
                messages_with_prediction = []
                assistant_count = 0
                
                for msg in messages:
                    if msg["role"] == "assistant":
                        # This is the ground truth
                        ground_truth.append({
                            "role": "assistant",
                            "content": msg["content"],
                            "thinking": msg.get("thinking", None)
                        })
                        
                        # For the first assistant message, replace with model prediction
                        # For subsequent ones, keep original (multi-turn conversations)
                        if assistant_count == 0:
                            messages_with_prediction.append({
                                "role": "assistant", 
                                "content": response,
                                "thinking": None
                            })
                        else:
                            messages_with_prediction.append({
                                "role": "assistant",
                                "content": msg["content"],
                                "thinking": msg.get("thinking", None)
                            })
                        assistant_count += 1
                    else:
                        # Keep system and user messages as-is
                        messages_with_prediction.append({
                            "role": msg["role"],
                            "content": msg["content"],
                            "thinking": msg.get("thinking", None)
                        })
                
                # Store result in messages format
                result = {
                    "messages": messages_with_prediction,
                    "ground_truth": ground_truth
                }
                fine_tuned_results.append(result)
            
            # Save fine-tuned model results
            fine_tuned_results_path = os.path.join(eval_output_dir, "fine-tunedmodel.jsonl")
            with open(fine_tuned_results_path, 'w', encoding='utf-8') as f:
                for result in fine_tuned_results:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
            logger.info(f"Fine-tuned model results saved to: {fine_tuned_results_path}")
            
            # Clean up fine-tuned model
            del fine_tuned_model, fine_tuned_tokenizer
            torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"Error evaluating fine-tuned model: {e}")
            fine_tuned_results = []
    
    # Create comparison results (optional - side-by-side comparison)
    if base_results and fine_tuned_results:
        logger.info("Creating comparison results...")
        comparison_results = []
        
        for base_result, ft_result in zip(base_results, fine_tuned_results):
            # Extract base model response from messages
            base_response = None
            ft_response = None
            ground_truth_content = None
            
            # Get base model response
            for msg in base_result["messages"]:
                if msg["role"] == "assistant":
                    base_response = msg["content"]
                    break
            
            # Get fine-tuned model response  
            for msg in ft_result["messages"]:
                if msg["role"] == "assistant":
                    ft_response = msg["content"]
                    break
            
            # Get ground truth
            if base_result["ground_truth"]:
                ground_truth_content = base_result["ground_truth"][0]["content"]
            
            comparison_result = {
                "ground_truth": ground_truth_content,
                "base_model_response": base_response,
                "fine_tuned_model_response": ft_response,
                "training_method": "peft" if model_args.use_peft else ("spectrum" if script_args.spectrum_config_path else "full")
            }
            comparison_results.append(comparison_result)
        
        # Save comparison results
        comparison_results_path = os.path.join(eval_output_dir, "model_comparison.jsonl")
        with open(comparison_results_path, 'w', encoding='utf-8') as f:
            for result in comparison_results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        logger.info(f"Comparison results saved to: {comparison_results_path}")
    
    # Create evaluation summary
    summary = {
        "evaluation_timestamp": datetime.now().isoformat(),
        "model_name": model_args.model_name_or_path,
        "training_method": "peft" if model_args.use_peft else ("spectrum" if script_args.spectrum_config_path else "full"),
        "eval_samples": eval_samples,
        "base_model_evaluated": len(base_results) > 0,
        "fine_tuned_model_evaluated": len(fine_tuned_results) > 0,
        "base_model_path": base_model_path,
        "fine_tuned_model_path": fine_tuned_model_path,
        "output_files": {
            "base_model_responses": "basemodel.jsonl" if base_results else None,
            "fine_tuned_model_responses": "fine-tunedmodel.jsonl" if fine_tuned_results else None,
            "model_comparison": "model_comparison.jsonl" if base_results and fine_tuned_results else None
        }
    }
    
    summary_path = os.path.join(eval_output_dir, "evaluation_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Evaluation summary saved to: {summary_path}")
    
    logger.info("=" * 50)
    logger.info("Model Evaluation Completed")
    logger.info("=" * 50)


def train_function(model_args: ModelConfig, script_args: ScriptArguments, training_args: SFTConfig) -> None:
    """
    Main training function that orchestrates the entire SFT process.
    
    Args:
        model_args: Model configuration from TRL parser
        script_args: Custom script arguments
        training_args: Training configuration from TRL parser
    """
    logger.info("=" * 50)
    logger.info("Starting Supervised Fine-Tuning")
    logger.info("=" * 50)

    logger.info(f"\n\n🌀🌀🌀 MODALITY: {script_args.modality_type} 🌀🌀🌀")

    
    # Log all parameters
    logger.info(f"Model parameters: {model_args}")
    logger.info(f"Script parameters: {script_args}")  
    logger.info(f"Training parameters: {training_args}")

    # Load datasets
    train_dataset, eval_dataset = load_datasets(script_args)
    
    # Setup tokenizer
    tokenizer_or_processor = None
    if script_args.tokenizer_name_or_path:
        tokenizer_or_processor = setup_tokenizer(script_args, model_args)
    elif script_args.processor_name_or_path:
        tokenizer_or_processor = setup_processor(script_args, model_args)
    else:
        assert tokenizer_or_processor is not None, "please specify `tokenizer_name_or_path` (text) or `processor_name_or_path` (vision)"
    
    # Configure PEFT if needed
    peft_config = None
    if model_args.use_peft:
        logger.info(
            "\n\n"
            "🪫🔋🪫🔋🪫🔋🪫🔋🪫🔋🪫🔋🪫🔋🪫🪫🔋🪫🔋🪫🔋🪫🔋🪫🔋🪫🔋🪫🔋🪫🪫🔋🪫🔋🪫🪫\n"
            "🪫   CONFIGURING PEFT    🪫\n"
            "🪫🔋🪫🔋🪫🔋🪫🔋🪫🔋🪫🔋🪫🔋🪫🪫🔋🪫🔋🪫🔋🪫🔋🪫🔋🪫🔋🪫🔋🪫🪫🔋🪫🔋🪫🪫\n"
        )

        peft_config = get_peft_config(model_args)
    else:
        spectrum_bool = False
        if script_args.spectrum_config_path:
            spectrum_bool = True
        logger.info(
            "\n\n"
            "🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟\n"
            "🌟   RUNNING FINE-TUNING (Spectrum: %s)    🌟\n"
            "🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟\n"
            % spectrum_bool
        )

    
    # Load and configure model
    model_kwargs = create_model_kwargs(model_args, training_args, script_args)
    model = load_model(model_args, training_args, script_args, model_kwargs)
    model = configure_model_for_training(model, script_args)

    def collate_fn_images(examples):
        # Convert chat messages to text template
        texts = [
            tokenizer_or_processor.apply_chat_template(
                example["messages"], tokenize=False, add_generation_prompt=False
            ).strip()
            for example in examples
        ]
    
        # Extract images from messages (always multimodal-safe)
        images = [process_vision_info(example["messages"]) for example in examples]
    
        # Tokenize texts and images
        batch = tokenizer_or_processor(
            images=images, 
            text=texts, 
            return_tensors="pt", 
            padding=True
        )
    
        # Prepare labels (mask padding + special image tokens)
        labels = batch["input_ids"].clone()
        labels[labels == tokenizer_or_processor.tokenizer.pad_token_id] = -100
    
        # Mask image tokens (boi/eoi etc.)
        image_token_ids = [
            tokenizer_or_processor.tokenizer.convert_tokens_to_ids(tok)
            for tok in tokenizer_or_processor.tokenizer.special_tokens_map.values()
            if "image" in tok.lower() or "boi" in tok.lower() or "eoi" in tok.lower()
        ]
        for tok_id in image_token_ids:
            labels[labels == tok_id] = -100
    
        batch["labels"] = labels
        return batch
    
    
    def collate_fn_audio(examples):
        # 1. Convert chat messages into text prompts
        texts = [
            tokenizer_or_processor.apply_chat_template(
                example["messages"], tokenize=False, add_generation_prompt=False
            ).strip()
            for example in examples
        ]
    
        # 2. Extract audio waveforms for each example
        audios = [process_audio_info(example["messages"]) for example in examples]
    
        # 3. Tokenize text + audio together
        batch = tokenizer_or_processor(
            audios=audios,
            text=texts,
            return_tensors="pt",
            padding=True,
        )
    
        # 4. Build labels (mask padding tokens)
        labels = batch["input_ids"].clone()
        labels[labels == tokenizer_or_processor.tokenizer.pad_token_id] = -100
    
        # 5. Mask audio special tokens (<|audio_bos|>, <|audio_eos|>, etc.)
        special_tokens = tokenizer_or_processor.tokenizer.special_tokens_map.values()
    
        audio_tokens = []
        for tok in special_tokens:
            if isinstance(tok, str):
                audio_tokens.append(tok)
            elif isinstance(tok, list):
                audio_tokens.extend(tok)
    
        audio_token_ids = [
            tokenizer_or_processor.tokenizer.convert_tokens_to_ids(tok)
            for tok in audio_tokens
            if isinstance(tok, str)
            and ("audio" in tok.lower() or "bo" in tok.lower() or "eo" in tok.lower())
        ]
    
        for tok_id in audio_token_ids:
            labels[labels == tok_id] = -100
    
        batch["labels"] = labels
        return batch

    # collate functions are applicable for multi-modal datasets like images/video/audio
    collator_fn = None
    if script_args.modality_type == "text":
        pass
    elif script_args.modality_type == "image":
        collator_fn = collate_fn_images
    elif script_args.modality_type == "video":
        raise AssertionError(f"current modality {script_args.modality_type} is unsupported!")
    elif script_args.modality_type == "audio":
        collator_fn = collate_fn_audio
    else:
        raise AssertionError(f"current modality {script_args.modality_type} is unsupported - choose `image`, `video`, `audio` or `text`!")
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        data_collator=collator_fn,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,  # Add eval dataset
        processing_class=tokenizer_or_processor,
        peft_config=peft_config,
    )
    
    # Print trainable parameters for PEFT
    if trainer.accelerator.is_main_process and peft_config:
        trainer.model.print_trainable_parameters()

    # Check for existing checkpoint
    last_checkpoint = get_checkpoint_path(training_args)
    if last_checkpoint and training_args.resume_from_checkpoint is None:
        logger.info(f"Resuming training from checkpoint: {last_checkpoint}")

    # Start training
    start_time = datetime.now()
    logger.info(f"Starting training at {start_time.strftime('%Y-%m-%d %H:%M:%S')} for {training_args.num_train_epochs} epochs")
    
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    
    # Log training metrics
    metrics = train_result.metrics
    metrics['train_samples'] = len(train_dataset)
    trainer.log_metrics('train', metrics)
    trainer.save_metrics('train', metrics)
    trainer.save_state()
    
    # Prepare model for inference
    if trainer.is_fsdp_enabled and peft_config:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type('FULL_STATE_DICT')
    
    # Restore cache for inference
    trainer.model.config.use_cache = True
    
    # Save model based on training type
    if model_args.use_peft:
        save_peft_model(trainer, training_args, model_args)
    else:
        save_full_model(trainer, training_args, model_args)
    
    # Wait for all processes before evaluation
    if hasattr(training_args, 'distributed_state'):
        training_args.distributed_state.wait_for_everyone()
    
    # Run evaluation on main process only
    if trainer.accelerator.is_main_process and script_args.run_evaluation:
        logger.info("Starting post-training evaluation...")
        try:
            # Clean up trainer to free memory
            del trainer
            torch.cuda.empty_cache()
            
            # Run evaluation
            run_evaluation(
                eval_dataset, 
                model_args, 
                script_args, 
                training_args, 
                max_samples=script_args.eval_max_samples
            )
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            logger.warning("Continuing without evaluation...")
    elif not script_args.run_evaluation:
        logger.info("Skipping post-training evaluation (run_evaluation=False)")
    
    end_time = datetime.now()
    training_duration = end_time - start_time
    logger.info(f"Training completed successfully in {training_duration}")
    logger.info("=" * 50)



def main() -> None:
    """
    Main entry point for the SFT training script.
    
    Parses arguments using TRL parser and runs the training function.
    """
    try:
        # Parse arguments using TRL parser (preserving core functionality)
        parser = TrlParser((ModelConfig, ScriptArguments, SFTConfig))
        model_args, script_args, training_args = parser.parse_args_and_config()

        # Set seed for reproducibility
        set_seed(training_args.seed)
        logger.info(f"Set random seed to {training_args.seed}")

        # Run the main training loop
        train_function(model_args, script_args, training_args)
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise


if __name__ == '__main__':
    main()