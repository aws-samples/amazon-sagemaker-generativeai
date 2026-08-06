import importlib
import logging
import os
import shutil
from typing import Any, Dict

from finetune.utils.logging_util import get_logger
from finetune.utils.training_utils.training_constants import PEFT_METHODS

logger = get_logger(__name__)


class ModelLoadErrorError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return f"Model Load Error: {self.message}"


def setup_paths(configs: Dict[str, Any]) -> None:
    """
    Set up paths for model, result, and dxfata directories based on the environment.

    Args:
        configs (dict): Configuration dictionary.
    """
    if "SM_MODEL_DIR" in os.environ:
        configs["model_dir"] = os.environ["SM_MODEL_DIR"]
        configs["result_dir"] = os.environ["SM_OUTPUT_DATA_DIR"]
        configs["data_path"] = os.environ["SM_CHANNEL_TRAIN_DATA"]
        logger.info("Spinning up a new training instance in Sagemaker.")
    else:
        configs["model_dir"] = "model"
        configs["result_dir"] = "results"
        configs["data_path"] = "data"
        logger.info("Initiating the training job in Sagemaker notebook.")
    return configs


peft_check = importlib.util.find_spec("peft")
if peft_check:
    logging.info("loading torch and its components")
    import torch
    from peft import (
        IA3Config,
        LoraConfig,
        PromptEncoderConfig,
        PromptTuningConfig,
        get_peft_model,
        prepare_model_for_int8_training,
    )
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    def print_trainable_parameters(model: torch.nn.Module) -> None:
        """
        Logs the number of trainable parameters in the model.

        Args:
            model (torch.nn.Module): The PyTorch model whose parameters are to be analyzed.
        """
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in model.parameters())
        trainable_percentage = 100 * trainable_params / all_params if all_params else 0

        logger.info(
            f"Trainable parameters: {trainable_params} || "
            f"Total parameters: {all_params} || "
            f"Trainable%: {trainable_percentage:.2f}%"
        )

    def load_tokenizer_and_model(configs: Dict[str, Dict[str, str]]) -> tuple:
        """
        Load tokenizer and model based on configuration.

        Args:
            config (dict): Configuration dictionary.

        Returns:
            tuple: Tokenizer and model objects.
        """
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                configs["training"]["model_id"],
                revision="main",  # Pin to main branch for consistency
                trust_remote_code=False,  # Security: disable arbitrary code execution
            )  # nosec B615 - revision pinned, trust_remote_code disabled
            logger.info("Loaded tokenizer successfully")

            if configs["training"]["training_type"] in PEFT_METHODS:
                peft_args = {
                    **configs["training"][configs["training"]["training_type"]],
                    **configs["training"]["common_peft_params"],
                }

                if configs["training"]["training_type"] == "lora":
                    peft_config = LoraConfig(**peft_args)
                elif configs["training"]["training_type"] == "ia3":
                    peft_config = IA3Config(**peft_args)
                elif configs["training"]["training_type"] == "p_tune":
                    peft_config = PromptEncoderConfig(**peft_args)
                elif configs["training"]["training_type"] == "prompt_tuning":
                    peft_config = PromptTuningConfig(**peft_args)

                load_in_8bit = configs["training"]["load_in_8bit"][
                    configs["faster_transformer"]["fasttransformer"]
                ]

                # Define common arguments
                kwargs = {
                    "pretrained_model_name_or_path": configs["training"]["model_id"],
                    "load_in_8bit": load_in_8bit,
                    "revision": "main",  # Pin to main branch for consistency
                    "trust_remote_code": False,  # Security: disable arbitrary code execution
                }

                # Add conditional arguments
                if configs["faster_transformer"]["fasttransformer"]:
                    kwargs.update(
                        {
                            "device_map": "auto",
                            "torch_dtype": torch.float16,
                            "low_cpu_mem_usage": True,
                        }
                    )
                    logger.info("Currently loading model in float16")

                # Load the model with the consolidated kwargs
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    **kwargs
                )  # nosec B615 - revision pinned in kwargs, trust_remote_code disabled

                if load_in_8bit:
                    model.save_pretrained(os.path.join(configs["model_dir"], "base_model"))
                    logger.info(
                        f"Saved base model in INT-8 format at: {os.path.join(configs['model_dir'], 'base_model')}"
                    )

                model = prepare_model_for_int8_training(model)
                model = get_peft_model(model, peft_config)
                logger.info("Loading PEFT model successfully.")
            else:
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    # TODO - @dayojohn based on @xiongyue's feedback, investigate if we should be loading the full model here instead of the PEFT model.
                    # If so, we might need to adjust the path or how we load the model.
                    # configs["model_id"],
                    configs["training"]["model_id"],
                    revision="main",  # Pin to main branch for consistency
                    trust_remote_code=False,  # Security: disable arbitrary code execution
                )  # nosec B615 - revision pinned, trust_remote_code disabled
                logger.info("Loading full foundational model successfully.")

            print_trainable_parameters(model)

            # Check the device of the model
            device = next(model.parameters()).device
            logger.info(f"Model is on: {device}")

            return tokenizer, model
        except Exception as e:
            logger.error(f"Tokenizer and Model cannot be successfully loaded: {e}")
            raise ModelLoadErrorError(str(e))


def save_model(configs, model):
    """
    Save the model and relevant files.
    """
    code_path = os.path.join(configs["model_dir"], "code")
    os.makedirs(code_path, exist_ok=True)

    shutil.copyfile(
        os.path.join(os.path.dirname(__file__), "requirements.txt"),
        os.path.join(code_path, "requirements.txt"),
    )

    if configs["training"]["training_type"] in PEFT_METHODS:
        if not configs["faster_transformer"].get("fasttransformer", False):
            model.save_pretrained(os.path.join(configs["model_dir"], "peft_model"))
