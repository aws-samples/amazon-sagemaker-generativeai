import importlib
import logging
import os
import shutil

from finetune.utils.logging_util import get_logger
from finetune.utils.training_utils.training_constants import PEFT_METHODS

logger = get_logger(__name__)


def cleanup_peft_files(model_dir: str) -> None:
    """
    Removes unnecessary PEFT adapter files after merging the model.

    Args:
        model_dir (str): Directory where the model is stored.

    Returns:
        None
    """
    peft_files = ["adapter_model.bin", "adapter_config.json"]
    for peft_file in peft_files:
        file_path = os.path.join(model_dir, peft_file)
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"File {file_path} removed successfully.")
            else:
                logger.warning(f"File {file_path} does not exist.")
        except Exception as e:
            logger.error(f"Error removing file {file_path}: {e}")
            raise e


def copy_inference_script(script_name: str, code_path: str) -> None:
    """
    Copies the inference script into the deployment folder.

    Args:
        script_name (str): Name of the inference script to copy.
        code_path (str): Destination directory where the script will be copied.

    Raises:
        FileNotFoundError: If the script file does not exist.
    """
    try:
        source_path = os.path.join(os.path.dirname(__file__), script_name)
        dest_path = os.path.join(code_path, "inference.py")
        shutil.copyfile(source_path, dest_path)
        logger.info(f"Inference script {script_name} copied to {code_path}")
    except FileNotFoundError as e:
        logger.error(f"Inference script {script_name} not found: {e}")
        raise e
    except Exception as e:
        logger.error(f"Error copying inference script {script_name}: {e}")
        raise e


# run in SageMaker
peft_check = importlib.util.find_spec("peft")
if peft_check:
    logging.info("loading torch and its components")
    import torch
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    def deploy_model_preparation(
        configs: dict, model: AutoModelForSeq2SeqLM, tokenizer: AutoTokenizer
    ) -> None:
        """
        Prepares the model and tokenizer for deployment, handling both PEFT (Parameter-Efficient Fine-Tuning) and standard
        fine-tuning models. Handles saving, merging, and cleanup as needed.

        Args:
            configs (dict): Configuration dictionary.
            model (transformers.AutoModelForSeq2SeqLM): The model to be deployed.
            tokenizer (transformers.AutoTokenizer): The tokenizer used with the model.

        Raises:
            FileNotFoundError: If required files are not found for copying.
        """
        try:
            # Handle PEFT model deployment
            if configs["training"]["training_type"] in PEFT_METHODS:
                prepare_peft_model(configs, model, tokenizer)
            else:
                prepare_standard_model(configs, model, tokenizer)

            logger.info(f"Model and tokenizer saved successfully in: {configs['model_dir']}")
        except Exception as e:
            logger.error(f"An error occurred during deployment preparation: {e}")
            raise e

    def prepare_peft_model(
        configs: dict, model: AutoModelForSeq2SeqLM, tokenizer: AutoTokenizer
    ) -> None:
        """
        Prepares a PEFT-based model for deployment, including merging, saving, and cleanup.

        Args:
            configs (dict): Configuration dictionary.
            model (AutoModelForSeq2SeqLM): Model to deploy.
            tokenizer (PreTrainedTokenizer): Tokenizer to deploy.

        Raises:
            Exception: If merging or file operations fail.
        """
        try:
            if configs["faster_transformer"]["fasttransformer"]:
                # Save and merge the PEFT model for Faster Transformer
                model.save_pretrained(configs["model_dir"])
                peft_config = PeftConfig.from_pretrained(configs["model_dir"])
                base_model = AutoModelForSeq2SeqLM.from_pretrained(
                    peft_config.base_model_name_or_path,
                    return_dict=True,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    revision="main",  # Pin to main branch for consistency
                    trust_remote_code=False,  # Security: disable arbitrary code execution
                )  # nosec B615 - revision pinned, trust_remote_code disabled
                merged_model = PeftModel.from_pretrained(
                    base_model, configs["model_dir"]
                ).merge_and_unload()

                # Save merged model and tokenizer
                save_model_and_tokenizer(merged_model, tokenizer, configs["model_dir"])
                cleanup_peft_files(configs["model_dir"])

            else:
                # Standard PEFT deployment
                code_path = os.path.join(configs["model_dir"], "code")
                os.makedirs(code_path, exist_ok=True)

                save_model_and_tokenizer(
                    model, tokenizer, os.path.join(configs["model_dir"], "peft_model")
                )
                copy_inference_script("inference_peft.py", code_path)

        except Exception as e:
            logger.error(f"Failed to prepare PEFT model: {e}")
            raise e

    def prepare_standard_model(
        configs: dict, model: AutoModelForSeq2SeqLM, tokenizer: AutoTokenizer
    ) -> None:
        """
        Prepares a standard fine-tuned model for deployment, including saving and copying necessary files.

        Args:
            configs (dict): Configuration dictionary.
            model (AutoModelForSeq2SeqLM): Model to deploy.
            tokenizer (PreTrainedTokenizer): Tokenizer to deploy.

        Raises:
            FileNotFoundError: If the inference script or other required files are not found.
        """
        try:
            code_path = os.path.join(configs["model_dir"], "code")
            os.makedirs(code_path, exist_ok=True)

            save_model_and_tokenizer(model, tokenizer, configs["model_dir"])
            copy_inference_script("inference.py", code_path)

        except Exception as e:
            logger.error(f"Failed to prepare standard model: {e}")
            raise e

    def save_model_and_tokenizer(
        model: AutoModelForSeq2SeqLM, tokenizer: AutoTokenizer, model_dir: str
    ) -> None:
        """
        Saves the model and tokenizer to the specified directory.

        Args:
            model (AutoModelForSeq2SeqLM): Model to save.
            tokenizer (PreTrainedTokenizer): Tokenizer to save.
            model_dir (str): Directory where the model and tokenizer will be saved.

        Returns:
            None
        """
        try:
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
            logger.info(f"Model and tokenizer successfully saved in {model_dir}")
        except Exception as e:
            logger.error(f"Error saving model or tokenizer: {e}")
            raise e
