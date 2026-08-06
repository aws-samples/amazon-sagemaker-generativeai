# this training script is used for specific purpose of using FastTransformer for fast inference time. It uses all GPUs of training instance to train the model to accomodate the large model size. The model will be loaded in fp16 format to match the requirement of FastTransformer. The final model will be saved in merged form, which means the modified weights of the base model. No inference.py file will be saved and the output format is a folder that exactly the same as required by Fast Transformer saved in s3 bucket.

import importlib
import logging
import os

import nltk

peft_check = importlib.util.find_spec("peft")
if peft_check:
    logging.info("loading torch and its components")
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

nltk.download("punkt")

from finetune.constants import TOKENIZERS_PARALLELISM
from finetune.utils import import_yaml_files
from finetune.utils.training_utils import (
    InputDataset,
    deploy_model_preparation,
    evaluate_model,
    load_datasets,
    load_tokenizer_and_model,
    save_results,
    setup_optimizer,
    setup_paths,
    train_epoch,
)

# Initialize logger
from finetune.utils.logging_util import get_logger

logger = get_logger(__name__)


class SagemakerEnvironmentError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return f"SageMaker Environment Setup Error: {self.message}"


def train_model():
    """
    Main function to train a machine learning model, including:
    - Loading configurations, tokenizer, and model.
    - Setting up data loaders, optimizer, and learning rate scheduler.
    - Executing the training and evaluation loops for a specified number of epochs.
    - Saving the model and results.

    Raises:
        Exception: If any part of the process fails, it is caught and logged.
    """
    try:
        # Load configurations
        configs = import_yaml_files()
        logger.info("Starting the training process.")
        configs = setup_paths(configs)

        # Set up environment variables
        os.environ["TOKENIZERS_PARALLELISM"] = TOKENIZERS_PARALLELISM
        logger.debug(f"Set TOKENIZERS_PARALLELISM: {TOKENIZERS_PARALLELISM}")

        setup_sagemaker_environment(configs)

        # Load tokenizer and model
        tokenizer, model = load_tokenizer_and_model(configs)

        # Load datasets
        eval_dataset, train_dataloader, eval_dataloader = load_datasets(configs, tokenizer)

        # Setup optimizer and scheduler
        optimizer, lr_scheduler = setup_optimizer(configs, model, train_dataloader)

        # Training and evaluation loops
        res = run_training_loop(
            configs,
            model,
            train_dataloader,
            eval_dataset,
            eval_dataloader,
            optimizer,
            lr_scheduler,
            tokenizer,
        )

        # Save results and prepare the model for deployment
        save_results(configs, res, "final_results")
        deploy_model_preparation(configs, model, tokenizer)

        logger.info("Training process completed successfully.")

    except Exception as e:
        logger.error(f"Error in train_model: {e}", exc_info=True)
        raise e


def setup_sagemaker_environment(configs: dict) -> None:
    """
    Sets up the SageMaker environment by defining the necessary environment variables.

    Args:
        configs (dict): Configuration dictionary.

    Returns:
        None
    """
    try:
        sm_module_dir = os.path.join(
            "s3://", configs["aws_setup"]["default_bucket"], configs["data"]["s3_root_folder"]
        )
        os.environ["SM_MODULE_DIR"] = sm_module_dir
        logger.debug(f"Set SM_MODULE_DIR: {sm_module_dir}")
    except KeyError as e:
        err_message = f"KeyError: Missing AWS configuration in 'configs': {e}"
        logger.error(err_message)
        raise KeyError(err_message)
    except Exception as e:
        err_message = f"Error setting SageMaker environment: {e}"
        logger.error(err_message)
        raise SagemakerEnvironmentError(e)


def run_training_loop(
    configs: dict,
    model: torch.nn.Module,
    train_dataloader: DataLoader,
    eval_dataset: InputDataset,
    eval_dataloader: DataLoader,
    optimizer,
    lr_scheduler,
    tokenizer: AutoTokenizer,
) -> dict:
    """
    Runs the training and evaluation loop for the specified number of epochs.

    Args:
        configs (dict): Configuration dictionary.
        model (torch.nn.Module): The model to be trained.
        train_dataloader (DataLoader): Dataloader for training data.
        eval_dataset (InputDataset): Evaluation dataset.
        eval_dataloader (DataLoader): Dataloader for evaluation data.
        optimizer: Optimizer for training.
        lr_scheduler: Learning rate scheduler for training.
        tokenizer (AutoTokenizer): Tokenizer used for processing text inputs.

    Returns:
        dict: Dictionary containing the training and evaluation metrics.
    """
    res = {}
    train_params = configs["training"].get("training_params", {})
    data_params = configs["training"].get("data_params", {})

    # model_name = configs["training"]["model_id"].split("/")[-1].strip()
    tr_data_frac = int(data_params["train_data_frac"] * 100)

    for epoch in range(train_params["num_epochs"]):
        logger.info(f"Starting Epoch {epoch + 1}/{train_params['num_epochs']}")

        # Start training for one epoch
        tr_res = train_epoch(configs, model, train_dataloader, optimizer, lr_scheduler, res, epoch)

        # Evaluate the model
        out_name = f"tf_{tr_data_frac}-ne_{epoch + 1}"
        eval_res = evaluate_model(
            configs, model, eval_dataset, eval_dataloader, tokenizer, epoch, out_name
        )

        # Log and save the results
        res[epoch] = eval_res["metrics"]
        res[epoch].update(
            {
                "train_ppl": tr_res["train_ppl"],
                "train_epoch_loss": tr_res["train_epoch_loss"],
                "eval_ppl": eval_res["eval_ppl"],
                "eval_epoch_loss": eval_res["eval_epoch_loss"],
                "tr_time": tr_res["tr_time"],
                "eval_time": eval_res["eval_time"],
            }
        )

        logger.info(
            f"Epoch {epoch + 1} Results: "
            f"Training_PPL: {tr_res['train_ppl']:.4f}, "
            f"Training_Loss: {tr_res['train_epoch_loss']:.4f}, "
            f"Eval_PPL: {eval_res['eval_ppl']:.4f}, "
            f"Eval_Loss: {eval_res['eval_epoch_loss']:.4f}, "
            f"Metrics: {eval_res['metrics']}"
        )

    return res


if __name__ == "__main__":
    train_model()
