import importlib
import logging

from finetune.utils.logging_util import get_logger
from finetune.utils.training_utils.training_constants import NO_DECAY

logger = get_logger(__name__)


class OptimizerLoadingError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return f"Optimizer Loading Error: {self.message}"


peft_check = importlib.util.find_spec("peft")
if peft_check:
    logging.info("loading torch and its components")
    import torch
    from transformers import AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup

    def setup_optimizer(
        configs: dict, model: AutoModelForSeq2SeqLM, train_dataloader: torch.utils.data.DataLoader
    ) -> tuple:
        """
        Set up optimizer and learning rate scheduler for training.

        Args:
            configs (dict): Configuration dictionary containing training parameters.
            model (AutoModelForSeq2SeqLM): The model to optimize.
            train_dataloader (torch.utils.data.DataLoader): The training dataloader for determining steps.

        Returns:
            tuple: Optimizer and learning rate scheduler.
        """
        try:
            # Extract training parameters from configs
            train_params = configs["training"].get("training_params", {})
            lr = train_params.get("lr", 5e-5)  # Default learning rate if not provided
            weight_decay = train_params.get("wt_decay", 0.01)
            num_epochs = train_params.get("num_epochs", 15)

            # Log the optimizer configuration
            logger.info(
                f"Setting up optimizer with lr={lr}, weight_decay={weight_decay}, num_epochs={num_epochs}"
            )

            # Parameter grouping for weight decay vs no weight decay
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in model.named_parameters()
                        if not any(nd in n for nd in NO_DECAY)
                    ],
                    "weight_decay": weight_decay,
                },
                {
                    "params": [
                        p for n, p in model.named_parameters() if any(nd in n for nd in NO_DECAY)
                    ],
                    "weight_decay": 0.0,
                },
            ]

            # Initialize optimizer
            optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)
            logger.info("Optimizer initialized successfully.")

            # Calculate the total number of training steps
            num_training_steps = len(train_dataloader) * num_epochs
            num_warmup_steps = int(0.10 * num_training_steps)  # 10% warmup steps
            logger.info(f"Training steps: {num_training_steps}, Warmup steps: {num_warmup_steps}")

            # Initialize the learning rate scheduler
            lr_scheduler = get_linear_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )
            logger.info("Learning rate scheduler initialized successfully.")

            return optimizer, lr_scheduler

        except KeyError as e:
            logger.error(f"Missing key in configs: {e}")
            raise

        except Exception as e:
            logger.error(f"Error setting up optimizer or scheduler: {e}")
            raise OptimizerLoadingError(str(e))
