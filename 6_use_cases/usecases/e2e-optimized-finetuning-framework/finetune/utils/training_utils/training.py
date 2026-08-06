import importlib
import logging
from time import time

from finetune.utils.logging_util import get_logger

logger = get_logger(__name__)


class OptimizerError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return f"Optimizer Error: {self.message}"


# run in Sagemaker
peft_check = importlib.util.find_spec("peft")
if peft_check:
    logging.info("loading torch and its components")
    import torch
    from tqdm import tqdm

    def train_epoch(
        configs: dict,
        model: torch.nn.Module,
        train_dataloader,
        optimizer,
        lr_scheduler,
        res: dict,
        epoch: int,
    ) -> dict:
        """
        Train the model for one epoch.

        Args:
            configs (dict): Configuration dictionary containing training parameters.
            model (torch.nn.Module): Model to be trained.
            train_dataloader (DataLoader): DataLoader for the training data.
            optimizer (torch.optim.Optimizer): Optimizer for model training.
            lr_scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
            res (dict): Dictionary to store training results.
            epoch (int): Current epoch number.

        Returns:
            dict: Dictionary containing training loss, perplexity, and time spent.
        """
        device = configs["training"]["device"]
        # use_fast_transformer = configs.get("faster_transformer", {}).get("fasttransformer", False)

        model.train()
        total_loss = 0
        start_time = time()

        for step, batch in enumerate(tqdm(train_dataloader)):
            try:
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items()}

                # Forward pass
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = outputs.loss  # loss from transformer

                total_loss += loss.detach().float()

                # Backward pass
                loss.backward()

                # Optimization step and learning rate scheduling
                optimizer_step(optimizer, lr_scheduler)

                # Zero gradients
                optimizer.zero_grad()

            except Exception as e:
                logger.error(f"Error during training at step {step}, skip this step: {e}")
                torch.cuda.empty_cache()
                continue

        end_time = time()

        # Compute final training loss and perplexity
        train_epoch_loss = total_loss / len(train_dataloader)
        train_ppl = torch.exp(train_epoch_loss)

        logger.info(
            f"Epoch {epoch}: Train Loss: {train_epoch_loss.item():.4f}, Perplexity: {train_ppl.item():.4f}, Time: {end_time - start_time:.2f}s"
        )

        return {
            "train_ppl": train_ppl.item(),
            "train_epoch_loss": train_epoch_loss.item(),
            "tr_time": (end_time - start_time),
        }

    def optimizer_step(
        optimizer: torch.optim.Optimizer, lr_scheduler: torch.optim.lr_scheduler._LRScheduler
    ) -> None:
        """
        Perform a single optimization step and update the learning rate scheduler.

        Args:
            optimizer (torch.optim.Optimizer): Optimizer used in training.
            lr_scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.

        Returns:
            None
        """
        try:
            optimizer.step()
            lr_scheduler.step()
        except Exception as e:
            logger.error(f"Error during optimizer or scheduler step: {e}")
            raise OptimizerError(str(e))
