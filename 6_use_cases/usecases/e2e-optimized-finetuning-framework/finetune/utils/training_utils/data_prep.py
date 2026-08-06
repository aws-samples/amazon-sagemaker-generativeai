import importlib
import logging
import os
from functools import partial
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from finetune.utils.logging_util import get_logger

logger = get_logger(__name__)


class DatasetLoadErrorError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return f"Dataset Load Error: {self.message}"


# run in SageMaker
peft_check = importlib.util.find_spec("peft")
if peft_check:
    logging.info("loading torch and its components")
    from datasets import Dataset
    from torch.utils.data import DataLoader
    from transformers import DataCollatorForSeq2Seq, PreTrainedTokenizerBase

    class InputDataset:
        """
        A class to handle dataset loading, processing, and subsetting for model training.
        """

        def __init__(
            self,
            path: str,
            tokenizer: PreTrainedTokenizerBase,
            subset_frac: Optional[float] = None,
            prefix: str = "",
            max_ip_len: Optional[int] = None,
            max_target_len: Optional[int] = None,
        ):
            """
            Initialize the InputDataset with dataset path and tokenizer for processing.

            Args:
                path (str): Path to the dataset (JSON format expected).
                tokenizer (PreTrainedTokenizerBase): Tokenizer to process text into tokens.
                subset_frac (Optional[float]): Fraction of the dataset to sample.
                prefix (str): Prefix to append to each input sample.
                max_ip_len (Optional[int]): Maximum input sequence length for tokenization.
                max_target_len (Optional[int]): Maximum target sequence length for tokenization.
            """
            self.path = path
            self.tokenizer = tokenizer
            self.subset_frac = subset_frac
            self.prefix = prefix
            self.max_ip_len = max_ip_len
            self.max_target_len = max_target_len

            # Load dataset and apply subsetting
            self.data_df = self._load_data()
            self._subset_data()

            # Convert DataFrame to HuggingFace Dataset
            self.dataset = Dataset.from_pandas(self.data_df)
            logger.info(f"Dataset loaded from {self.path}. Rows: {len(self.data_df)}")

            # Apply data processing
            self.model_dataset = self._process_dataset()

        def _load_data(self) -> pd.DataFrame:
            """
            Load data from the provided JSON path.

            Returns:
                pd.DataFrame: Loaded dataset as a pandas DataFrame.
            """
            try:
                data_df = pd.read_json(self.path)
                logger.info(f"Successfully loaded data from {self.path}")
                return data_df
            except ValueError as e:
                err_message = f"Failed to load JSON data from {self.path}: {e}"
                logger.error(err_message)
                raise ValueError(err_message)

        def _subset_data(self) -> None:
            """
            Apply subset sampling if `subset_frac` is provided.
            """
            if self.subset_frac:
                original_size = len(self.data_df)
                self.data_df = self.data_df.sample(frac=self.subset_frac).reset_index(drop=True)
                logger.info(
                    f"Subset applied. Reduced dataset from {original_size} to {len(self.data_df)} rows."
                )

        def _process_dataset(self) -> Dataset:
            """
            Process the dataset by tokenizing input and target columns.

            Returns:
                Dataset: Tokenized HuggingFace dataset ready for model consumption.
            """
            try:
                processed_dataset = self.dataset.map(
                    partial(
                        self.process_sample,
                        prefix=self.prefix,
                        max_ip_len=self.max_ip_len,
                        max_op_len=self.max_target_len,
                        tokenizer=self.tokenizer,
                    ),
                    batched=True,
                    remove_columns=self.data_df.columns.tolist(),
                )
                logger.info("Dataset post-processing completed. Model-ready dataset created.")
                return processed_dataset
            except Exception as e:
                logger.error(f"Error in dataset processing: {e}")
                raise

        def get_length_percentile(self, percentile: float) -> Tuple[float, float]:
            """
            Calculate the token length percentile for both inputs and labels.

            Args:
                percentile (float): Percentile value to calculate.

            Returns:
                Tuple[float, float]: Percentile values for input and output text lengths.
            """
            try:
                tokenized_ip_len_arr = [len(e) for e in self.model_dataset["input_ids"]]
                tokenized_target_len_arr = [len(e) for e in self.model_dataset["labels"]]

                input_percentile = np.percentile(tokenized_ip_len_arr, percentile)
                output_percentile = np.percentile(tokenized_target_len_arr, percentile)

                logger.info(f"{percentile}th percentile of input token length: {input_percentile}")
                logger.info(
                    f"{percentile}th percentile of target token length: {output_percentile}"
                )

                return input_percentile, output_percentile
            except KeyError as e:
                logger.error(f"Error calculating length percentiles: {e}")
                raise

        @staticmethod
        def process_sample(
            sample,
            prefix: str,
            max_ip_len: int,
            max_op_len: int,
            tokenizer: PreTrainedTokenizerBase,
        ) -> Dict[str, Union[list, int]]:
            """
            Process and tokenize a sample row.

            Args:
                sample (dict): A row of the dataset.
                prefix (str): Prefix to prepend to input text.
                max_ip_len (int): Maximum token length for inputs.
                max_op_len (int): Maximum token length for outputs.
                tokenizer (PreTrainedTokenizerBase): Tokenizer for text tokenization.

            Returns:
                dict: Tokenized inputs and labels.
            """
            try:
                inputs = [prefix + item for item in sample["Input_Info"]]
                model_inputs = tokenizer(inputs, max_length=max_ip_len, truncation=True)
                labels = tokenizer(
                    text_target=sample["Target_Summary"], max_length=max_op_len, truncation=True
                )
                model_inputs["labels"] = labels["input_ids"]
                return model_inputs
            except KeyError as e:
                logger.error(f"Error processing sample: Missing keys {e}")
                raise

    # Function to load datasets for training and evaluation
    def load_datasets(
        configs: dict, tokenizer: PreTrainedTokenizerBase
    ) -> Tuple[InputDataset, DataLoader, DataLoader]:
        """
        Load training and evaluation datasets, and return their DataLoader objects.

        Args:
            configs (dict): Configuration dictionary.
            tokenizer (PreTrainedTokenizerBase): Tokenizer for the text.

        Returns:
            Tuple[InputDataset, DataLoader, DataLoader]: Evaluation dataset and dataloaders for training and evaluation datasets.
        """
        data_params = configs["training"]["data_params"]
        task_prefix = configs["training"]["task_prefix"]

        try:
            # Load training dataset
            train_dataset = InputDataset(
                path=os.path.join(configs["data_path"], "train.json"),
                tokenizer=tokenizer,
                subset_frac=data_params.get("train_data_frac", None),
                prefix=task_prefix,
                max_ip_len=data_params.get("max_ip_len", 512),
                max_target_len=data_params.get("max_op_len", 128),
            )

            # Load evaluation dataset
            eval_dataset = InputDataset(
                path=os.path.join(configs["data_path"], "eval.json"),
                tokenizer=tokenizer,
                prefix=task_prefix,
                max_ip_len=data_params.get("max_ip_len", 512),
                max_target_len=data_params.get("max_op_len", 128),
            )

            # Create data collator for Seq2Seq
            data_collator = DataCollatorForSeq2Seq(
                tokenizer=tokenizer,
                label_pad_token_id=configs["training"]["label_pad_token_id"],
                padding="longest",
            )

            # Create DataLoader for training data
            train_dataloader = DataLoader(
                train_dataset.model_dataset,
                shuffle=True,
                collate_fn=data_collator,
                batch_size=data_params["tr_batch_size"],
                pin_memory=True,
            )

            # Create DataLoader for evaluation data
            eval_dataloader = DataLoader(
                eval_dataset.model_dataset,
                collate_fn=data_collator,
                batch_size=data_params["eval_batch_size"],
                pin_memory=True,
            )

            # Check for training data
            for batch in train_dataloader:
                for key, value in batch.items():
                    logger.info(f"Train batch '{key}' is on device: {value.device}")
                break  # Check just the first batch

            # Check for evaluation data
            for batch in eval_dataloader:
                for key, value in batch.items():
                    logger.info(f"Eval batch '{key}' is on device: {value.device}")
                break  # Check just the first batch

            return eval_dataset, train_dataloader, eval_dataloader
        except Exception as e:
            logger.error(f"Training and Evaluation dataset cannot be loaded properly: {e}")
            raise DatasetLoadErrorError(str(e))
