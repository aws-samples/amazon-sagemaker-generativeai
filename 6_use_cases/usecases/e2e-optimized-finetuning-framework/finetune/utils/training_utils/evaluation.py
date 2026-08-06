import importlib
import logging
import os
from time import time
from typing import List, Tuple

import numpy as np
import pandas as pd
import yaml
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

from finetune.utils.logging_util import get_logger
from finetune.utils.training_utils.data_prep import InputDataset

logger = get_logger(__name__)


class PostprocessingError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return f"Text Postprocessing Error: {self.message}"


class MetricsComputeError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return f"Metrics Computation Error: {self.message}"


class EvaluateError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return f"Evaluation Error: {self.message}"


def postprocess_text(preds: List[str], labels: List[str]) -> Tuple[List[str], List[str]]:
    """
    Strips whitespace and tokenizes text by sentences, as expected by certain evaluation metrics like ROUGE-L.

    Args:
        preds (list[str]): List of predicted texts.
        labels (list[str]): List of reference texts (ground truth).

    Returns:
        tuple[list[str], list[str]]: The post-processed predictions and labels.
    """
    try:
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # Tokenize each sentence and join by newline
        # rougeLSum expects newline after each sentence
        preds = ["\n".join(sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(sent_tokenize(label)) for label in labels]

        return preds, labels
    except Exception as e:
        logger.error(e)
        raise PostprocessingError(str(e))


def save_results(configs: dict, res: dict, out_name: str) -> None:
    """
    Save the results and experiment configuration to CSV and YAML files.

    Args:
        configs (dict): Experiment configuration dictionary.
        res (dict): Dictionary containing the experiment results.
        out_name (str): Output file name prefix.

    Returns:
        None
    """
    result_dir = configs.get("result_dir", "")
    if not result_dir:
        logger.error("No result directory specified in configuration.")
        return

    # Ensure the result directory exists
    os.makedirs(result_dir, exist_ok=True)

    # Save metrics to CSV
    try:
        res_df = pd.DataFrame(res)
        csv_path = os.path.join(result_dir, f"{out_name}_genr_metrics.csv")
        res_df.to_csv(csv_path, index=False)
        logger.info(f"Generation metrics successfully saved at {csv_path}")
    except Exception as e:
        logger.error(f"Error saving generation metrics to CSV: {e}")
        return

    # Save config to YAML
    try:
        yaml_path = os.path.join(result_dir, f"{out_name}_experiment_confg.yaml")
        with open(yaml_path, "w") as yamlfile:
            yaml.dump(configs, yamlfile)
        logger.info(f"Experiment configuration successfully saved at {yaml_path}")
    except Exception as e:
        logger.error(f"Error saving experiment configuration to YAML: {e}")


# run in SageMaker
peft_check = importlib.util.find_spec("peft")
if peft_check:
    logging.info("loading torch and its components")
    import evaluate
    import torch
    from transformers import AutoModelForSeq2SeqLM, PreTrainedTokenizerBase

    def compute_metrics(
        eval_preds: Tuple[List[str], List[str]], metric_type: str = "lexical"
    ) -> dict:
        """
        Compute lexical and semantic metrics for model evaluation.

        Args:
            eval_preds (tuple[list[str], list[str]]): Tuple of predicted and reference texts.
            metric_type (str, optional): Type of metrics to compute, either "lexical" or "semantic".


        Returns:
            dict: Dictionary containing computed metrics.
        """
        try:
            rouge_metric = evaluate.load(os.path.join(os.path.dirname(__file__), "rouge/rouge.py"))
            bleu_metric = evaluate.load(
                os.path.join(os.path.dirname(__file__), "google_bleu/google_bleu.py")
            )

            decoded_preds, decoded_labels = eval_preds
            decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

            results_bleu = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels)
            results_rouge = rouge_metric.compute(
                predictions=decoded_preds, references=decoded_labels, use_stemmer=True
            )

            result = {
                k: round(v * 100, 2)
                for res in [results_bleu, results_rouge]
                for k, v in res.items()
            }

            if metric_type == "semantic":
                # bert_metric = evaluate.load("bertscore")
                bert_metric = evaluate.load(
                    os.path.join(os.path.dirname(__file__), "bertscore/bertscore.py")
                )
                results_bert = bert_metric.compute(
                    predictions=decoded_preds,
                    references=decoded_labels,
                    model_type="distilbert-base-uncased",
                    rescale_with_baseline=True,
                    lang="en",
                )
                results_bert = {
                    k: np.round(np.mean(v), 3) for k, v in results_bert.items() if k != "hashcode"
                }
                result.update({k: round(v * 100, 4) for k, v in results_bert.items()})

            prediction_lens = [len(decoded_pred.split()) for decoded_pred in decoded_preds]
            result["gen_len"] = np.mean(prediction_lens)

            return result
        except NameError as e:
            raise NameError(str(e))
        except Exception as e:
            logger.error(e)
            raise MetricsComputeError(str(e))

    def evaluate_model(
        configs: dict,
        model: AutoModelForSeq2SeqLM,
        eval_dataset: InputDataset,
        eval_dataloader,
        tokenizer: PreTrainedTokenizerBase,
        epoch: int,
        out_name: str,
    ) -> dict:
        """
        Evaluate the model on a given evaluation set and compute metrics.

        Args:
            config (dict): Configuration dictionary.
            model (AutoModelForSeq2SeqLM): Pre-trained model to evaluate.
            eval_dataset (InputDataset): Dataset for evaluation.
            eval_dataloader: DataLoader for evaluation data.
            tokenizer (PreTrainedTokenizerBase): Tokenizer used for encoding/decoding.
            epoch (int): Current epoch number (for logging purposes).
            out_name (str): Name used for output file storage.

        Returns:
            dict: Dictionary containing evaluation results (e.g., perplexity, loss, and metrics).
        """
        gnr_params = configs["generation"]
        try:
            torch.cuda.empty_cache()
            model.eval()
            eval_loss = 0

            model_input = []
            eval_preds = []
            eval_reference = []

            start_time = time()
            for step, batch in enumerate(tqdm(eval_dataloader)):
                target = batch["labels"]
                target[target == -100] = tokenizer.pad_token_id
                torch.cuda.empty_cache()

                # batch = {k: v.to(configs['training']['device']) for k, v in batch.items()} if not configs['faster_transformer'].get('fasttransformer', True) else {k: v for k, v in batch.items()}
                batch = {k: v.to(configs["training"]["device"]) for k, v in batch.items()}
                with torch.no_grad():
                    outputs = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                    )
                    generation_ip = {**batch, **gnr_params}
                    model_generations = model.generate(**generation_ip)

                loss = outputs.loss
                eval_loss += loss.detach().float()

                model_generations = tokenizer.batch_decode(
                    model_generations, skip_special_tokens=True
                )
                model_generations = np.array(
                    [
                        model_generations[i : (i + gnr_params["num_return_sequences"])]
                        for i in range(
                            0, len(model_generations), gnr_params["num_return_sequences"]
                        )
                    ]
                )
                eval_preds.append(model_generations)

                eval_reference.extend(tokenizer.batch_decode(target, skip_special_tokens=True))
                model_input.extend(
                    tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
                )

            torch.cuda.empty_cache()
            end_time = time()

            # Save predictions
            eval_preds = np.concatenate(eval_preds)
            eval_dataset.data_df["model_input"] = model_input
            eval_dataset.data_df["model_target"] = eval_reference

            for i in range(gnr_params["num_return_sequences"]):
                eval_dataset.data_df[f"model_generation-{i}"] = eval_preds[:, i].tolist()

            eval_dataset.data_df.to_csv(
                os.path.join(configs["result_dir"], f"{out_name}_model_gen.csv"), index=False
            )

            # Compute evaluation metrics
            metrics = {
                k: 0
                for k in compute_metrics(
                    (eval_preds[:, 0].tolist(), eval_reference), configs["training"]["metric_type"]
                ).keys()
            }
            for i in range(eval_preds.shape[-1]):
                temp_metric = compute_metrics(
                    (eval_preds[:, i].tolist(), eval_reference), configs["training"]["metric_type"]
                )
                for k in metrics.keys():
                    metrics[k] += temp_metric[k]

            metrics = {k: v / eval_preds.shape[-1] for k, v in metrics.items()}

            eval_epoch_loss = eval_loss / len(eval_dataloader)
            eval_ppl = torch.exp(eval_epoch_loss)
            logger.info(
                f"Epoch {epoch}: Perplexity={eval_ppl:.4f}, Loss={eval_epoch_loss:.4f}, Metrics={metrics}"
            )

            return {
                "eval_ppl": eval_ppl.item(),
                "eval_epoch_loss": eval_epoch_loss.item(),
                "eval_time": (end_time - start_time),
                "metrics": metrics,
            }
        except Exception as e:
            logger.error(f"Evaluation cannot be done: {e}")
            raise EvaluateError(str(e))
