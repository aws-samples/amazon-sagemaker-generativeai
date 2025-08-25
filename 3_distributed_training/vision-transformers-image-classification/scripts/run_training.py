from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
from datetime import datetime
from distutils.util import strtobool
import logging
import os
import re
from typing import Optional

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import torch
import torchvision
from torchvision.transforms import Normalize, Resize, ToTensor, Compose
from PIL import Image
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt

from transformers.trainer_utils import get_last_checkpoint
from transformers import ViTImageProcessor, ViTForImageClassification
from transformers import TrainingArguments, Trainer, set_seed
from trl import TrlParser, ModelConfig
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay



########################
# Custom dataclasses
########################
# @dataclass
# class ScriptArguments:
#     dataset_id_or_path: str
#     model_name_or_path: str
#     model_revision: str
#     torch_dtype: str 
#     attn_implementation : str
#     use_liger: str
#     bf16: str
#     tf32: str
#     output_dir: str
#     dataset_splits: str = "train"
@dataclass
class ScriptArguments:
    """
    Arguments for the script execution.
    """

    train_dataset_id_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the training dataset,"}
    )

    validation_dataset_id_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the training dataset,"}
    )
    # test_dataset_id_or_path: Optional[str] = field(
    #     default=None,
    #     metadata={"help": "Path to the training dataset,"}
    # )

    # model_name_or_path: Optional[str] = field(
    #     default=None,
    #     metadata={"help": "Model ID to use for SFT training"}
    # )

    # model_revision: str = field(
    #     default="main",
    #     metadata={"help": "revision"}
    # )
    # torch_dtype: str = field(
    #     default="bfloat16",
    #     metadata={"help": "torch_dtype"}
    # )

    # attn_implementation: str = field(
    #     default="false",
    #     metadata={"help": "attn_implementation"}
    # )
    # use_liger: str = field(
    #     default="false",
    #     metadata={"help": "Hugging Face API token"}
    # )

    # bf16: str = field(
    #     default="true",
    #     metadata={"help": "test"}
    # )
    # tf32: str = field(
    #     default="true",
    #     metadata={"help": "tf32"}
    # )
    # output_dir: str = field(
    #     default="",
    #     metadata={"help": "output directory"}
    # )
    dataset_splits: Optional[str] = field(
        default="train",
        metadata={"help": "dataset splits"}
    )
# @dataclass
# class ModelConfig:
#     model_name_or_path: str
#     model_revision: str
#     torch_dtype: str 
#     attn_implementation : str
#     use_liger: str
#     bf16: str
#     tf32: str
#     output_dir: str
########################
# Setup logging
########################
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
)
logger.addHandler(handler)

########################
# Helper functions
########################
def set_custom_env(env_vars: Dict[str, str]) -> None:
    """
    Set custom environment variables.

    Args:
        env_vars (Dict[str, str]): A dictionary of environment variables to set.
                                   Keys are variable names, values are their corresponding values.

    Returns:
        None

    Raises:
        TypeError: If env_vars is not a dictionary.
        ValueError: If any key or value in env_vars is not a string.
    """
    if not isinstance(env_vars, dict):
        raise TypeError("env_vars must be a dictionary")

    for key, value in env_vars.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError("All keys and values in env_vars must be strings")

    os.environ.update(env_vars)

    # Optionally, print the updated environment variables
    print("Updated environment variables:")
    for key, value in env_vars.items():
        print(f"  {key}: {value}")


def get_checkpoint(training_args: TrainingArguments):
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    return last_checkpoint



###########################################################################################################


def train_function(
    model_args: ModelConfig,
    script_args: ScriptArguments,
    training_args: TrainingArguments,
):
    """Main training function."""
    #########################
    # Log parameters
    #########################
    logger.info(f'Script parameters {script_args}')
    logger.info(f'Training/evaluation parameters {training_args}')

    ###############
    # Load datasets
    ###############
    if script_args.train_dataset_id_or_path.endswith('.parquet'):
        train_dataset = load_dataset(
            'parquet', data_files=script_args.train_dataset_id_or_path, split="train"
        )
        validation_dataset = load_dataset(
            'parquet', data_files=script_args.validation_dataset_id_or_path, split="train"
        )
        # test_dataset = load_dataset(
        #     'parquet', data_files=script_args.test_dataset_id_or_path, split="train"
        # )
    else:
        train_dataset, test_dataset = load_dataset(
            script_args.dataset_id_or_path, split=["train[:600000]","test[:100000]"]
        )

    #splits = train_dataset.train_test_split(test_size=0.1)
    trainds = train_dataset
    valds = validation_dataset
    
    logger.info(
        f'Loaded training dataset with {len(trainds)} samples'
    )
    logger.info(
        f'Loaded validation dataset with {len(valds)} samples'
    )
    # logger.info(
    #     f'Loaded dataset with {len(test_dataset)} samples and the following features: {train_dataset.features}'
    # )
    itos = dict((k,v) for k,v in enumerate(trainds.features['label'].names))
    stoi = dict((v,k) for k,v in enumerate(trainds.features['label'].names))

    img, lab = trainds[0]['image'], itos[trainds[0]['label']]
       #######################
    # Load pretrained model
    #######################

    # define model kwargs
    model_kwargs = dict(
        revision=model_args.model_revision,  # What revision from Huggingface to use, defaults to main
        trust_remote_code=model_args.trust_remote_code,  # Whether to trust the remote code, this also you to fine-tune custom architectures
        attn_implementation=model_args.attn_implementation,  # What attention implementation to use, defaults to flash_attention_2
        torch_dtype=(
            model_args.torch_dtype
            if model_args.torch_dtype in ['auto', None]
            else getattr(torch, model_args.torch_dtype)
        ),  # What torch dtype to use, defaults to auto
        # use_cache=(
        #     False if training_args.gradient_checkpointing else True
        # ),  # Whether
        low_cpu_mem_usage=(
            True
            if not strtobool(
                os.environ.get("ACCELERATE_USE_DEEPSPEED", "false")
            )
            else None
        ),  # Reduces memory usage on CPU for loading the model
    )


    # Check which training method to use and if 4-bit quantization is needed
    if model_args.load_in_4bit:
        model_kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=model_kwargs['torch_dtype'],
            bnb_4bit_quant_storage=model_kwargs['torch_dtype'],
        )
    processor = ViTImageProcessor.from_pretrained(model_args.model_name_or_path)
    
    mu, sigma = processor.image_mean, processor.image_std
    size = processor.size
    logger.info(size)

    norm = Normalize(mean=mu, std=sigma)

    _transf = Compose([
        Resize(size['height']),
        ToTensor(),
        norm
    ])

    def transf(arg):
        arg['pixels'] = [_transf(image.convert('RGB')) for image in arg['image']]
        return arg

    trainds.set_transform(transf)
    valds.set_transform(transf)
    #test_dataset.set_transform(transf)


    #model = ViTForImageClassification.from_pretrained(script_args.model_name_or_path)
    #logger.info(model.classifier) #The google/vit-base-patch16-224 model is originally fine tuned on imagenet-1K with 1000 output classes

    model = ViTForImageClassification.from_pretrained(model_args.model_name_or_path, num_labels=1001,  ignore_mismatched_sizes=True, id2label=itos, label2id=stoi)
    logger.info(model.classifier)
    logger.info(model.config)
    

    training_args.distributed_state.wait_for_everyone()  # wait for all processes to load

    def collate_fn(examples):
        pixels = torch.stack([example["pixels"] for example in examples])
        labels = torch.tensor([example["label"] for example in examples])
        return {"pixel_values": pixels, "labels": labels}

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return dict(accuracy=accuracy_score(predictions, labels))

    if model_args.use_peft:
        peft_config = get_peft_config(model_args)
    else:
        peft_config = None
    ########################
    # Initialize the Trainer
    ########################
    trainer = Trainer(
    model,
    training_args, 
    train_dataset=trainds,
    eval_dataset=valds,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    processing_class=processor,
    )
    
    if trainer.accelerator.is_main_process and peft_config:
        trainer.model.print_trainable_parameters()

    ###############
    # Training loop
    ###############
    # Check for last checkpoint
    last_checkpoint = get_checkpoint(training_args)
    if (
        last_checkpoint is not None
        and training_args.resume_from_checkpoint is None
    ):
        logger.info(
            f'Checkpoint detected, resuming training at {last_checkpoint}.'
        )

    logger.info(
        f'*** Starting training {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} for {training_args.num_train_epochs} epochs***'
    )
    
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    # log metrics
    metrics = train_result.metrics
    metrics['train_samples'] = len(trainds)
    trainer.log_metrics('train', metrics)
    trainer.save_metrics('train', metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################

    logger.info('*** Save model ***')
    if trainer.is_fsdp_enabled and peft_config:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type(
            'FULL_STATE_DICT'
        )
    # Restore k,v cache for fast inference
    trainer.model.config.use_cache = True
    trainer.save_model(training_args.output_dir)
    logger.info(f'Model saved to {training_args.output_dir}')
    training_args.distributed_state.wait_for_everyone()  # wait for all processes to load

    processor.save_pretrained(training_args.output_dir)
    logger.info(f'processor saved to {training_args.output_dir}')

    # Save everything else on main process
    if trainer.accelerator.is_main_process:
        trainer.create_model_card({'tags': ['sft', 'tutorial', 'amin']})
    # push to hub if needed
    if training_args.push_to_hub is True:
        logger.info('Pushing to hub...')
        trainer.push_to_hub()

    logger.info('*** Training complete! ***')


def main():
    parser = TrlParser((ModelConfig, ScriptArguments, TrainingArguments))
    model_args, script_args, training_args = parser.parse_args_and_config()

    # Set seed for reproducibility
    set_seed(training_args.seed)

    # Run the main training loop
    train_function(model_args, script_args, training_args)
    

if __name__ == '__main__':
    main()
