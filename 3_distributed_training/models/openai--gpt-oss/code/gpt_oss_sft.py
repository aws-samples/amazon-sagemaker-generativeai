from dataclasses import dataclass
from datetime import datetime
from distutils.util import strtobool
import logging
import os
import re
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, BitsAndBytesConfig, Mxfp4Config
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import is_liger_kernel_available
from trl import SFTTrainer, TrlParser, ModelConfig, SFTConfig, get_peft_config
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM


########################
# Custom dataclasses
########################
@dataclass
class ScriptArguments:
    dataset_id_or_path: str
    dataset_train_split: str = "train"
    dataset_test_split: str = "test"
    tokenizer_name_or_path: str = None
    max_seq_length: int = 1024


########################
# Setup logging
########################
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

########################
# Helper functions
########################

def get_checkpoint(training_args: SFTConfig):
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    return last_checkpoint


def merge_adapter_and_save_model(model_path_or_id, save_dir, save_tokenizer=True):
    model = AutoPeftModelForCausalLM.from_pretrained(
        model_path_or_id,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    )  
    # Merge LoRA and base model and save
    model = model.merge_and_unload()        
    model.save_pretrained(save_dir, safe_serialization=True, max_shard_size="4GB")

    # save tokenizer
    if save_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(model_path_or_id)
        tokenizer.save_pretrained(save_dir) 


###########################################################################################################

def train_function(model_args: ModelConfig, script_args: ScriptArguments, training_args: SFTConfig):
    """Main training function."""
    
    logger.info(f'Model parameters {model_args}')
    logger.info(f'Script parameters {script_args}')
    logger.info(f'Training/evaluation parameters {training_args}')

    if script_args.dataset_id_or_path.endswith('.jsonl'):
        dataset = load_dataset('json', data_files=script_args.dataset_id_or_path, split='train')
        train_dataset = dataset.select(range(900))
        test_dataset = dataset.select(range(900, 1000))
    else:
        if script_args.config is not None:
            train_dataset = load_dataset(script_args.dataset_id_or_path, script_args.config, split=script_args.dataset_train_split)
            test_dataset = load_dataset(script_args.dataset_id_or_path, script_args.config, split=script_args.dataset_test_split)
        else:
            train_dataset = load_dataset(script_args.dataset_id_or_path, split=script_args.dataset_train_split)
            test_dataset = load_dataset(script_args.dataset_id_or_path, split=script_args.dataset_test_split)
    
    logger.info(f'Loaded training dataset with {len(train_dataset)} samples and the following features: {train_dataset.features}')
    logger.info(f'Loaded test/eval dataset with {len(test_dataset)} samples and the following features: {test_dataset.features}')

    logger.info(train_dataset[0])
    
    tokenizer = AutoTokenizer.from_pretrained(
        script_args.tokenizer_name_or_path if script_args.tokenizer_name_or_path else model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token is None: 
        tokenizer.pad_token = tokenizer.eos_token
    
    
    #######################
    # Load pretrained model
    #######################
    
    # define model kwargs
    model_kwargs = dict(
        trust_remote_code=model_args.trust_remote_code, 
        attn_implementation=model_args.attn_implementation, 
        torch_dtype=model_args.torch_dtype if model_args.torch_dtype in ['auto', None] else getattr(torch, model_args.torch_dtype), 
        use_cache=False if training_args.gradient_checkpointing else True, 
        low_cpu_mem_usage=True if not strtobool(os.environ.get("ACCELERATE_USE_DEEPSPEED", "false")) else None,
    )
    
    # Check which training method to use and if 4-bit quantization is needed
    if model_args.load_in_4bit: 
        model_kwargs['quantization_config'] = Mxfp4Config(dequantize=True)
    
    if model_args.use_peft:
        logger.info("\n=======================\nfine-tuning using peft\n=======================\n")
        peft_config = get_peft_config(model_args)
    else:
        logger.info("\n=======================\nfull fine-tuning\n=======================\n")
        peft_config = None
    
    # load the model with our kwargs
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    
    training_args.distributed_state.wait_for_everyone()

    ########################
    # Initialize the Trainer
    ########################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    try:
        print(f"Training info:\n{trainer.model.print_trainable_parameters()}")
    except Exception as e:
        print(f"Cant print trainable parameters skipping!")

    ###############
    # Training loop
    ###############
    # Check for last checkpoint
    last_checkpoint = get_checkpoint(training_args)
    logger.info(f"Found checkpoints > {last_checkpoint}")
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f'Checkpoint detected, resuming training at {last_checkpoint}.')

    logger.info(f'*** Starting training {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} for {training_args.num_train_epochs} epochs***')
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    # log metrics
    metrics = train_result.metrics
    metrics['train_samples'] = len(train_dataset)
    trainer.log_metrics('train', metrics)
    trainer.save_metrics('train', metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    
    logger.info('*** Save model ***')
    if trainer.is_fsdp_enabled and peft_config:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type('FULL_STATE_DICT')
    
    # Restore k,v cache for fast inference
    trainer.model.config.use_cache = True

    #######################
    # Peft Model Save Path
    #######################
    if model_args.use_peft:
        logger.info("saving peft model")

        # save model and tokenizer
        last_checkpoint_output_dir = os.path.join(training_args.output_dir, "last_checkpoint")

        trainer.save_model(last_checkpoint_output_dir)
        logger.info(f'Model saved to {last_checkpoint_output_dir}')
        
        # wait for all processes to load
        training_args.distributed_state.wait_for_everyone() 

        tokenizer.save_pretrained(last_checkpoint_output_dir)
        logger.info(f'Tokenizer saved to {last_checkpoint_output_dir}')

        # Save everything else on main process
        if trainer.accelerator.is_main_process:
            trainer.create_model_card({'tags': ['sagemaker', 'gpt-oss']})

        del model 
        torch.cuda.empty_cache()

        # save final model to model output dir
        model_save_dir = os.path.join(os.environ["SM_MODEL_DIR"], model_args.model_name_or_path) if os.environ["SM_MODEL_DIR"] else os.path.join("/opt/ml/model", model_args.model_name_or_path)
        logger.info(f'saving final model to {model_save_dir}')
        
        # merge and save
        merge_adapter_and_save_model(
            model_path_or_id=last_checkpoint_output_dir, 
            save_dir=model_save_dir, 
            save_tokenizer=True
        )
    
    #######################
    # Full model save path
    #######################
    else:
        logger.info("saving full model")

        # save final model to model output dir
        model_save_dir = os.path.join(os.environ["SM_MODEL_DIR"], model_args.model_name_or_path) if os.environ["SM_MODEL_DIR"] else os.path.join("/opt/ml/model", model_args.model_name_or_path)
        logger.info(f'saving final model to {model_save_dir}')

        trainer.save_model(model_save_dir)
        logger.info(f'Model saved to {model_save_dir}')
        
        # wait for all processes to load
        training_args.distributed_state.wait_for_everyone() 

        tokenizer.save_pretrained(last_checkpoint_output_dir)
        logger.info(f'Tokenizer saved to {last_checkpoint_output_dir}')

    # Save everything else on main process
    if trainer.accelerator.is_main_process:
        trainer.create_model_card({'tags': ['sagemaker', 'gpt-oss']})

    logger.info('Training complete!')


def main():
    parser = TrlParser((ModelConfig, ScriptArguments, SFTConfig))
    model_args, script_args, training_args = parser.parse_args_and_config()

    print("model_args", model_args)
    print("script_args", script_args)
    print("training_args", training_args)

    # Set seed for reproducibility
    set_seed(training_args.seed)

    # Run the main training loop
    train_function(model_args, script_args, training_args)


if __name__ == '__main__':
    main()