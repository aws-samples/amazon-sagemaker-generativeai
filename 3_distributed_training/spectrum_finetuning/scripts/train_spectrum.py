import datetime
import os
import re
from dataclasses import dataclass, field
from typing import Dict, Optional
from itertools import chain
import logging
import torch
from accelerate import Accelerator
from datasets import load_dataset
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, set_seed
from peft import AutoPeftModelForCausalLM, LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, TrlParser
import mlflow


logger = logging.getLogger(__name__)
logging.basicConfig(
    encoding='utf-8',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


@dataclass
class ScriptArguments:
    """
    Arguments for the script execution.
    """

    # PEFT Configuration
    enable_lora: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable LoRA for training"}
    )

    enable_quantization: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable quantization"}
    )

    lora_r: Optional[int] = field(
        default=8,
        metadata={"help": "lora_r"}
    )

    lora_alpha: Optional[int] = field(
        default=16,
        metadata={"help": "lora_dropout"}
    )

    lora_dropout: Optional[float] = field(
        default=0.1,
        metadata={"help": "lora_dropout"}
    )

    merge_weights: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to merge adapter with base model"}
    )

    #MLflow Configuration
    mlflow_uri: Optional[str] = field(
        default=None,
        metadata={"help": "MLflow tracking ARN"}
    )

    mlflow_experiment_name: Optional[str] = field(
        default=None,
        metadata={"help": "MLflow experiment name"}
    )

    # Dataset Configuration
    train_dataset_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the training dataset"}
    )

    test_dataset_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the test dataset"}
    )

    # Spectrum Configuration
    enable_spectrum: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable Spectrum for training"}
    )

    spectrum_config_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the spectrum config"}
    )

    # Model and Model Access Configuration
    model_id: str = field(
        default=None,
        metadata={"help": "Model ID to use for SFT training."}
    )

    hf_token: str = field(
        default="",
        metadata={"help": "Hugging Face API token"}
    )


def init_distributed():
    # Initialize the process group
    torch.distributed.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=5400))  # Use "gloo" backend for CPU
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    return local_rank


def download_model(model_name):
    print("Downloading model ", model_name)

    os.makedirs("/tmp/tmp_folder", exist_ok=True)

    snapshot_download(repo_id=model_name, local_dir="/tmp/tmp_folder")

    print(f"Model {model_name} downloaded under /tmp/tmp_folder")


def group_texts(examples, block_size=2048):
    """
    Groups a list of tokenized text examples into fixed-size blocks for language model training.

    Args:
        examples (dict): A dictionary where keys are feature names (e.g., "input_ids") and values 
                         are lists of tokenized sequences.
        block_size (int, optional): The size of each chunk. Defaults to 2048.

    Returns:
        dict: A dictionary containing the grouped chunks for each feature. An additional "labels" key 
              is included, which is a copy of the "input_ids" key.
    """
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


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
    logger.info("Updated environment variables:")
    for key, value in env_vars.items():
        logger.info(f"  {key}: {value}")


def setup_model_for_spectrum(model, spectrum_config_path):
    """
    Configures a PyTorch model for Spectrum training by selectively freezing and unfreezing parameters
    based on a YAML configuration file.

    This function first freezes all model parameters and then selectively unfreezes
    parameters specified in the configuration file. The configuration file should contain
    a list of parameter patterns that should remain trainable.

    Parameters:
        model : torch.nn.Module
            The PyTorch model to be configured for training
        spectrum_config_path : str
            Path to the YAML configuration file containing the list of parameters to unfreeze.
            The file should list parameters with each line starting with "- " followed by
            the parameter pattern.

    Returns:
        torch.nn.Module
            The modified model with selective parameters frozen/unfrozen according to
            the configuration
    """

    
    unfrozen_parameters = []
    with open(spectrum_config_path, "r") as fin:
        yaml_parameters = fin.read()

    # get the unfrozen parameters from the yaml file
    for line in yaml_parameters.splitlines():
        if line.startswith("- "):
            unfrozen_parameters.append(line.split("- ")[1])

    # freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    # unfreeze Spectrum parameters
    for name, param in model.named_parameters():
        if any(
            re.match(unfrozen_param, name)
            for unfrozen_param in unfrozen_parameters
        ):
            param.requires_grad = True

    #Sanity check for trainable parameters
    trainable_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(name)
            
    logger.info(f"Trainable parameters: {trainable_params}")

    return model

def load_and_prepare_data(train_data_location=None, test_data_location=None, tokenizer=None):
    # Load datasets
    logger.info(f"Loading Datasets")
    train_ds = load_dataset(
        "json",
        data_files=os.path.join(train_data_location, "dataset.json"),
        split="train"
    )

    if test_data_location:
        test_ds = load_dataset(
            "json",
            data_files=os.path.join(test_data_location, "dataset.json"),
            split="train"
        )
    else:
        test_ds = None

    logger.info(f"Total number of train samples: {len(train_ds)}")
    logger.info(f"Total number of test samples: {len(test_ds)}")
    logger.info(f"Converting dataset to match chat template...")

    lm_train_dataset = train_ds.map(create_conversation, batched=False, remove_columns=list(train_ds.features), fn_kwargs={"tokenizer": tokenizer})

    if test_ds is not None:
        lm_test_dataset = test_ds.map(create_conversation, batched=False, remove_columns=list(test_ds.features), fn_kwargs={"tokenizer": tokenizer})
    else:
        lm_test_dataset = None

    logger.info(f"Dataset conversion complete.")

    return lm_train_dataset, lm_test_dataset

    
def create_conversation(sample, tokenizer=None):
    """
    Creates a formatted conversation structure for language model processing.

    This function takes a sample dictionary, and formats it into a standardized 
    conversation structure suitable for language model interaction.

    Args:
        sample : dict
            A dictionary containing the following keys:
            - context (str): Background information or context for the conversation
            - question (str): The specific question being asked
            - answers (str): The corresponding answers to the question

    Returns:
        dict
            A dictionary containing the processed conversation text with key:
            - text (str): The formatted conversation text processed through the tokenizer
    """

    # System message for the assistant 
    system_message = f"Answer the question based on the knowledge you have or shared by the user." 

    query = query = f"""
        -- context --
        {sample["context"]}
         -- question --
        {sample["question"]}
        -- answer --
     """

    messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": query},
            {"role": "assistant", "content": sample["answers"]}
        ]
    
    text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=True
    )

    return {"text": text}

def train(script_args, training_args):
    set_seed(training_args.seed)

    mlflow_enabled = (
        script_args.mlflow_uri is not None
        and script_args.mlflow_experiment_name is not None
        and script_args.mlflow_uri != ""
        and script_args.mlflow_experiment_name != ""
    )

    accelerator = Accelerator()

    if script_args.hf_token is not None:
        os.environ.update({"HF_TOKEN": script_args.hf_token})
        accelerator.wait_for_everyone()

    # Download model based on training setup (single or multi-node)
    if int(os.environ.get("SM_HOST_COUNT", 1)) == 1:
        if accelerator.is_main_process:
            download_model(script_args.model_id)
    else:
        download_model(script_args.model_id)

    accelerator.wait_for_everyone()

    script_args.model_id = "/tmp/tmp_folder"

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_id)

    # Set Tokenizer pad Token
    tokenizer.pad_token = tokenizer.eos_token

    # Load datasets and perform necessary training conversions
    lm_train_dataset, lm_test_dataset = load_and_prepare_data(
        train_data_location=script_args.train_dataset_path,
        test_data_location=script_args.test_dataset_path,
        tokenizer=tokenizer
    )

    if mlflow_enabled and accelerator.is_main_process:
        logger.info(f"MLflow tracking under: {script_args.mlflow_experiment_name}")
        mlflow.start_run(run_name=os.environ.get("MLFLOW_RUN_NAME", None))
        train_dataset_mlflow = mlflow.data.from_pandas(lm_train_dataset.to_pandas(), name="train_dataset")
        mlflow.log_input(train_dataset_mlflow, context="train")

        if lm_train_dataset is not None:
            test_dataset_mlflow = mlflow.data.from_pandas(lm_test_dataset.to_pandas(), name="test_dataset")
            mlflow.log_input(test_dataset_mlflow, context="test")

    accelerator.wait_for_everyone()

    if training_args.bf16:
        logger.info("Initializing flash_attention_2")
        torch_dtype = torch.bfloat16
        model_configs = {
            "attn_implementation": "flash_attention_2",
            "torch_dtype": torch_dtype,
        }
    else:
        torch_dtype = torch.float32
        model_configs = dict()

    if training_args.fsdp is not None and training_args.fsdp != "" and \
        training_args.fsdp_config is not None and len(training_args.fsdp_config) > 0:
        bnb_config_params = {
            "bnb_4bit_quant_storage": torch_dtype
        }

        training_args.gradient_checkpointing_kwargs = {
            "use_reentrant": False
        }
        
    else:
        bnb_config_params = dict()

    if script_args.enable_quantization:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            **bnb_config_params
        )

    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_id,
        trust_remote_code=True,
        quantization_config=bnb_config if script_args.enable_quantization else None,
        use_cache=not training_args.gradient_checkpointing,
        cache_dir="/tmp/.cache",
        **model_configs
    )

    if training_args.fsdp is None and training_args.fsdp_config is None:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

        if training_args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
    else:
        logger.info("Enabling FDSP")
        if training_args.gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    if script_args.enable_lora:
        logger.info("Enabling LoRA")
        peft_config = LoraConfig(
            r=script_args.lora_r,
            lora_alpha=script_args.lora_alpha,
            target_modules="all-linear",
            lora_dropout=script_args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
    else:
        peft_config = None
    
    if script_args.enable_spectrum:
        logger.info("Enabling Spectum")
        model = setup_model_for_spectrum(
            model, script_args.spectrum_config_path
        )

    logger.info("Disabling checkpointing and setting up logging")
    training_args.save_strategy="no"
    training_args.logging_strategy="steps"
    training_args.logging_steps=1
    training_args.log_on_each_node=False
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=lm_train_dataset,
        eval_dataset=lm_test_dataset if lm_test_dataset is not None else None,
        processing_class=tokenizer,
        peft_config=peft_config
    )

    accelerator.wait_for_everyone()

    logger.info("Beginning Training")
    trainer.train()
    logger.info("Completing Training")

    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")


    if script_args.enable_lora and script_args.merge_weights:
        logger.info("Merge weights is set to True, will fuse the adapter to the base model")
        
        tmp_output_dir = "/tmp/model/output"
    
        # merge adapter weights with base model and save
        # save int 4 model
        logger.info(f"Temporarily saving adapter to {tmp_output_dir}")
        trainer.model.config.use_cache = True
        trainer.save_model(tmp_output_dir)
    
        if accelerator.is_main_process:
            # clear memory
            del model
            del trainer
    
            torch.cuda.empty_cache()

            logger.info("Loading PEFT model")
            # load PEFT model
            model = AutoPeftModelForCausalLM.from_pretrained(
                tmp_output_dir,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )

            logger.info("Fusing adapter to base model")
            # Merge LoRA and base model and save
            model = model.merge_and_unload()
            logger.info("Saving fused model")
            model.save_pretrained(
                training_args.output_dir,
                safe_serialization=True
            )
    else:
        logger.info("Saving model")
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.save_model(training_args.output_dir)

    accelerator.wait_for_everyone()
    
    logger.info("Saving tokenizer")
    tokenizer.save_pretrained(training_args.output_dir)
    
    logger.info("TRAINING COMPLETED")

if __name__ == "__main__":
    
    #Initialize the distributed environment. This must happen first.
    local_rank = init_distributed()
    torch.distributed.barrier(device_ids=[local_rank])

    #Parse out the training and script arguments to pass to the trainer
    parser = TrlParser((ScriptArguments, TrainingArguments))
    script_args, training_args = parser.parse_args_and_config()

    # Set up custom environment variables
    custom_env: Dict[str, str] = {
        "HF_DATASETS_TRUST_REMOTE_CODE": "TRUE",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_TOKEN": script_args.hf_token,
    }
    set_custom_env(custom_env)

    # If MLflow configuration is supplied, set up MLflow logging and experiement tracking
    if script_args.mlflow_uri is not None and script_args.mlflow_experiment_name is not None and \
            script_args.mlflow_uri != "" and script_args.mlflow_experiment_name != "":

        logger.info("Initializing MLflow")
        mlflow.enable_system_metrics_logging()
        mlflow.autolog()
        mlflow.set_tracking_uri(script_args.mlflow_uri)
        mlflow.set_experiment(script_args.mlflow_experiment_name)

        current_datetime = datetime.datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d-%H-%M")
        set_custom_env({"MLFLOW_RUN_NAME": f"Fine-tuning-{formatted_datetime}"})
        set_custom_env({"MLFLOW_EXPERIMENT_NAME": script_args.mlflow_experiment_name})
        
    train(script_args, training_args)
