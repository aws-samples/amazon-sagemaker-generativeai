import os
import boto3
import json
import tarfile
import argparse
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
    default_data_collator,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from peft import (
    get_peft_model,
    LoraConfig,
    prepare_model_for_kbit_training,
)
from peft.tuners.lora import LoraLayer
from datasets import load_from_disk
import torch
import bitsandbytes as bnb
import sagemaker
import shutil
from smexperiments_callback import SageMakerExperimentsCallback



def s3_download(s3_bucket, s3_object_key, local_file_name, s3_client=boto3.client('s3')):
    meta_data = s3_client.head_object(Bucket=s3_bucket, Key=s3_object_key)
    total_length = int(meta_data.get('ContentLength', 0))
    with tqdm(total=total_length,  
              desc=f'source: s3://{s3_bucket}/{s3_object_key}', 
              bar_format="{percentage:.1f}%|{bar:25} | {rate_fmt} | {desc}",  
              unit='B', 
              unit_scale=True, 
              unit_divisor=1024
             ) as pbar:
        with open(local_file_name, 'wb') as f:
            s3_client.download_fileobj(s3_bucket, s3_object_key, f, Callback=pbar.update)

            
def download_and_untar_s3_tar(destination_file_path, source_s3_path):

    src_s3_bucket = source_s3_path.split('/')[2]
    src_s3_prefix = "/".join(source_s3_path.split('/')[3:])
    destination_file_path = os.path.join(destination_file_path, os.path.basename(source_s3_path))
    
    print(f"Downloading file from {src_s3_bucket}/{src_s3_prefix} to {destination_file_path}")

    s3_download(
        s3_bucket=src_s3_bucket, 
        s3_object_key=src_s3_prefix, 
        local_file_name=destination_file_path
    )

    # Create a tarfile object and extract the contents to the local disk
    tar = tarfile.open(destination_file_path, "r")
    tar.extractall()
    tar.close()
    
    
def model_data_uri_from_model_package(model_group_name, region="us-east-1"):

    sagemaker_session = sagemaker.session.Session(boto3.session.Session(region_name=region))
    region = sagemaker_session.boto_region_name

    sm_client = boto3.client('sagemaker', region_name=region)

    model_packages = sm_client.list_model_packages(
        ModelPackageGroupName=model_group_name
    )['ModelPackageSummaryList']

    model_package_name = sorted(
        [
            (package['ModelPackageVersion'], package['ModelPackageArn']) 
            for package in model_packages if package['ModelApprovalStatus'] == 'Approved'], 
        reverse=False
    )[-1][-1]
    print(f"found model package: {model_package_name}")
    
    return sm_client.describe_model_package(
        ModelPackageName=model_package_name
    )['InferenceSpecification']['Containers'][0]['ModelDataUrl']


# Reference: https://github.com/artidoro/qlora/blob/main/qlora.py
def print_trainable_parameters(
    model, 
    use_4bit=False
):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    if use_4bit:
        trainable_params /= 2
    print(
        f"all params: {all_param:,d} || trainable params: {trainable_params:,d} || trainable%: {100 * trainable_params / all_param}"
    )


# Reference: https://github.com/artidoro/qlora/blob/main/qlora.py
def find_all_linear_names(
    model
):
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, bnb.nn.Linear4bit):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def create_peft_model(
    model, 
    r_value, 
    lora_alpha, 
    lora_dropout, 
    task_type,
    gradient_checkpointing=True, 
    bf16=True
):

    # prepare int-4 model for training
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=gradient_checkpointing
    )
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # get lora target modules
    modules = find_all_linear_names(model)
    print(f"Found {len(modules)} modules to quantize: {modules}")

    peft_config = LoraConfig(
        r=r_value,
        lora_alpha=lora_alpha,
        target_modules=modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=task_type
    )

    model = get_peft_model(model, peft_config)

    model.print_trainable_parameters()
    
    return model


def finetune_llm(args):
    # set seed
    set_seed(args.seed)
    
    print(f"loading dataset from {args.sm_train_dir} and {args.sm_validation_dir}")

    train_dataset = load_from_disk(args.sm_train_dir)
    validation_dataset = load_from_disk(args.sm_validation_dir)
    
    print(f"region: {args.region}")
    # gets the latest base model from model package group
    model_data_uri = model_data_uri_from_model_package(
        model_group_name=args.base_model_group_name,
        region=args.region
    )
    
    print(f"base model s3 uri: {model_data_uri} from: {args.base_model_group_name} \n")
    
    download_and_untar_s3_tar(
        destination_file_path="/tmp/", 
        source_s3_path=model_data_uri
    )
    
    print(os.listdir(f"./{args.model_id}"))
    print(f"Untar base model to ./{args.model_id}")
    
    # load model from the hub with a bnb config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    local_model_path = f"./{args.model_id}"

    model = AutoModelForCausalLM.from_pretrained(
        local_model_path,
        use_cache=False if args.gradient_checkpointing else True,
        device_map="auto",
        quantization_config=bnb_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)

    # create peft config
    model = create_peft_model(
        model, 
        r_value=args.lora_r, 
        lora_alpha=args.lora_alpha, 
        lora_dropout=args.lora_dropout, 
        task_type=args.task_type,
        gradient_checkpointing=args.gradient_checkpointing, 
        bf16=args.bf16
    )

    # Define training args
    training_args = TrainingArguments(
        output_dir=f"{args.sm_output_dir}/{args.model_id}/trainer-outputs",
        per_device_train_batch_size=args.per_device_train_batch_size,
        bf16=args.bf16,  # Use BF16 if available
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        gradient_checkpointing=args.gradient_checkpointing,
        logging_dir=f"{args.sm_output_dir}/{args.model_id}/logs",
        logging_strategy="steps",
        logging_steps=args.sm_exp_logging_steps,
        save_strategy="no",
    )

    # Create Trainer instance with SageMaker experiments callback
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=default_data_collator,
        callbacks=[SageMakerExperimentsCallback(region=args.region)]
    )
    
    # mutes warnings during training, reenable during inference
    model.config.use_cache = False 

    # Start training
    trainer.train()
    
    # Start evaluation
    trainer.evaluate()
    
    temp_dir="/tmp/model/"
    
    if args.merge_weights:
        
        trainer.model.save_pretrained(temp_dir, safe_serialization=False)
        # clear memory
        del model
        del trainer
        torch.cuda.empty_cache()
        
        from peft import AutoPeftModelForCausalLM

        # load PEFT model in fp16
        model = AutoPeftModelForCausalLM.from_pretrained(
            temp_dir,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        )  
        # Merge LoRA and base model and save
        model = model.merge_and_unload()        
        model.save_pretrained(
            args.sm_model_dir, safe_serialization=True, max_shard_size="2GB"
        )   
        
        source_dir = './djl-inference/'

        # copy djl-inference files to model directory
        for f in os.listdir(source_dir):
            source_f = os.path.join(source_dir, f)
            
            # Copy the files to the destination folder
            shutil.copy(source_f, args.sm_model_dir)
        
    else:   
        # save finetuned LoRA model and then the tokenizer for inference
        trainer.model.save_pretrained(
            args.sm_model_dir, 
            safe_serialization=True
        )
    tokenizer.save_pretrained(
        args.sm_model_dir
    )
    
    print("Done!")
    

def read_parameters():
    parser = argparse.ArgumentParser()
    # add model id and dataset path argument
    parser.add_argument("--model_id", type=str, help="Hugging face model id to use for training.")
    parser.add_argument("--base_model_group_name", type=str, help="Base model package group name to use for downloading base model")
    parser.add_argument("--epochs", type=int, default=1, help="No of epochs to train the model")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="Batch size to use for training.")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Model learning rate")
    parser.add_argument("--seed", type=int, default=8, help="Seed to use for training")
    parser.add_argument("--gradient_checkpointing", type=bool, default=True, help="Path to deepspeed config file")
    parser.add_argument("--bf16", type=bool, default=False if torch.cuda.get_device_capability()[0] == 8 else False, help="Whether to use bf16.")
    parser.add_argument("--lora_r", type=int, default=64, help="Lora attention dimension value")
    parser.add_argument("--lora_alpha", type=int, default=16, help="The alpha parameter for Lora scaling")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="The dropout probability for Lora layers")
    parser.add_argument(
        "--task_type", 
        type=str, default="CAUSAL_LM", 
        help="Choose from: CAUSAL_LM, FEATURE_EXTRACTION, QUESTION_ANS, SEQ_2_SEQ_LM, SEQ_CLS, TOKEN_CLS"
    )
    parser.add_argument(
        "--merge_weights",
        action='store_true',
        help="Whether to merge LoRA weights with base model.",
    )
    parser.add_argument("--sm_exp_logging_steps", type=int, default=2, help="Step interval to start logging to console/sagemaker experiments")
    parser.add_argument("--region", type=str, default="us-east-1", help="SageMaker job execution region")
    
    # sagemaker env args: refer to this for more arguments: https://github.com/aws/sagemaker-training-toolkit/blob/master/ENVIRONMENT_VARIABLES.md
    parser.add_argument("--sm_model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--sm_train_dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--sm_validation_dir", type=str, default=os.environ["SM_CHANNEL_VALIDATION"])
    parser.add_argument("--sm_current_host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--sm_hosts", type=list, default=os.environ["SM_HOSTS"])
    parser.add_argument("--sm_output_dir", type=list, default=os.environ["SM_OUTPUT_DIR"])
    parser.add_argument("--n_gpus", type=list, default=os.environ["SM_NUM_GPUS"])
    
    args, _ = parser.parse_known_args()
    return args


def main():
    args = read_parameters()
    print(args)
    finetune_llm(args)


if __name__ == "__main__":
    main()
