#!/usr/bin/env python
# coding: utf-8

# ## Fine tune model with GRPO verifiable reward

# In[1]:


import os
os.environ['hf_token']=""


# In[2]:


from huggingface_hub import login
login(token=os.environ["hf_token"])


# In[3]:


import sagemaker

sagemaker_session = sagemaker.Session()
bucket_name = sagemaker_session.default_bucket()
default_prefix = sagemaker_session.default_bucket_prefix


# In[4]:


from datasets import load_dataset
from scripts.utils.gsm8k import GSM8K
# Get the dataset from Huggingface
Num_shots = 8
dataset = GSM8K(split='train', include_answer=False, include_reasoning=True, few_shot=True, num_shots=Num_shots, seed=42, cot=True).dataset.shuffle(seed=42)


# In[5]:


dataset


# In[6]:


dataset_train_val = dataset.train_test_split(test_size=0.1)


# In[7]:


dataset_train_val


# In[8]:


dataset['prompt'][2]


# In[ ]:





# In[ ]:





# In[ ]:





# Train the model using the Model Trainer API

# In[9]:


import boto3
import shutil
import sagemaker
sagemaker_session = sagemaker.Session()
s3_client = boto3.client('s3')

bucket_name = sagemaker_session.default_bucket()
default_prefix = sagemaker_session.default_bucket_prefix

# save train_dataset to s3 using our SageMaker session
if default_prefix:
    input_path = f"{default_prefix}/datasets/finetuning-modeltrainer-rlvr"
else:
    input_path = f"datasets/finetuning-modeltrainer-rlvr"

train_dataset_s3_path = f"s3://{bucket_name}/{input_path}/train/dataset.json"
val_dataset_s3_path = f"s3://{bucket_name}/{input_path}/val/dataset.json"

# Save datasets to s3
# We will fine tune only with 20 records due to limited compute resource for the workshop
dataset_train_val['train'].to_json("./data/train/dataset.json", orient="records")
dataset_train_val['test'].to_json("./data/val/dataset.json", orient="records")

s3_client.upload_file("./data/train/dataset.json", bucket_name, f"{input_path}/train/dataset.json")
s3_client.upload_file("./data/val/dataset.json", bucket_name, f"{input_path}/val/dataset.json")

shutil.rmtree("./data")

print(f"Training data uploaded to:")
print(train_dataset_s3_path)
print(val_dataset_s3_path)


# In[10]:


MLFLOW_TRACKING_SERVER_ARN = 'arn:aws:sagemaker:us-east-1:783764584149:mlflow-tracking-server/MLflow3-test' # or "arn:aws:sagemaker:us-west-2:<account-id>:mlflow-tracking-server/<server-name>"


# In[11]:


import sagemaker
from sagemaker.config import load_sagemaker_config
sagemaker_session = sagemaker.Session()

bucket_name = sagemaker_session.default_bucket()
default_prefix = sagemaker_session.default_bucket_prefix
configs = load_sagemaker_config()
instance_type = "ml.p4d.24xlarge" #"ml.g6.48xlarge" # Override the instance type if you want to get a different container version
instance_count = 1
config_filename = "Qwen2.5-0.5B.yaml" 
print(instance_type)
image_uri = sagemaker.image_uris.retrieve(
    framework="pytorch",
    region=sagemaker_session.boto_session.region_name,
    version="2.7.1",
    instance_type=instance_type,
    image_scope="training"
)
print(config_filename)
print(image_uri)


# In[12]:


from sagemaker.modules.configs import (
    CheckpointConfig,
    Compute,
    OutputDataConfig,
    SourceCode,
    StoppingCondition,
)
from sagemaker.modules.distributed import Torchrun
from sagemaker.modules.train import ModelTrainer
env = {}
env["FI_PROVIDER"] = "efa"
env["NCCL_PROTO"] = "simple"
env["NCCL_SOCKET_IFNAME"] = "eth0"
env["NCCL_IB_DISABLE"] = "1"
env["NCCL_DEBUG"] = "WARN"
env["HF_token"] = os.environ['hf_token']
env["CONFIG_PATH"] = f"recipes/{config_filename}"
env["MLFLOW_EXPERIMENT_NAME"]= "grpo-rlvr"
env["MLFLOW_TAGS"] =  '{"source.job": "sm-training-jobs", "source.type": "grpo-rlvr", "source.framework": "pytorch"}'
env["MLFLOW_TRACKING_URI"] =  MLFLOW_TRACKING_SERVER_ARN
# Define the script to be run
source_code = SourceCode(
    source_dir="./scripts",
    requirements="requirements.txt",
    entry_script="run_finetuning.sh",
)

# Define the compute
compute_configs = Compute(
    instance_type=instance_type,
    instance_count=instance_count,
    keep_alive_period_in_seconds=3600,
)

# define Training Job Name
job_name = f"train-{config_filename.split('/')[-1].replace('.', '-').replace('yaml', 'rlvr')}-shots-{Num_shots}"
print(job_name)
# define OutputDataConfig path
if default_prefix:
    output_path = f"s3://{bucket_name}/{default_prefix}/{job_name}"
else:
    output_path = f"s3://{bucket_name}/{job_name}"

# Define the ModelTrainer
model_trainer = ModelTrainer(
    training_image=image_uri,
     environment=env,
    source_code=source_code,
    base_job_name=job_name,
    compute=compute_configs,
    stopping_condition=StoppingCondition(max_runtime_in_seconds=18000),
    output_data_config=OutputDataConfig(s3_output_path=output_path),
    checkpoint_config=CheckpointConfig(
        s3_uri=output_path + "/checkpoint", local_path="/opt/ml/checkpoints"
    ),
)


# In[13]:


from sagemaker.modules.configs import InputData

# Pass the input data
train_input = InputData(
    channel_name="train",
    data_source=train_dataset_s3_path, # S3 path where training data is stored
)

val_input = InputData(
    channel_name="val",
    data_source=val_dataset_s3_path, # S3 path where training data is stored
)

# Check input channels configured
data = [train_input, val_input]
data


# In[14]:


model_trainer.train(input_data_config=data, wait=False)


# ***
# 
# ## Load Fine-Tuned model
# 
# Note: Run `train_fn` with `merge_weights=True` for merging the trained adapter

# ### Download model

# In[16]:


import boto3
import json
import sagemaker
# define Training Job Name
sagemaker_session = sagemaker.Session()
Num_shots = 8
bucket_name = sagemaker_session.default_bucket()
default_prefix = sagemaker_session.default_bucket_prefix
job_prefix = f"train-{config_filename.split('/')[-1].replace('.', '-').replace('yaml', 'rlvr')}-shots-{Num_shots}"


# In[17]:


def get_last_job_name(job_name_prefix):
    sagemaker_client = boto3.client('sagemaker')

    matching_jobs = []
    next_token = None

    while True:
        # Prepare the search parameters
        search_params = {
            'Resource': 'TrainingJob',
            'SearchExpression': {
                'Filters': [
                    {
                        'Name': 'TrainingJobName',
                        'Operator': 'Contains',
                        'Value': job_name_prefix
                    },
                    {
                        'Name': 'TrainingJobStatus',
                        'Operator': 'Equals',
                        'Value': "Completed"
                    }
                ]
            },
            'SortBy': 'CreationTime',
            'SortOrder': 'Descending',
            'MaxResults': 100
        }

        # Add NextToken if we have one
        if next_token:
            search_params['NextToken'] = next_token

        # Make the search request
        search_response = sagemaker_client.search(**search_params)

        # Filter and add matching jobs
        matching_jobs.extend([
            job['TrainingJob']['TrainingJobName'] 
            for job in search_response['Results']
            if job['TrainingJob']['TrainingJobName'].startswith(job_name_prefix)
        ])

        # Check if we have more results to fetch
        next_token = search_response.get('NextToken')
        if not next_token or matching_jobs:  # Stop if we found at least one match or no more results
            break

    if not matching_jobs:
        raise ValueError(f"No completed training jobs found starting with prefix '{job_name_prefix}'")

    return matching_jobs[0]


# In[18]:


job_name = get_last_job_name(job_prefix)

job_prefix, job_name


# In[ ]:





# #### Inference configurations

# ## Download model data

# In[21]:


import boto3
import os

if default_prefix:
    object_key = f"{default_prefix}/{job_prefix}/{job_name}/output/model.tar.gz"
else:
    object_key = f"{job_prefix}/{job_name}/output/model.tar.gz"



# Local paths
local_archive_path = f"./temp/{job_name}/model.tar.gz" #'./temp/model.tar.gz'
local_model_dir = f"./temp/extracted_model/{job_name}/" #'./temp/extracted_model'

# Create the /tmp directory if it doesn't exist
os.makedirs(os.path.dirname(local_archive_path), exist_ok=True)
os.makedirs(local_model_dir, exist_ok=True)

# Download the file from S3
s3_client.download_file(bucket_name, object_key, local_archive_path)

print(f"Downloaded {object_key} to {local_archive_path}")


# ### Extract The model data

# In[144]:


import tarfile

# Extract the tar.gz file
with tarfile.open(local_archive_path, "r:gz") as tar:
    tar.extractall(path=local_model_dir)

print(f"Extracted model files to {local_model_dir}")


# ### Evaluate The model

# At first we need to merge the adapter

# In[19]:


import re
from datasets import load_dataset
from dataclasses import dataclass, field
import tempfile
from typing import Optional
import torch
from peft import AutoPeftModelForCausalLM
from peft import PeftConfig, PeftModel, AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
import evaluate


# In[20]:


# --- 1. Load the dataset, tokenizer, and model ---
# Use the GSM8K test split.
dataset = GSM8K(split='test', include_answer=False, include_reasoning=True, few_shot=True, num_shots=8, seed=42, cot=True).dataset.shuffle(seed=42)

dataset = dataset.select(range(50))


# In[21]:


dataset['prompt'][0]


# In[22]:


def merge_and_save_model(model_path_or_id, save_dir, save_tokenizer=True):
    # Load the base model and tokenizer
    config = PeftConfig.from_pretrained(model_path_or_id)
    base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path_or_id)
    
    # Add special tokens to the tokenizer
    #tokenizer.add_special_tokens({'pad_token': ''})
    
    # Resize the token embeddings of the base model
    base_model.resize_token_embeddings(len(tokenizer))
    
    # Now load the PEFT model with the resized base model
    model = PeftModel.from_pretrained(base_model, model_path_or_id)
    
    # Merge LoRA and base model and save
    model = model.merge_and_unload()        
    model.save_pretrained(save_dir, safe_serialization=True, max_shard_size="3GB")
  
    # save tokenizer
    if save_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(model_path_or_id)
        tokenizer.save_pretrained(save_dir) 
        
def extract_answer(text):
    """
    Extracts the numerical answer from the model's text output.
    This function looks for the final number in the output, which is a common practice.
    It removes commas to handle large numbers correctly.
    """
    # The `re.findall` finds all sequences of digits, potentially with a minus sign.
    numbers = re.findall(r'-?\d+', text.replace(',', ''))
    if numbers:
        # We assume the final number is the answer.
        return numbers[-1]
    return None
# Run the evaluation 
# For a full evaluation, you would generate a CoT prompt with examples from the train set.
# For simplicity, this example uses a zero-shot prompt.
# Few-shot CoT prompting is the standard approach for best results.
def evaluate_on_gsm8k(model, tokenizer, dataset):
    correct_count = 0
    total_count = len(dataset)
    model.eval()
    for i, example in enumerate(dataset):
        question = example["question"]
        ground_truth = example["final_answer"]

        # Create a simple prompt. For CoT, you would construct a more complex prompt.
        prompt = example["prompt"]

        # Generate the model's response.
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, do_sample=False, max_new_tokens=1024, pad_token_id=tokenizer.eos_token_id)
        model_output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the model's predicted answer.
        predicted_answer = extract_answer(model_output_text)

        # print(f"--- Example {i+1}/{total_count} ---")
        # print(f"Question: {question}")
        # print(f"Model Output: {model_output_text}")
        # print(f"Extracted Answer: {predicted_answer}")
        # print(f"Ground Truth: {ground_truth}")
        # print(f"--------------------------------")

        if predicted_answer and predicted_answer == ground_truth:
            correct_count += 1
            #print("Status: Correct\n")
        else:
            correct_count=correct_count
            #print("Status: Incorrect\n")

    accuracy = correct_count / total_count
    print("--- Evaluation Summary ---")
    print(f"Total problems: {total_count}")
    print(f"Correct predictions: {correct_count}")
    print(f"Accuracy: {accuracy:.4f}")


# In[23]:


#merge_and_save_model(f"./temp/extracted_model/{job_name}/Qwen2.5-0.5B-RL-VR-GRPO", f"./temp/merged-weights/{job_name}/", save_tokenizer=True)



# In[24]:


# Load a pre-trained model and tokenizer
#model_name = "Qwen/Qwen2.5-0.5B"
model_name = f"./temp/merged-weights/{job_name}/"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


# #### results for 8 shots model

# In[ ]:


evaluate_on_gsm8k(model,tokenizer,dataset)


# #### results for 4 shots model

# In[139]:


evaluate_on_gsm8k(model,tokenizer,dataset)


# #### results for 2 shots model

# In[127]:


evaluate_on_gsm8k(model,tokenizer,dataset)


# #### results for 0 shots model

# In[107]:


evaluate_on_gsm8k(model,tokenizer,dataset)


# ### Run evaluation on base model

# In[108]:


# Load a pre-trained model and tokenizer from Hugging Face Hub.
# You can replace this with your own model.
model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForCausalLM.from_pretrained(model_name)


# In[109]:


evaluate_on_gsm8k(base_model,tokenizer,dataset)


# In[ ]:




