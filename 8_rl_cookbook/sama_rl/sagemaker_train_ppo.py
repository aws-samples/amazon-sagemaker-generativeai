#!/usr/bin/env python3
import logging
import json
import os
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def install_dependencies():
    """Install exact dependencies from qwen-sentiment.py"""
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers", "trl==0.11.3", "wandb"])

def patch_backend_detection():
    import transformers.utils.import_utils
    def fixed_requires_backends(obj, backends):
        return
    transformers.utils.requires_backends = fixed_requires_backends

def build_dataset(config, dataset_name="stanfordnlp/imdb", input_min_text_length=2, input_max_text_length=8):
    """
    Build dataset exactly like qwen-sentiment.py
    """
    from transformers import AutoTokenizer
    from datasets import load_dataset
    from trl.core import LengthSampler
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load imdb with datasets
    ds = load_dataset(dataset_name, split="train")
    ds = ds.rename_columns({"text": "review"})
    ds = ds.filter(lambda x: len(x["review"]) > 200, batched=False)

    input_size = LengthSampler(input_min_text_length, input_max_text_length)

    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["review"])[: input_size()]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds

def collator(data):
    """Collator function exactly like qwen-sentiment.py"""
    return dict((key, [d[key] for d in data]) for key in data[0])

def main():
    logger.info("Installing dependencies...")
    install_dependencies()
    
    logger.info("Patching backend detection...")
    patch_backend_detection()
    
    logger.info("Starting PPO training following qwen-sentiment pattern...")
    
    # Import exactly like qwen-sentiment.py
    import torch
    import gc
    from tqdm import tqdm
    from transformers import pipeline, AutoTokenizer
    from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
    from trl.core import LengthSampler
    
    # Load config from SageMaker
    default_config = {
        "model": {"name": "Qwen/Qwen2.5-0.5B-Instruct"},
        "data": {"dataset_name": "stanfordnlp/imdb", "input_min_length": 2, "input_max_length": 8},
        "training": {"learning_rate": 1.41e-5, "max_steps": 50},
        "reward": {"model_name": "lvwerra/distilbert-imdb", "batch_size": 16}
    }
    
    config_data = None
    config_paths = [
        "/opt/ml/input/config/hyperparameters.json",
        "/opt/ml/input/data/config/config.json"
    ]
    
    for config_path in config_paths:
        try:
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config_data = json.load(f)
                logger.info("Loaded config from {}".format(config_path))
                break
        except Exception as e:
            logger.warning("Could not load config from {}: {}".format(config_path, e))
    
    if not config_data:
        config_dict = default_config
    else:
        config_dict = config_data.get("config", default_config)
    
    logger.info("Using config: {}".format(config_dict))
    
    # Create PPOConfig exactly like qwen-sentiment.py
    config = PPOConfig(
        model_name=config_dict["model"]["name"],
        learning_rate=float(config_dict.get("training", {}).get("learning_rate", 1.41e-5)),
        gradient_checkpointing=True,
        remove_unused_columns=False,
        log_with="wandb" if os.getenv("WANDB_API_KEY") else None,
    )
    
    sent_kwargs = {"top_k": None, "function_to_apply": "none", "batch_size": 1}
    
    logger.info("Building dataset...")
    dataset = build_dataset(
        config, 
        config_dict["data"]["dataset_name"],
        config_dict["data"].get("input_min_length", 2),
        config_dict["data"].get("input_max_length", 8)
    )
    
    logger.info("Loading models...")
    # Load models exactly like qwen-sentiment.py
    model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    logger.info("Initializing PPOTrainer...")
    # Initialize PPOTrainer exactly like qwen-sentiment.py
    ppo_trainer = PPOTrainer(
        config, model, ref_model, tokenizer, dataset=dataset, data_collator=collator
    )
    
    logger.info("Loading BERT sentiment classifier...")
    # Load BERT classifier exactly like qwen-sentiment.py
    device = ppo_trainer.accelerator.device
    if ppo_trainer.accelerator.num_processes == 1:
        device = 0 if torch.cuda.is_available() else "cpu"
    
    sentiment_pipe = pipeline(
        "sentiment-analysis", 
        model=config_dict.get("reward", {}).get("model_name", "lvwerra/distilbert-imdb"), 
        device=device
    )
    
    # Generation settings exactly like qwen-sentiment.py
    output_min_length = config_dict.get("training", {}).get("output_min_length", 4)
    output_max_length = config_dict.get("training", {}).get("output_max_length", 16)
    output_length_sampler = LengthSampler(output_min_length, output_max_length)
    
    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
    }
    
    logger.info("Starting training loop...")
    max_steps = min(int(config_dict.get("training", {}).get("max_steps", 50)), 20)
    
    # Training loop exactly like qwen-sentiment.py
    for epoch, batch in enumerate(tqdm(ppo_trainer.dataloader)):
        if epoch >= max_steps:
            break
            
        query_tensors = batch["input_ids"]

        # Get response from model
        response_tensors = []
        for query in query_tensors:
            gen_len = output_length_sampler()
            generation_kwargs["max_new_tokens"] = gen_len
            query_response = ppo_trainer.generate(query, **generation_kwargs).squeeze()
            response_len = len(query_response) - len(query)
            response_tensors.append(query_response[-response_len:])
        
        batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

        # Compute sentiment score
        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
        positive_scores = [
            item["score"]
            for output in pipe_outputs
            for item in output
            if item["label"] == "POSITIVE"
        ]
        rewards = [torch.tensor(score) for score in positive_scores]

        # Run PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        
        torch.cuda.empty_cache()
        gc.collect()
        
        # Clear memory after each step
        torch.cuda.empty_cache()
        gc.collect()
        
        logger.info("PPO step {}/{} completed, mean reward: {:.4f}".format(
            epoch + 1, max_steps, torch.stack(rewards).mean().item()
        ))
    
    # Save model
    logger.info("Saving model...")
    model.save_pretrained("/opt/ml/model")
    tokenizer.save_pretrained("/opt/ml/model")
    
    logger.info("PPO training completed!")

if __name__ == "__main__":
    main()
