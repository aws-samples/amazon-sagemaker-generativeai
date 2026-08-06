TOKENIZERS_PARALLELISM = "false"
PEFT_METHODS = ["ia3", "lora", "p_tune", "prompt_tuning"]
NO_DECAY = ["bias", "gamma", "beta"]

# Canonical source bucket containing sample training data for first-time setup.
# When a new user runs with an auto-created bucket, sample data is copied from here.
SEED_DATA_BUCKET = "finetune-blog"
SEED_DATA_ROOT_FOLDER = "finetune-blog"
