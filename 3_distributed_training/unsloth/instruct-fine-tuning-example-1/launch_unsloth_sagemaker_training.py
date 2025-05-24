import sagemaker
from sagemaker.estimator import Estimator
import shutil
import os

def main():
    role = "arn:aws:iam::811828458885:role/SuryaKariAdmin"
    session = sagemaker.Session()
    bucket = session.default_bucket()

    # File paths
    training_script = "sagemaker_unsloth_qwen2_5_trainer.py"
    original_dataset_file = "data/alpaca_autotune_datagen_with_cot_4000.json"

    # Copy and rename dataset to match what training expects
    renamed_dataset = "data.json"
    shutil.copyfile(original_dataset_file, renamed_dataset)

    # Upload script and dataset
    s3_script_uri = session.upload_data(training_script, bucket=bucket, key_prefix="scripts")
    s3_dataset_uri = session.upload_data(renamed_dataset, bucket=bucket, key_prefix="datasets")

    print("Uploaded script to:", s3_script_uri)
    print("Uploaded dataset to:", s3_dataset_uri)

    # Estimator config using your custom Docker image
    estimator = Estimator(
        image_uri="811828458885.dkr.ecr.us-east-2.amazonaws.com/unsloth-train",
        role=role,
        instance_count=1,
        instance_type="ml.g5.2xlarge",
        volume_size=100,
        max_run=3600 * 4,
        entry_point=training_script,
        source_dir=".",
    )

    # Map the S3 dataset to the expected input channel
    inputs = {
        "train": s3_dataset_uri,
        "train_file": s3_dataset_uri
    }

    estimator.fit(inputs)

if __name__ == "__main__":
    main()
