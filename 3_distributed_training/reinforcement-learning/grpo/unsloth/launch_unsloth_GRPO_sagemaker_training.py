import sagemaker
from sagemaker.estimator import Estimator

def main():
    role = "arn:aws:iam::811828458885:role/SuryaKariAdmin"
    session = sagemaker.Session()

    training_script = "sagemaker_grpo_training_wb_tracing.py"

    # Upload only the script
    s3_script_uri = session.upload_data(training_script, bucket=session.default_bucket(), key_prefix="scriptfolder")

    estimator = Estimator(
        image_uri="811828458885.dkr.ecr.us-east-2.amazonaws.com/unsloth-train-grpo",
        role=role,
        instance_count=1,
        instance_type="ml.g5.16xlarge",
        volume_size=100,
        max_run=3600 * 16,
        entry_point=training_script,
        source_dir=".",
        dependencies=["estimator_requirements.txt"],  # <- install wandb
        hyperparameters={  # optional, or handled in entrypoint.sh
            "PER_DEVICE_TRAIN_BATCH_SIZE": "1",
            "GRADIENT_ACCUMULATION_STEPS": "4",
            "NUM_TRAIN_EPOCHS": "3"
        },
        environment={
            "WANDB_API_KEY": "923d4369fa7415f96348bc3d001e780d1a367e0b",
            "WANDB_PROJECT": "grpo-training",
            "WANDB_RUN_NAME": "grpo-run-01",
            "WANDB_CONSOLE": "wrap"

        }
    )

    estimator.fit()

if __name__ == "__main__":
    main()
