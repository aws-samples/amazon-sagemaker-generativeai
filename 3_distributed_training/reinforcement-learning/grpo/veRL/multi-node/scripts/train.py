from dataclasses import dataclass, field
from huggingface_hub import snapshot_download
import glob
import logging
import os
import subprocess
import traceback
from transformers import (
    TrainingArguments,
)
from trl import TrlParser
from typing import Dict, Optional, Tuple
import wandb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class ScriptArguments:
    """
    Arguments for the script execution.
    """

    # Model parameters
    model_id: str = field(
        default=None, metadata={"help": "Model ID to use for training"}
    )

    token: str = field(default=None, metadata={"help": "Hugging Face API token"})

    # Experiment and tracking
    experiment_name: Optional[str] = field(
        default="codefu-7b-stage1", metadata={"help": "MLflow experiment name"}
    )

    wandb_token: str = field(default="", metadata={"help": "Wandb API token"})

    wandb_project: str = field(
        default="project", metadata={"help": "Wandb project name"}
    )

    # Data parameters
    train_dataset_path: Optional[str] = field(
        default=None, metadata={"help": "Path to the training dataset"}
    )

    test_dataset_path: Optional[str] = field(
        default=None, metadata={"help": "Path to the test dataset"}
    )

    # Algorithm configuration
    algorithm_adv_estimator: str = field(
        default="grpo", metadata={"help": "Advantage estimator algorithm"}
    )

    # Data configuration
    data_max_prompt_length: int = field(
        default=4096, metadata={"help": "Maximum prompt length"}
    )
    data_max_response_length: int = field(
        default=20480, metadata={"help": "Maximum response length"}
    )

    # Actor rollout ref model configuration
    actor_rollout_ref_model_use_remove_padding: bool = field(
        default=True, metadata={"help": "Use remove padding for model"}
    )

    # Actor rollout ref actor configuration
    actor_rollout_ref_actor_ppo_mini_batch_size: int = field(
        default=32, metadata={"help": "PPO mini batch size"}
    )
    actor_rollout_ref_actor_use_dynamic_bsz: bool = field(
        default=True, metadata={"help": "Use dynamic batch size"}
    )
    actor_rollout_ref_actor_ppo_micro_batch_size: int = field(
        default=4, metadata={"help": "PPO micro batch size"}
    )
    actor_rollout_ref_actor_use_kl_loss: bool = field(
        default=True, metadata={"help": "Use KL loss"}
    )
    actor_rollout_ref_actor_kl_loss_coef: float = field(
        default=0.001, metadata={"help": "KL loss coefficient"}
    )
    actor_rollout_ref_actor_kl_loss_type: str = field(
        default="low_var_kl", metadata={"help": "KL loss type"}
    )
    actor_rollout_ref_actor_ulysses_sequence_parallel_size: int = field(
        default=4, metadata={"help": "Ulysses sequence parallel size"}
    )

    # Actor rollout ref actor FSDP configuration
    actor_rollout_ref_actor_fsdp_config_param_offload: bool = field(
        default=True, metadata={"help": "FSDP parameter offload"}
    )
    actor_rollout_ref_actor_fsdp_config_grad_offload: bool = field(
        default=True, metadata={"help": "FSDP gradient offload"}
    )
    actor_rollout_ref_actor_fsdp_config_optimizer_offload: bool = field(
        default=True, metadata={"help": "FSDP optimizer offload"}
    )

    # Actor rollout ref rollout configuration
    actor_rollout_ref_rollout_log_prob_micro_batch_size: int = field(
        default=4, metadata={"help": "Rollout log prob micro batch size"}
    )
    actor_rollout_ref_rollout_tensor_model_parallel_size: int = field(
        default=2, metadata={"help": "Tensor model parallel size"}
    )
    actor_rollout_ref_rollout_name: str = field(
        default="vllm", metadata={"help": "Rollout engine name"}
    )
    actor_rollout_ref_rollout_temperature: float = field(
        default=1.0, metadata={"help": "Rollout temperature"}
    )
    actor_rollout_ref_rollout_gpu_memory_utilization: float = field(
        default=0.7, metadata={"help": "GPU memory utilization"}
    )
    actor_rollout_ref_rollout_n: int = field(
        default=8, metadata={"help": "Number of rollout samples"}
    )

    # Actor rollout ref reference configuration
    actor_rollout_ref_ref_log_prob_micro_batch_size: int = field(
        default=4, metadata={"help": "Reference log prob micro batch size"}
    )
    actor_rollout_ref_ref_fsdp_config_param_offload: bool = field(
        default=True, metadata={"help": "Reference FSDP parameter offload"}
    )

    # Algorithm KL control configuration
    algorithm_kl_ctrl_kl_coef: float = field(
        default=0.001, metadata={"help": "Algorithm KL control coefficient"}
    )

    # Trainer configuration
    trainer_critic_warmup: int = field(
        default=0, metadata={"help": "Critic warmup steps"}
    )
    trainer_save_freq: int = field(default=32, metadata={"help": "Save frequency"})
    trainer_test_freq: int = field(default=32, metadata={"help": "Test frequency"})


def download_model(model_name: str) -> None:
    print("Downloading model ", model_name)

    os.makedirs("/tmp/tmp_folder", exist_ok=True)

    snapshot_download(repo_id=model_name, local_dir="/tmp/tmp_folder")

    logger.info(f"Model {model_name} downloaded under /tmp/tmp_folder")


def prepare_data_paths(script_args: ScriptArguments) -> Tuple[str, str]:
    # Handle dataset paths - VERL expects file paths, not directory paths
    train_files = script_args.train_dataset_path
    val_files = script_args.test_dataset_path

    # If paths are directories, find the actual data files
    if os.path.isdir(train_files):
        # Look for common data file extensions
        for ext in ["*.parquet", "*.json", "*.jsonl", "*.csv"]:
            files = glob.glob(os.path.join(train_files, ext))
            if files:
                train_files = files[0]  # Use the first file found
                break
        else:
            # If no specific files found, use the directory path
            train_files = train_files.rstrip("/") + "/*"

    if os.path.isdir(val_files):
        # Look for common data file extensions
        for ext in ["*.parquet", "*.json", "*.jsonl", "*.csv"]:
            files = glob.glob(os.path.join(val_files, ext))
            if files:
                val_files = files[0]  # Use the first file found
                break
        else:
            # If no specific files found, use the directory path
            val_files = val_files.rstrip("/") + "/*"

    return train_files, val_files


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


def setup_wandb(script_args: ScriptArguments) -> None:
    """
    Set up Weights & Biases tracking.

    Args:
        script_args: Script arguments

    Returns:
        List of callbacks or None
    """
    if script_args.wandb_token and script_args.wandb_token != "":
        logger.info("Initializing Wandb")
        set_custom_env({"WANDB_API_KEY": script_args.wandb_token})
        wandb.login(key=script_args.wandb_token)
        return ["console", "wandb"]
    else:
        set_custom_env({"WANDB_DISABLED": "true"})
        return ["console"]


def train_fn(script_args: ScriptArguments, training_args: TrainingArguments) -> None:
    try:
        source_dir = os.environ.get("source_dir", "")
        current_dir = os.getcwd()  # This should be /opt/ml/input/data/code

        # Handle empty source_dir (script in same directory as launcher)
        if source_dir:
            absolute_source_dir = os.path.join(current_dir, source_dir)
        else:
            absolute_source_dir = current_dir

        train_files, val_files = prepare_data_paths(script_args)

        # Check wandb availability
        loggers = setup_wandb(script_args)

        # Pass environment variables to the script execution
        env_for_script = os.environ.copy()
        # Set the BASE_MODEL environment variable that VERL expects
        env_for_script.update(
            {
                "BASE_MODEL": script_args.model_id,
                "RUN_NAME": training_args.run_name,
                "WANDB_MODE": "online" if "wandb" in loggers else "offline",
            }
        )

        logger.info(f"Environment variables: {env_for_script}")

        # Ensure batch size is divisible by number of GPUs (8)
        train_batch_size = max(
            int(env_for_script["SM_NUM_GPUS"]),
            ((training_args.per_device_train_batch_size + 7) // 8) * 8,
        )
        val_batch_size = max(
            int(env_for_script["SM_NUM_GPUS"]),
            ((training_args.per_device_eval_batch_size + 7) // 8) * 8,
        )

        cmd = f"""
        python3 -m verl.trainer.main_ppo \\
            algorithm.adv_estimator={script_args.algorithm_adv_estimator} \\
            data.train_files={train_files} \\
            data.val_files={val_files} \\
            data.train_batch_size={train_batch_size} \\
            data.val_batch_size={val_batch_size} \\
            data.max_prompt_length={script_args.data_max_prompt_length} \\
            data.max_response_length={script_args.data_max_response_length} \\
            actor_rollout_ref.model.path={script_args.model_id} \\
            actor_rollout_ref.actor.optim.lr={training_args.learning_rate} \\
            actor_rollout_ref.model.use_remove_padding={script_args.actor_rollout_ref_model_use_remove_padding} \\
            actor_rollout_ref.actor.ppo_mini_batch_size={script_args.actor_rollout_ref_actor_ppo_mini_batch_size} \\
            actor_rollout_ref.actor.use_dynamic_bsz={script_args.actor_rollout_ref_actor_use_dynamic_bsz} \\
            actor_rollout_ref.actor.ppo_micro_batch_size={script_args.actor_rollout_ref_actor_ppo_micro_batch_size} \\
            actor_rollout_ref.actor.use_kl_loss={script_args.actor_rollout_ref_actor_use_kl_loss} \\
            actor_rollout_ref.actor.kl_loss_coef={script_args.actor_rollout_ref_actor_kl_loss_coef} \\
            actor_rollout_ref.actor.kl_loss_type={script_args.actor_rollout_ref_actor_kl_loss_type} \\
            actor_rollout_ref.actor.ulysses_sequence_parallel_size={script_args.actor_rollout_ref_actor_ulysses_sequence_parallel_size} \\
            actor_rollout_ref.model.enable_gradient_checkpointing={training_args.gradient_checkpointing} \\
            actor_rollout_ref.actor.fsdp_config.param_offload={script_args.actor_rollout_ref_actor_fsdp_config_param_offload} \\
            actor_rollout_ref.actor.fsdp_config.grad_offload={script_args.actor_rollout_ref_actor_fsdp_config_grad_offload} \\
            actor_rollout_ref.actor.fsdp_config.optimizer_offload={script_args.actor_rollout_ref_actor_fsdp_config_optimizer_offload} \\
            actor_rollout_ref.rollout.log_prob_micro_batch_size={script_args.actor_rollout_ref_rollout_log_prob_micro_batch_size} \\
            actor_rollout_ref.rollout.tensor_model_parallel_size={script_args.actor_rollout_ref_rollout_tensor_model_parallel_size} \\
            actor_rollout_ref.rollout.name={script_args.actor_rollout_ref_rollout_name} \\
            actor_rollout_ref.rollout.temperature={script_args.actor_rollout_ref_rollout_temperature} \\
            actor_rollout_ref.rollout.gpu_memory_utilization={script_args.actor_rollout_ref_rollout_gpu_memory_utilization} \\
            actor_rollout_ref.rollout.n={script_args.actor_rollout_ref_rollout_n} \\
            actor_rollout_ref.ref.log_prob_micro_batch_size={script_args.actor_rollout_ref_ref_log_prob_micro_batch_size} \\
            actor_rollout_ref.ref.fsdp_config.param_offload={script_args.actor_rollout_ref_ref_fsdp_config_param_offload} \\
            algorithm.kl_ctrl.kl_coef={script_args.algorithm_kl_ctrl_kl_coef} \\
            trainer.critic_warmup={script_args.trainer_critic_warmup} \\
            trainer.logger="{loggers}" \\
            trainer.project_name=thinking_llm_{script_args.experiment_name} \\
            trainer.experiment_name={training_args.run_name} \\
            trainer.n_gpus_per_node={int(env_for_script["SM_NUM_GPUS"])} \\
            trainer.nnodes={int(env_for_script["SM_HOST_COUNT"])} \\
            trainer.save_freq={script_args.trainer_save_freq} \\
            trainer.test_freq={script_args.trainer_test_freq} \\
            trainer.default_local_dir={training_args.output_dir}/{script_args.experiment_name}/{training_args.run_name} \\
            trainer.total_epochs={training_args.num_train_epochs}"""

        # Execute the training using shell=True to handle the multiline command
        logger.info(f"Running command:\n{cmd}")

        result = subprocess.run(
            cmd,
            check=True,
            shell=True,
            capture_output=False,
            text=True,
            env=env_for_script,
            cwd=absolute_source_dir,
        )

        logger.info("Bash script completed with return code: %s", result.returncode)

        if result.returncode != 0:
            logger.info(f"Training failed with return code: {result.returncode}")
            raise Exception(
                "Bash script failed with return code: %s", result.returncode
            )

        logger.info("Training completed successfully")
    except subprocess.CalledProcessError as e:
        stacktrace = traceback.format_exc()
        logger.error("Bash script failed with return code: %s", stacktrace)
        raise e
    except Exception as e:
        stacktrace = traceback.print_exc()
        logger.error("An error occurred during training: %s", stacktrace)

        raise e


def main():
    parser = TrlParser((ScriptArguments, TrainingArguments))
    script_args, training_args, _ = parser.parse_args_and_config(
        return_remaining_strings=True,
    )

    train_fn(script_args, training_args)


if __name__ == "__main__":
    main()
