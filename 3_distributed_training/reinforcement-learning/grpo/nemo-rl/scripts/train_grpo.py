# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# NeMo RL GRPO entry point for Amazon SageMaker Training Jobs.
# Designed to run via the Ray launcher: python launcher.py --entrypoint train_grpo.py
#
# The Ray launcher initializes the Ray cluster before loading this script
# via importlib. NeMo RL's init_ray() detects the existing cluster via
# ray.init(address="auto") and reuses it.
#
# Compatible with NeMo RL v0.5.0 (the version shipped in the NGC container).

import logging
import os
import pprint
from collections import defaultdict
from typing import Any, Optional

from omegaconf import OmegaConf

# Force all Ray actors to use system Python instead of uv-managed venvs.
# Must be set before importing the registry.
os.environ["NEMO_RL_PY_EXECUTABLES_SYSTEM"] = "1"

from nemo_rl.algorithms.grpo import MasterConfig, grpo_train, setup
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data import DataConfig
from nemo_rl.data.datasets import AllTaskProcessedDataset, load_response_dataset
from nemo_rl.data.interfaces import TaskDataProcessFnCallable, TaskDataSpec
from nemo_rl.data.processors import math_hf_data_processor
from nemo_rl.distributed.ray_actor_environment_registry import (
    ACTOR_ENVIRONMENT_REGISTRY,
)
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.environments.utils import create_env
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.utils.config import load_config
from nemo_rl.utils.logger import get_next_experiment_dir
from transformers import PreTrainedTokenizerBase

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

OmegaConf.register_new_resolver("mul", lambda a, b: a * b)

# Override all actor executables to use system Python.
for key in ACTOR_ENVIRONMENT_REGISTRY:
    ACTOR_ENVIRONMENT_REGISTRY[key] = PY_EXECUTABLES.SYSTEM


TokenizerType = PreTrainedTokenizerBase


def setup_data(
    tokenizer: TokenizerType,
    data_config: DataConfig,
    env_configs: dict[str, Any],
    seed: int,
) -> tuple[
    AllTaskProcessedDataset,
    Optional[AllTaskProcessedDataset],
    dict[str, EnvironmentInterface],
    dict[str, EnvironmentInterface],
]:
    """Setup datasets and environments (v0.5.0 API)."""
    logger.info("Setting up environments...")
    env_name = data_config["env_name"]
    env = create_env(env_name=env_name, env_configs=env_configs)

    logger.info("Setting up data...")
    default_task_spec = TaskDataSpec(
        task_name="math_default",
        prompt_file=data_config["prompt_file"],
        system_prompt_file=data_config["system_prompt_file"],
    )
    task_data_processors: dict[str, tuple[TaskDataSpec, TaskDataProcessFnCallable]] = (
        defaultdict(lambda: (default_task_spec, math_hf_data_processor))
    )

    data: Any = load_response_dataset(data_config, seed)
    task_spec = data.task_spec
    task_name = data.task_name
    assert hasattr(data, "processor"), "Dataset must have a processor attribute"
    task_data_processors[task_name] = (task_spec, data.processor)

    dataset = AllTaskProcessedDataset(
        data.formatted_ds["train"],
        tokenizer,
        default_task_spec,
        task_data_processors,
        max_seq_length=data_config["max_input_seq_length"],
    )

    val_dataset: Optional[AllTaskProcessedDataset] = None
    if data.formatted_ds["validation"]:
        val_dataset = AllTaskProcessedDataset(
            data.formatted_ds["validation"],
            tokenizer,
            default_task_spec,
            task_data_processors,
            max_seq_length=data_config["max_input_seq_length"],
        )

    task_to_env: dict[str, EnvironmentInterface] = defaultdict(lambda: env)
    task_to_env[task_name] = env
    return dataset, val_dataset, task_to_env, task_to_env


def main():
    config_path = os.environ.get(
        "NEMO_RL_CONFIG", "/opt/ml/input/data/config/grpo.yaml"
    )

    logger.info(f"Loading NeMo RL GRPO config from: {config_path}")
    config = load_config(config_path)
    config: MasterConfig = OmegaConf.to_container(config, resolve=True)

    logger.info("Final config:")
    pprint.pprint(config)

    config["logger"]["log_dir"] = get_next_experiment_dir(config["logger"]["log_dir"])
    logger.info(f"Using log directory: {config['logger']['log_dir']}")
    if config["checkpointing"]["enabled"]:
        logger.info(
            f"Using checkpoint directory: {config['checkpointing']['checkpoint_dir']}"
        )

    # Ray is already initialized by the launcher — skip init_ray()

    tokenizer = get_tokenizer(config["policy"]["tokenizer"])
    assert (
        config["policy"]["generation"] is not None
    ), "A generation config is required for GRPO"
    config["policy"]["generation"] = configure_generation_config(
        config["policy"]["generation"], tokenizer
    )

    (
        dataset,
        val_dataset,
        task_to_env,
        val_task_to_env,
    ) = setup_data(tokenizer, config["data"], config["env"], config["grpo"]["seed"])

    (
        policy,
        policy_generation,
        cluster,
        dataloader,
        val_dataloader,
        loss_fn,
        rl_logger,
        checkpointer,
        grpo_state,
        master_config,
    ) = setup(config, tokenizer, dataset, val_dataset)

    if "async_grpo" in config["grpo"] and config["grpo"]["async_grpo"]["enabled"]:
        from nemo_rl.algorithms.grpo import async_grpo_train

        logger.info("Running async GRPO training")
        async_config = config["grpo"]["async_grpo"]
        async_grpo_train(
            policy=policy,
            policy_generation=policy_generation,
            dataloader=dataloader,
            val_dataloader=val_dataloader,
            tokenizer=tokenizer,
            loss_fn=loss_fn,
            task_to_env=task_to_env,
            val_task_to_env=val_task_to_env,
            logger=rl_logger,
            checkpointer=checkpointer,
            grpo_save_state=grpo_state,
            master_config=master_config,
            max_trajectory_age_steps=async_config["max_trajectory_age_steps"],
        )
    else:
        logger.info("Running synchronous GRPO training")
        grpo_train(
            policy,
            policy_generation,
            dataloader,
            val_dataloader,
            tokenizer,
            loss_fn,
            task_to_env,
            val_task_to_env,
            rl_logger,
            checkpointer,
            grpo_state,
            master_config,
        )

    logger.info("GRPO training complete")


if __name__ == "__main__":
    main()
