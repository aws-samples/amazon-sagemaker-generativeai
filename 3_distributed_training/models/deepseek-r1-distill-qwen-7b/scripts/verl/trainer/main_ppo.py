# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""
import json
from verl import DataProto
import torch
from verl.utils.reward_score import (
    gsm8k,
    multiply,
    countdown,
    math_rwf,
    code_contests,
)
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


def _select_rm_score_fn(data_source):
    if data_source == "openai/gsm8k":
        return gsm8k.compute_score
    elif data_source == "code_contests":
        return code_contests.compute_score
    elif data_source == "lighteval/MATH":
        return math_rwf.compute_score
    elif "multiply" in data_source or "arithmetic" in data_source:
        return multiply.compute_score
    elif "countdown" in data_source:
        return countdown.compute_score
    else:
        raise NotImplementedError


class RewardManager:
    """The reward manager with Ray-based parallelization."""

    def __init__(self, tokenizer, num_examine) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine

        # Create a Ray remote function that doesn't rely on retrieving the tokenizer
        @ray.remote
        def process_reward_item(
            idx, valid_response_length, sequences_str, data_source, reward_model_data
        ):
            # Notice we receive valid_response_length and sequences_str directly

            # Handle reward model data
            if isinstance(reward_model_data, str):
                reward_model_data = json.loads(reward_model_data)
            ground_truth = reward_model_data["ground_truth"]

            # Dynamically import scoring functions
            from verl.utils.reward_score import (
                gsm8k,
                multiply,
                countdown,
                math_rwf,
                code_contests,
            )

            # Select scoring function
            if data_source == "openai/gsm8k":
                compute_score = gsm8k.compute_score
            elif data_source == "code_contests":
                compute_score = code_contests.compute_score
            elif data_source == "lighteval/MATH":
                compute_score = math_rwf.compute_score
            elif "multiply" in data_source or "arithmetic" in data_source:
                compute_score = multiply.compute_score
            elif "countdown" in data_source:
                compute_score = countdown.compute_score
            else:
                raise NotImplementedError

            # Calculate score
            score = compute_score(solution_str=sequences_str, ground_truth=ground_truth)

            return idx, score, valid_response_length, sequences_str, data_source

        # Store the remote function as a class attribute
        self.process_reward_item = process_reward_item

    def __call__(self, data: DataProto):
        """Compute rewards in parallel using Ray"""

        # If there is rm score, we directly return rm score
        if "rm_scores" in data.batch.keys():
            return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        already_print_data_sources = {}

        # For small batches (4 or fewer), use sequential processing
        if len(data) <= 2:
            # Original sequential code
            for i in range(len(data)):
                data_item = data[i]  # DataProtoItem

                prompt_ids = data_item.batch["prompts"]
                prompt_length = prompt_ids.shape[-1]
                valid_prompt_length = data_item.batch["attention_mask"][
                    :prompt_length
                ].sum()
                valid_prompt_ids = prompt_ids[-valid_prompt_length:]

                response_ids = data_item.batch["responses"]
                valid_response_length = data_item.batch["attention_mask"][
                    prompt_length:
                ].sum()
                valid_response_ids = response_ids[:valid_response_length]

                # decode
                sequences = torch.cat((valid_prompt_ids, valid_response_ids))
                sequences_str = self.tokenizer.decode(sequences)

                reward_model = data_item.non_tensor_batch["reward_model"]
                if isinstance(reward_model, str):
                    reward_model = json.loads(reward_model)

                ground_truth = reward_model["ground_truth"]

                # select rm_score
                data_source = data_item.non_tensor_batch["data_source"]
                compute_score_fn = _select_rm_score_fn(data_source)

                score = compute_score_fn(
                    solution_str=sequences_str, ground_truth=ground_truth
                )
                reward_tensor[i, valid_response_length - 1] = score

                if data_source not in already_print_data_sources:
                    already_print_data_sources[data_source] = 0

                if already_print_data_sources[data_source] < self.num_examine:
                    already_print_data_sources[data_source] += 1
                    print(sequences_str)
        else:
            # Parallel processing with Ray for larger batches
            futures = []
            for i in range(len(data)):
                data_item = data[i]

                # Do the tokenization work here in the main process
                prompt_ids = data_item.batch["prompts"]
                prompt_length = prompt_ids.shape[-1]
                valid_prompt_length = data_item.batch["attention_mask"][
                    :prompt_length
                ].sum()
                valid_prompt_ids = prompt_ids[-valid_prompt_length:]

                response_ids = data_item.batch["responses"]
                valid_response_length = data_item.batch["attention_mask"][
                    prompt_length:
                ].sum()
                valid_response_ids = response_ids[:valid_response_length]

                # Decode in the main process
                sequences = torch.cat((valid_prompt_ids, valid_response_ids))
                sequences_str = self.tokenizer.decode(sequences)

                # Send only serializable data to the Ray task
                futures.append(
                    self.process_reward_item.remote(
                        i,
                        valid_response_length,  # Pass this value to the worker
                        sequences_str,
                        data_item.non_tensor_batch["data_source"],
                        data_item.non_tensor_batch["reward_model"],
                    )
                )

            # Get results
            results = ray.get(futures)

            # Process results and update reward tensor
            for (
                idx,
                score,
                valid_response_length,
                sequences_str,
                data_source,
            ) in results:
                reward_tensor[idx, valid_response_length - 1] = score

                if data_source not in already_print_data_sources:
                    already_print_data_sources[data_source] = 0

                if already_print_data_sources[data_source] < self.num_examine:
                    already_print_data_sources[data_source] += 1
                    print(sequences_str)

        return reward_tensor


import ray
import hydra


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(
            runtime_env={
                "env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}
            }
        )

    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer

    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf

    pprint(
        OmegaConf.to_container(config, resolve=True)
    )  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer

    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == "fsdp":
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup

        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == "megatron":
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup

        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
    }

    global_pool_id = "global_pool"
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable:
        if config.reward_model.strategy == "fsdp":
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == "megatron":
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0)

    # Note that we always use function-based RM for validation
    val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=1)

    resource_pool_manager = ResourcePoolManager(
        resource_pool_spec=resource_pool_spec, mapping=mapping
    )

    trainer = RayPPOTrainer(
        config=config,
        tokenizer=tokenizer,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=ray_worker_group_cls,
        reward_fn=reward_fn,
        val_reward_fn=val_reward_fn,
    )
    trainer.init_workers()
    trainer.fit()


if __name__ == "__main__":
    main()
