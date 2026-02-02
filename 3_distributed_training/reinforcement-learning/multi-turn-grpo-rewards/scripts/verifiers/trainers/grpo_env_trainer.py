from typing import Callable, Optional, Union, Any, List, Dict, Tuple

from accelerate.utils import broadcast_object_list, gather, gather_object
from datasets import Dataset, IterableDataset
import torch
from torch import nn
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainerCallback,
    is_wandb_available,
    Trainer,
)
from transformers.utils import is_peft_available
from trl import GRPOTrainer, GRPOConfig
from trl.data_utils import apply_chat_template, maybe_apply_chat_template
from trl.import_utils import is_rich_available
from trl.trainer.utils import pad

from verifiers.envs.environment import Environment
from verifiers.utils.logging_utils import print_prompt_completions_sample

if is_peft_available():
    from peft import PeftConfig

if is_wandb_available():
    import wandb

RewardFunc = Union[str, PreTrainedModel, Callable[[List, List], List[float]]]

class GRPOEnvTrainer(GRPOTrainer):
    def __init__(
            self,
            model: Union[str, PreTrainedModel],
            env: Environment,        
            turn_reward_funcs: Union[RewardFunc, List[RewardFunc]],
            outcome_reward_funcs: Union[RewardFunc, List[RewardFunc]],
            turn_reward_weights: Optional[List[float]] = None,
            outcome_reward_weights: Optional[List[float]] = None,
            no_turn_reward: Optional[bool] = None,
            args: Optional[GRPOConfig] = None,
            train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
            eval_dataset: Optional[Union[Dataset, IterableDataset]] = None,
            processing_class: Optional[PreTrainedTokenizerBase] = None,
            callbacks: Optional[List[TrainerCallback]] = None,
            optimizers: Tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
            peft_config: Optional["PeftConfig"] = None,
            **kwargs,
    ):
        if not args.use_vllm:
            raise ValueError("vLLM must be enabled for GRPOEnvTrainer")
        if not (callable(turn_reward_funcs) or (isinstance(turn_reward_funcs, list) and all(callable(f) for f in turn_reward_funcs))):
            raise ValueError("turn_reward_funcs must be a function or a list of functions.")
        if not (callable(outcome_reward_funcs) or (isinstance(outcome_reward_funcs, list) and all(callable(f) for f in outcome_reward_funcs))):
            raise ValueError("outcome_reward_funcs must be a function or a list of functions.")

        self.turn_reward_funcs = turn_reward_funcs
        self.outcome_reward_funcs = outcome_reward_funcs
        self.combined_reward_funcs = turn_reward_funcs + outcome_reward_funcs

        self.num_turn_funcs = len(turn_reward_funcs)
        self.num_outcome_funcs = len(outcome_reward_funcs)
        self.num_combined_reward_funcs = len(self.combined_reward_funcs)

        if turn_reward_weights is None:
            self.turn_reward_weights = torch.ones(self.num_turn_funcs)
        else:
            self.turn_reward_weights = torch.tensor(turn_reward_weights)
        if outcome_reward_weights is None:
            self.outcome_reward_weights = torch.ones(self.num_outcome_funcs)
        else:
            self.outcome_reward_weights = torch.tensor(outcome_reward_weights)
        self.combined_reward_weights = torch.cat([self.turn_reward_weights, self.outcome_reward_weights], dim=0)

        self.no_turn_reward = no_turn_reward

        super().__init__(
            model=model,
            reward_funcs=self.combined_reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
            **kwargs,
        )
        self.env = env

    def _generate_and_score_completions(
         self, inputs: Dict[str, Union[torch.Tensor, Any]]   
    ) -> Dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        
        prompts = [x["prompt"] for x in inputs]
        prompt_ids, prompt_mask = self._prepare_prompt_inputs(inputs)
         
        if self.state.global_step != self._last_loaded_step:
            self._move_model_to_vllm()
            self._last_loaded_step = self.state.global_step
            
        completion_ids, completion_messages, completion_mask = self._generate_completions(prompts)

        prompt_completion_ids, attention_mask, logits_to_keep = self._prepare_model_inputs(
            prompt_ids, prompt_mask, completion_ids, completion_mask
        )
        
        old_per_token_logps, ref_per_token_logps = self._compute_logps(
            prompt_completion_ids, attention_mask, logits_to_keep
        )

        turn_rewards_per_func = self._calculate_rewards(
            prompts, completion_messages, self.turn_reward_funcs, inputs
        )
        outcome_rewards_per_func = self._calculate_rewards(
            prompts, completion_messages, self.outcome_reward_funcs, inputs
        )
        combined_rewards_per_func = self._calculate_rewards(
            prompts, completion_messages, self.combined_reward_funcs, inputs
        )

        turn_rewards = (turn_rewards_per_func * self.turn_reward_weights.to(device).unsqueeze(0)).sum(dim=1)
        outcome_rewards = (outcome_rewards_per_func * self.outcome_reward_weights.to(device).unsqueeze(0)).sum(dim=1)
        combined_rewards = (combined_rewards_per_func * self.combined_reward_weights.to(device).unsqueeze(0)).sum(dim=1)

        turn_mean_grouped_rewards, turn_std_grouped_rewards, turn_advantages = self._compute_normalized_advantages(turn_rewards, len(prompts))
        outcome_mean_grouped_rewards, outcome_std_grouped_rewards, outcome_advantages = self._compute_normalized_advantages(outcome_rewards, len(prompts))
        combined_mean_grouped_rewards, combined_std_grouped_rewards, combined_advantages = self._compute_normalized_advantages(combined_rewards, len(prompts))
        
        advantages = outcome_advantages if self.no_turn_reward else combined_advantages

        mode = "eval" if self.control.should_evaluate else "train"

        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics[mode]["completion_length"].append(completion_length)

        turn_rewards_per_func = turn_rewards_per_func.mean(dim=0)
        for i, reward_func in enumerate(self.turn_reward_funcs):
            reward_func_name = reward_func.__name__
            self._metrics[mode][f"rewards/turn/{reward_func_name}"].append(turn_rewards_per_func[i].item())
            
        outcome_rewards_per_func = outcome_rewards_per_func.mean(dim=0)
        for i, reward_func in enumerate(self.outcome_reward_funcs):
            reward_func_name = reward_func.__name__
            self._metrics[mode][f"rewards/outcome/{reward_func_name}"].append(outcome_rewards_per_func[i].item())

        self._metrics[mode]["reward/turn"].append(turn_rewards.mean().item())
        self._metrics[mode]["reward/outcome"].append(outcome_rewards.mean().item())
        self._metrics[mode]["reward/combined"].append(combined_rewards.mean().item())
        self._metrics[mode]["reward_std/turn"].append(turn_std_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std/outcome"].append(outcome_std_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std/combined"].append(combined_std_grouped_rewards.mean().item())

        if self.log_completions and self.state.global_step % self.args.logging_steps == 0:
            self._log_completion_samples(prompts, completion_messages, combined_rewards)

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
        }
        
    def _prepare_prompt_inputs(self, inputs):
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        prompt_inputs = self.processing_class(
            prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length:]
            prompt_mask = prompt_mask[:, -self.max_prompt_length:]
            
        return prompt_ids, prompt_mask
    
    def _generate_completions(self, prompts):
        all_prompts = gather_object(prompts)
        if self.accelerator.is_main_process:
            env_result = self.env.generate(
                prompts=all_prompts,
                llm=self.llm,
                sampling_params=self.sampling_params,
            )
            completion_ids = env_result['ids']
            completion_messages = env_result['messages']
            completion_mask = env_result['mask']
        else:
            completion_ids = [None] * len(all_prompts)
            completion_messages = [None] * len(all_prompts)
            completion_mask = [None] * len(all_prompts)

        completion_ids = broadcast_object_list(completion_ids, from_process=0)
        completion_messages = broadcast_object_list(completion_messages, from_process=0)
        completion_mask = broadcast_object_list(completion_mask, from_process=0)

        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )

        completion_ids = completion_ids[process_slice]
        completion_messages = completion_messages[process_slice]
        completion_mask = completion_mask[process_slice]
        
        device = self.accelerator.device
        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
        completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)

        completion_mask = [torch.tensor(mask, device=device) for mask in completion_mask]
        completion_mask = pad(completion_mask, padding_value=0)
        
        return completion_ids, completion_messages, completion_mask
    
    def _prepare_model_inputs(self, prompt_ids, prompt_mask, completion_ids, completion_mask):
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        return prompt_completion_ids, attention_mask, logits_to_keep
    
    def _compute_logps(self, prompt_completion_ids, attention_mask, logits_to_keep):
        with torch.no_grad():
            if self.num_iterations > 1:
                old_per_token_logps = self._get_per_token_logps(
                    self.model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                old_per_token_logps = None

            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids, attention_mask, logits_to_keep
                    )
                    
        return old_per_token_logps, ref_per_token_logps
    
    def _calculate_rewards(self, prompts, completions, reward_funcs, inputs):
        device = self.accelerator.device
        rewards_per_func = torch.zeros(len(prompts), len(reward_funcs), device=device)
        
        for i, reward_func in enumerate(reward_funcs):
            keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
            reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
            output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
            rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        return gather(rewards_per_func)
    
    def _compute_normalized_advantages(self, rewards, slice_length=None):
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        process_slice = slice(
            self.accelerator.process_index * slice_length,
            (self.accelerator.process_index + 1) * slice_length,
        )
        advantages = advantages[process_slice]
        
        return mean_grouped_rewards, std_grouped_rewards, advantages
    
    def _log_completion_samples(self, prompts, completions, rewards):
        prompts_to_log = gather_object(prompts)
        completions_to_log = gather_object(completions)
        rewards_to_log = rewards.tolist()

        if self.accelerator.is_main_process:
            if is_rich_available():
                print_prompt_completions_sample(
                    [str(prompts_to_log[0][-1]["content"])],
                    [completions_to_log[0]],
                    [rewards_to_log[0]],
                    self.state.global_step,
                )
            if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                import pandas as pd

                table = {
                    "step": [str(self.state.global_step)] * len(rewards),
                    "prompt": prompts_to_log,
                    "completion": completions_to_log,
                    "reward": rewards.tolist(),
                }
                df = pd.DataFrame(table)
                wandb.log({"completions": wandb.Table(dataframe=df)})
