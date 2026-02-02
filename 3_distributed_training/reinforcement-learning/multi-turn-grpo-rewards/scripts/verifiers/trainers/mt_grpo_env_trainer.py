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
from verifiers.trainers.grpo_env_trainer import GRPOEnvTrainer
if is_peft_available():
    from peft import PeftConfig

if is_wandb_available():
    import wandb

RewardFunc = Union[str, PreTrainedModel, Callable[[List, List], List[float]]]

class MTGRPOEnvTrainer(GRPOEnvTrainer):
    def __init__(
            self,
            model: Union[str, PreTrainedModel],
            env: Environment,        
            turn_reward_funcs: Union[RewardFunc, List[RewardFunc]],
            outcome_reward_funcs: Union[RewardFunc, List[RewardFunc]],
            turn_reward_weights: Optional[List[float]] = None,
            outcome_reward_weights: Optional[List[float]] = None,
            turn_advantage_coef: float = 1.0,
            args: Optional[GRPOConfig] = None,
            train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
            eval_dataset: Optional[Union[Dataset, IterableDataset]] = None,
            processing_class: Optional[PreTrainedTokenizerBase] = None,
            callbacks: Optional[List[TrainerCallback]] = None,
            optimizers: Tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
            peft_config: Optional["PeftConfig"] = None,
            **kwargs,
    ):

        super().__init__(
            model=model,
            env=env,
            turn_reward_funcs=turn_reward_funcs,
            outcome_reward_funcs=outcome_reward_funcs,
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
        self.turn_advantage_coef = turn_advantage_coef

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

        result_segment_indices = self._find_result_positions(completion_ids, completion_messages)

        advantages = self._assign_advantages(
            completion_mask, turn_advantages, outcome_advantages, combined_advantages, result_segment_indices
        )
        

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
        
    def _find_result_positions(self, completion_ids, completion_messages):
        result_segment_indices = []
        
        for i, messages in enumerate(completion_messages):
            result_segment = -1
            
            if isinstance(messages, list):
                for j, msg in enumerate(messages):
                    if msg.get('role') == 'assistant':
                        if j + 1 < len(messages) and messages[j + 1].get('role') == 'user':
                            user_msg = messages[j + 1].get('content', '')
                            if '<result>' in user_msg:
                                result_segment = j + 1
                                break
            elif isinstance(messages, str):
                raise ValueError("Completion is a string, which is not supported.")
            
            result_segment_indices.append(result_segment)
            
        return result_segment_indices
    
    def _assign_advantages(self, completion_mask, turn_advantages, outcome_advantages, combined_advantages, result_segment_indices):
        device = self.accelerator.device
        batch_size, seq_len = completion_mask.shape
        assigned_advantages = torch.zeros_like(completion_mask, dtype=torch.float32)

        def get_segment_boundaries(mask_row):
            boundaries = [0]
            for j in range(1, seq_len):
                if mask_row[j] != mask_row[j - 1]:
                    boundaries.append(j)
            boundaries.append(seq_len)
            return boundaries

        for i in range(batch_size):
            result_segment = result_segment_indices[i]
            mask_row = completion_mask[i]
            
            outcome_adv  = outcome_advantages[i]
            turn_adv = turn_advantages[i]
            combined_adv = combined_advantages[i]        

            outcome_adv_expanded = outcome_adv * torch.ones_like(mask_row, dtype=torch.float32)
            turn_adv_expanded = turn_adv * torch.ones_like(mask_row, dtype=torch.float32)
            combined_adv_expanded = combined_adv * torch.ones_like(mask_row, dtype=torch.float32)

            if result_segment > 0:
                segment_boundaries = get_segment_boundaries(mask_row)
                if result_segment < len(segment_boundaries):
                    split_point = segment_boundaries[result_segment]
                    before_result_mask = (torch.arange(seq_len, device=device) < split_point).float() * mask_row
                    assigned_advantages[i] = outcome_adv_expanded + self.turn_advantage_coef * turn_adv_expanded * before_result_mask
                else:
                    raise ValueError(f"Not enough segments found in completion {i}")
            else:
                assigned_advantages[i] = combined_adv_expanded

        return assigned_advantages

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)

        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        advantages = inputs["advantages"]
        old_per_token_logps = inputs["old_per_token_logps"] if self.num_iterations > 1 else per_token_logps.detach()
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon, 1 + self.epsilon)
        
        if len(advantages.shape) == 2:
            per_token_loss1 = coef_1 * advantages
            per_token_loss2 = coef_2 * advantages
        else:
            per_token_loss1 = coef_1 * advantages.unsqueeze(1)
            per_token_loss2 = coef_2 * advantages.unsqueeze(1)
            
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl
        loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()

        mode = "eval" if self.control.should_evaluate else "train"

        if self.beta != 0.0:
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
            self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        is_clipped = (per_token_loss1 < per_token_loss2).float()
        clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
        self._metrics[mode]["clip_ratio"].append(self.accelerator.gather_for_metrics(clip_ratio).mean().item())
        return loss
