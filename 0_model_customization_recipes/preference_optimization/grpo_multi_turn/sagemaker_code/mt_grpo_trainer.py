"""
Multi-Turn GRPO Trainer for Tool Calling Agents

Supports:
- Multi-turn conversation with tool execution
- Turn-level and outcome-level rewards
- Position-aware advantage assignment
- Dynamic tool/reward function loading
- YAML-based configuration via TrlParser
- Distributed training with DeepSpeed and Accelerate

Usage:
    python mt_grpo_trainer.py --config hf_recipes/Qwen/Qwen3-1.7B--mt-grpo.yaml
"""

import importlib.util
import inspect
import json
import logging
import os
import random
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from accelerate.utils import broadcast_object_list, gather, gather_object
from datasets import Dataset, IterableDataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import is_peft_available
from trl import GRPOConfig, GRPOTrainer, TrlParser, ModelConfig
from trl.data_utils import maybe_apply_chat_template
from trl.trainer.utils import pad

if is_peft_available():
    from peft import PeftConfig

if is_wandb_available():
    import wandb


def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    return logging.getLogger(__name__)


logger = setup_logging()


@dataclass
class ScriptArguments:
    dataset_name: str = field(default="triviaqa", metadata={"help": "Dataset name (triviaqa)"})
    tools_script: Optional[str] = field(default=None, metadata={"help": "Path to tool functions script"})
    reward_fn: Optional[str] = field(default=None, metadata={"help": "Path to reward functions script"})
    max_env_steps: int = field(default=2, metadata={"help": "Max environment steps"})
    turn_advantage_coef: float = field(default=1.0, metadata={"help": "Turn advantage coefficient"})
    # vllm_server_host: str = field(default="0.0.0.0", metadata={"help": "vLLM server host"})
    # vllm_server_port: int = field(default=8000, metadata={"help": "vLLM server port"})


class XMLParser:
    def __init__(self, fields: List[Union[str, Tuple[str, ...]]]):
        self._fields = []
        for field_def in fields:
            if isinstance(field_def, str):
                self._fields.append((field_def, [field_def]))
            elif isinstance(field_def, tuple):
                self._fields.append((field_def[0], list(field_def)))

    def parse(self, text: str, strip: bool = True) -> Any:
        results = {}
        for canonical, alternatives in self._fields:
            for alt in alternatives:
                pattern = rf"<{alt}>\s*(.*?)\s*</{alt}>"
                match = re.search(pattern, text, re.DOTALL)
                results[alt] = match.group(1).strip() if match and strip else (match.group(1) if match else None)
        return SimpleNamespace(**results)

    def format(self, **kwargs) -> str:
        parts = []
        for canonical, alternatives in self._fields:
            value = kwargs.get(canonical)
            if value is None:
                for alt in alternatives:
                    if alt in kwargs:
                        value = kwargs[alt]
                        break
            if value is not None:
                parts.append(f"<{canonical}>\n{value}\n</{canonical}>")
        return "\n".join(parts)


def infer_schema_from_function(func: Callable) -> Dict[str, Any]:
    sig = inspect.signature(func)
    doc = inspect.getdoc(func) or ""
    description = doc.split("\n\n")[0].strip()
    args = {}
    for name, param in sig.parameters.items():
        args[name] = {
            "type": str(param.annotation.__name__ if param.annotation != inspect.Parameter.empty else "any"),
            "description": "",
        }
    return {"name": func.__name__, "description": description, "args": args}


def format_tool_descriptions(schemas: List[Dict[str, Any]]) -> str:
    descriptions = []
    for schema in schemas:
        desc = [f"{schema['name']}: {schema['description']}"]
        desc.append("\nArguments:")
        for arg_name, arg_info in schema['args'].items():
            desc.append(f"  - {arg_name}: {arg_info['description']}")
        descriptions.append("\n".join(desc))
    return "\n\n".join(descriptions)


SYSTEM_PROMPT_TEMPLATE = """\
You have access to the following tools to help solve problems:

{tool_descriptions}

Follow these steps exactly once:
1. Think through your reasoning inside <reasoning> tags
2. Use a tool by writing a JSON command inside <tool> tags with:
   - "name": the tool to use
   - "args": the arguments for the tool
3. You will see the tool's output inside <result> tags
4. Think through the tool's output inside <reasoning> tags
5. Based on your reasoning, provide your final answer inside <answer> tags

Important:
- Use the tool exactly once - DO NOT attempt to call the tool again
- Tools expect specific JSON input formats
- After getting the tool result, analyze it before giving your answer
"""


class VLLMChatResponse:
    def __init__(self, text: str, prompt_token_ids: List[int], output_token_ids: List[int]):
        self.prompt_token_ids = prompt_token_ids
        self.outputs = [SimpleNamespace(text=text, token_ids=output_token_ids)]


class VLLMServerClient:
    """
    Client for vLLM's OpenAI-compatible API server.
    
    Start the server with:
        python -m vllm.entrypoints.openai.api_server --model <model> --port 8000
    
    NOT compatible with `trl vllm-serve` which uses different endpoints.
    """
    def __init__(self, host: str, port: int, tokenizer: Any, model_name: Optional[str] = None):
        self.base_url = f"http://{host}:{port}"
        self.tokenizer = tokenizer
        self.model_name = model_name  # Will be auto-detected if None
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import httpx
            self._client = httpx.Client(base_url=self.base_url, timeout=600.0)
        return self._client

    def _get_model_name(self) -> str:
        """Get the model name from the server if not already set."""
        if self.model_name:
            return self.model_name
        try:
            response = self.client.get("/v1/models")
            response.raise_for_status()
            models = response.json().get("data", [])
            if models:
                self.model_name = models[0]["id"]
                logger.info(f"Auto-detected vLLM model: {self.model_name}")
                return self.model_name
        except Exception as e:
            logger.warning(f"Could not auto-detect model name: {e}")
        return "default"

    def chat(self, messages_list: List[List[Dict[str, str]]], sampling_params: Any = None, use_tqdm: bool = False) -> List[VLLMChatResponse]:
        responses = []
        max_tokens = getattr(sampling_params, 'max_tokens', 512)
        temperature = getattr(sampling_params, 'temperature', 1.0)
        top_p = getattr(sampling_params, 'top_p', 1.0)
        stop = getattr(sampling_params, 'stop', None)
        
        # Get model name (auto-detect on first call)
        model_name = self._get_model_name()

        for messages in messages_list:
            try:
                prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                prompt_token_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
                request_data = {
                    "model": model_name,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                }
                if stop:
                    request_data["stop"] = stop
                response = self.client.post("/v1/chat/completions", json=request_data)
                response.raise_for_status()
                result = response.json()
                response_text = result["choices"][0]["message"]["content"] or ""
                if stop and getattr(sampling_params, 'include_stop_str_in_output', False):
                    if result["choices"][0].get("finish_reason") == "stop":
                        for stop_str in stop:
                            if not response_text.endswith(stop_str):
                                response_text = response_text + stop_str
                                break
                output_token_ids = self.tokenizer.encode(response_text, add_special_tokens=False)
                responses.append(VLLMChatResponse(text=response_text, prompt_token_ids=prompt_token_ids, output_token_ids=output_token_ids))
            except Exception as e:
                logger.error(f"Error in vLLM chat: {e}")
                prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) if messages else ""
                prompt_token_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
                responses.append(VLLMChatResponse(text=f"Error: {str(e)}", prompt_token_ids=prompt_token_ids, output_token_ids=[]))
        return responses


class MultiTurnToolEnv:
    def __init__(self, tools: List[Callable], system_prompt: str, max_steps: int = 2, mask_env_response: bool = True, max_workers: int = 10, sleep_time: float = 0.1):
        self.tool_schemas = [infer_schema_from_function(tool) for tool in tools]
        self.tools = {tool.__name__: tool for tool in tools}
        tool_descriptions = format_tool_descriptions(self.tool_schemas)
        self.system_prompt = system_prompt.format(tool_descriptions=tool_descriptions)
        self.sampling_args = {"skip_special_tokens": False, "spaces_between_special_tokens": False, "n": 1, "stop": ["</tool>", "</answer>"], "include_stop_str_in_output": True}
        self.env_mask = 0 if mask_env_response else 1
        self.max_workers = max_workers
        self.max_steps = max_steps
        self.sleep_time = sleep_time
        self.llm_parser = XMLParser(fields=["reasoning", ("tool", "answer")])
        self.env_parser = XMLParser(fields=["result"])

    def _get_step_count(self, messages: List[Dict[str, str]]) -> int:
        step_count = 0
        for message in messages[1:]:
            if message.get("role") == "assistant":
                try:
                    parsed = self.llm_parser.parse(message["content"])
                    if hasattr(parsed, 'tool') and parsed.tool is not None:
                        step_count += 1
                except Exception:
                    pass
        return step_count

    def is_completed(self, messages: List[Dict[str, str]]) -> bool:
        try:
            if self._get_step_count(messages) >= self.max_steps:
                return True
            parsed = self.llm_parser.parse(messages[-1]["content"])
            return hasattr(parsed, 'answer') and parsed.answer is not None
        except Exception:
            return False

    def call_tool(self, tool_json: str) -> str:
        try:
            command = json.loads(tool_json)
            tool_name = command.get("name")
            if not tool_name or tool_name not in self.tools:
                return f"Error: Unknown tool '{tool_name}'"
            return str(self.tools[tool_name](**command.get("args", {})))
        except json.JSONDecodeError:
            return "Error: Invalid JSON format"
        except Exception as e:
            return f"Error: {str(e)}"

    def env_response(self, messages: List[Dict[str, str]]) -> Dict[str, str]:
        try:
            parsed = self.llm_parser.parse(messages[-1]["content"])
            if hasattr(parsed, 'tool') and parsed.tool is not None:
                result = self.call_tool(parsed.tool)
                return {"role": "user", "content": self.env_parser.format(result=result)}
        except Exception:
            pass
        return {"role": "user", "content": "Error: Tool command not found or invalid."}

    def step(self, states: List[Dict[str, Any]], llm: Any, sampling_params: Any) -> List[Dict[str, Any]]:
        live_indices = [i for i, s in enumerate(states) if not s["completed"]]
        messages_to_step = [states[i]["messages"] for i in live_indices]
        llm_responses = llm.chat(messages_to_step, sampling_params=sampling_params, use_tqdm=False)

        def update_state(j, llm_response):
            time.sleep(self.sleep_time * random.random())
            state = states[j].copy()
            if len(state["prompt_ids"]) == 0:
                state["prompt_ids"] = llm_response.prompt_token_ids
            state["messages"].append({"role": "assistant", "content": llm_response.outputs[0].text})
            total_prev_len = len(state["prompt_ids"]) + len(state["completion_ids"])
            env_response_len = len(list(llm_response.prompt_token_ids)) - total_prev_len
            new_completion_len = len(llm_response.outputs[0].token_ids)
            state["completion_mask"].extend([self.env_mask] * env_response_len)
            state["completion_mask"].extend([1] * new_completion_len)
            state["completion_ids"] = list(llm_response.prompt_token_ids)
            state["completion_ids"].extend(list(llm_response.outputs[0].token_ids))
            state["completion_ids"] = state["completion_ids"][len(state["prompt_ids"]):]
            if self.is_completed(state["messages"]) or len(state["completion_ids"]) > sampling_params.max_tokens:
                state["completed"] = True
                state["completion_ids"] = state["completion_ids"][:sampling_params.max_tokens]
                state["completion_mask"] = state["completion_mask"][:len(state["completion_ids"])]
            else:
                state["messages"].append(self.env_response(state["messages"]))
            return j, state

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(lambda args: update_state(*args), [(j, llm_responses[i]) for i, j in enumerate(live_indices)]))
        for j, state in results:
            states[j] = state
        return states

    def generate(self, prompts: List[List[Dict[str, Any]]], llm: Any, sampling_params: Any) -> Dict[str, List[Any]]:
        custom_sp = sampling_params.clone()
        for k, v in self.sampling_args.items():
            setattr(custom_sp, k, v)
        states = [{"messages": list(m), "prompt_messages": len(m), "prompt_ids": [], "completed": False, "completion_ids": [], "completion_mask": []} for m in prompts]
        while not all(s["completed"] for s in states):
            states = self.step(states, llm, custom_sp)
        return {"ids": [s["completion_ids"] for s in states], "messages": [s["messages"][s["prompt_messages"]:] for s in states], "mask": [s["completion_mask"] for s in states]}


RewardFunc = Union[str, PreTrainedModel, Callable[[List, List], List[float]]]


class MTGRPOEnvTrainer(GRPOTrainer):
    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        env: MultiTurnToolEnv,
        turn_reward_funcs: List[RewardFunc],
        outcome_reward_funcs: List[RewardFunc],
        turn_reward_weights: Optional[List[float]] = None,
        outcome_reward_weights: Optional[List[float]] = None,
        turn_advantage_coef: float = 1.0,
        vllm_server_host: str = "0.0.0.0",
        vllm_server_port: int = 8000,
        args: Optional[GRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        **kwargs,
    ):
        self.turn_reward_funcs = turn_reward_funcs if isinstance(turn_reward_funcs, list) else [turn_reward_funcs]
        self.outcome_reward_funcs = outcome_reward_funcs if isinstance(outcome_reward_funcs, list) else [outcome_reward_funcs]
        self.combined_reward_funcs = self.turn_reward_funcs + self.outcome_reward_funcs
        self.num_turn_funcs = len(self.turn_reward_funcs)
        self.num_outcome_funcs = len(self.outcome_reward_funcs)
        self.turn_reward_weights = torch.ones(self.num_turn_funcs) if turn_reward_weights is None else torch.tensor(turn_reward_weights)
        self.outcome_reward_weights = torch.ones(self.num_outcome_funcs) if outcome_reward_weights is None else torch.tensor(outcome_reward_weights)
        self.combined_reward_weights = torch.cat([self.turn_reward_weights, self.outcome_reward_weights], dim=0)
        self.turn_advantage_coef = turn_advantage_coef
        self._vllm_server_host = vllm_server_host
        self._vllm_server_port = vllm_server_port

        if args.use_vllm:
            logger.warning("Disabling TRL's use_vllm - using custom vLLM client")
            args.use_vllm = False

        super().__init__(model=model, reward_funcs=self.combined_reward_funcs, args=args, train_dataset=train_dataset, eval_dataset=eval_dataset, processing_class=processing_class, callbacks=callbacks, optimizers=optimizers, peft_config=peft_config, **kwargs)
        self.env = env
        self._init_vllm_client()
        self._last_loaded_step = -1
        if not hasattr(self, 'epsilon'):
            self.epsilon = 0.2
        logger.info(f"MTGRPOEnvTrainer initialized with vLLM server: {vllm_server_host}:{vllm_server_port}")

    def _init_vllm_client(self):
        self.llm = VLLMServerClient(host=self._vllm_server_host, port=self._vllm_server_port, tokenizer=self.processing_class)
        self.sampling_params = SimpleNamespace(max_tokens=self.args.max_completion_length, temperature=1.0, top_p=1.0, stop=None, include_stop_str_in_output=True, skip_special_tokens=False, spaces_between_special_tokens=False, n=1)
        self.sampling_params.clone = lambda: SimpleNamespace(**vars(self.sampling_params))

    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep):
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        logits = logits[:, -(logits_to_keep + 1):-1, :]
        labels = input_ids[:, -logits_to_keep:]
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        return torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    def _generate_and_score_completions(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]
        prompt_ids, prompt_mask = self._prepare_prompt_inputs(inputs)
        if self.state.global_step != self._last_loaded_step:
            self._last_loaded_step = self.state.global_step
        completion_ids, completion_messages, completion_mask = self._generate_completions(prompts)
        prompt_completion_ids, attention_mask, logits_to_keep = self._prepare_model_inputs(prompt_ids, prompt_mask, completion_ids, completion_mask)
        old_per_token_logps, ref_per_token_logps = self._compute_logps(prompt_completion_ids, attention_mask, logits_to_keep)
        turn_rewards_per_func = self._calculate_rewards(prompts, completion_messages, self.turn_reward_funcs, inputs)
        outcome_rewards_per_func = self._calculate_rewards(prompts, completion_messages, self.outcome_reward_funcs, inputs)
        combined_rewards_per_func = self._calculate_rewards(prompts, completion_messages, self.combined_reward_funcs, inputs)
        turn_rewards = (turn_rewards_per_func * self.turn_reward_weights.to(device).unsqueeze(0)).sum(dim=1)
        outcome_rewards = (outcome_rewards_per_func * self.outcome_reward_weights.to(device).unsqueeze(0)).sum(dim=1)
        combined_rewards = (combined_rewards_per_func * self.combined_reward_weights.to(device).unsqueeze(0)).sum(dim=1)
        _, turn_std, turn_advantages = self._compute_normalized_advantages(turn_rewards, len(prompts))
        _, outcome_std, outcome_advantages = self._compute_normalized_advantages(outcome_rewards, len(prompts))
        _, combined_std, combined_advantages = self._compute_normalized_advantages(combined_rewards, len(prompts))
        result_segment_indices = self._find_result_positions(completion_messages)
        advantages = self._assign_advantages(completion_mask, turn_advantages, outcome_advantages, combined_advantages, result_segment_indices)
        mode = "eval" if self.control.should_evaluate else "train"
        self._metrics[mode]["completion_length"].append(self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item())
        self._metrics[mode]["reward/turn"].append(turn_rewards.mean().item())
        self._metrics[mode]["reward/outcome"].append(outcome_rewards.mean().item())
        self._metrics[mode]["reward/combined"].append(combined_rewards.mean().item())
        if self.log_completions and self.state.global_step % self.args.logging_steps == 0:
            self._log_completion_samples(prompts, completion_messages, combined_rewards)
        return {"prompt_ids": prompt_ids, "prompt_mask": prompt_mask, "completion_ids": completion_ids, "completion_mask": completion_mask, "old_per_token_logps": old_per_token_logps, "ref_per_token_logps": ref_per_token_logps, "advantages": advantages}

    def _prepare_prompt_inputs(self, inputs):
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        prompt_inputs = self.processing_class(prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False)
        prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length:]
            prompt_mask = prompt_mask[:, -self.max_prompt_length:]
        return prompt_ids, prompt_mask

    def _generate_completions(self, prompts):
        all_prompts = gather_object(prompts)
        if self.accelerator.is_main_process:
            env_result = self.env.generate(prompts=all_prompts, llm=self.llm, sampling_params=self.sampling_params)
            completion_ids, completion_messages, completion_mask = env_result['ids'], env_result['messages'], env_result['mask']
        else:
            completion_ids = [None] * len(all_prompts)
            completion_messages = [None] * len(all_prompts)
            completion_mask = [None] * len(all_prompts)
        completion_ids = broadcast_object_list(completion_ids, from_process=0)
        completion_messages = broadcast_object_list(completion_messages, from_process=0)
        completion_mask = broadcast_object_list(completion_mask, from_process=0)
        process_slice = slice(self.accelerator.process_index * len(prompts), (self.accelerator.process_index + 1) * len(prompts))
        completion_ids = completion_ids[process_slice]
        completion_messages = completion_messages[process_slice]
        completion_mask = completion_mask[process_slice]
        device = self.accelerator.device
        completion_ids = pad([torch.tensor(ids, device=device) for ids in completion_ids], padding_value=self.processing_class.pad_token_id)
        completion_mask = pad([torch.tensor(mask, device=device) for mask in completion_mask], padding_value=0)
        return completion_ids, completion_messages, completion_mask

    def _prepare_model_inputs(self, prompt_ids, prompt_mask, completion_ids, completion_mask):
        return torch.cat([prompt_ids, completion_ids], dim=1), torch.cat([prompt_mask, completion_mask], dim=1), completion_ids.size(1)

    def _compute_logps(self, prompt_completion_ids, attention_mask, logits_to_keep):
        with torch.no_grad():
            old_per_token_logps = self._get_per_token_logps(self.model, prompt_completion_ids, attention_mask, logits_to_keep) if self.num_iterations > 1 else None
            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep)
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(self.model, prompt_completion_ids, attention_mask, logits_to_keep)
        return old_per_token_logps, ref_per_token_logps

    def _calculate_rewards(self, prompts, completions, reward_funcs, inputs):
        device = self.accelerator.device
        rewards_per_func = torch.zeros(len(prompts), len(reward_funcs), device=device)
        for i, reward_func in enumerate(reward_funcs):
            keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
            reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
            output = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
            rewards_per_func[:, i] = torch.tensor(output, dtype=torch.float32, device=device)
        return gather(rewards_per_func)

    def _compute_normalized_advantages(self, rewards, slice_length):
        mean = rewards.view(-1, self.num_generations).mean(dim=1).repeat_interleave(self.num_generations, dim=0)
        std = rewards.view(-1, self.num_generations).std(dim=1).repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean) / (std + 1e-4)
        process_slice = slice(self.accelerator.process_index * slice_length, (self.accelerator.process_index + 1) * slice_length)
        return mean, std, advantages[process_slice]

    def _find_result_positions(self, completion_messages):
        result_segment_indices = []
        for messages in completion_messages:
            result_segment = -1
            if isinstance(messages, list):
                for j, msg in enumerate(messages):
                    if msg.get('role') == 'assistant':
                        if j + 1 < len(messages) and messages[j + 1].get('role') == 'user':
                            if '<result>' in messages[j + 1].get('content', ''):
                                result_segment = j + 1
                                break
            result_segment_indices.append(result_segment)
        return result_segment_indices

    def _assign_advantages(self, completion_mask, turn_advantages, outcome_advantages, combined_advantages, result_segment_indices):
        device = self.accelerator.device
        batch_size, seq_len = completion_mask.shape
        assigned = torch.zeros_like(completion_mask, dtype=torch.float32)
        def get_boundaries(mask_row):
            boundaries = [0]
            for j in range(1, seq_len):
                if mask_row[j] != mask_row[j - 1]:
                    boundaries.append(j)
            boundaries.append(seq_len)
            return boundaries
        for i in range(batch_size):
            result_seg = result_segment_indices[i]
            mask_row = completion_mask[i]
            outcome_exp = outcome_advantages[i] * torch.ones_like(mask_row, dtype=torch.float32)
            turn_exp = turn_advantages[i] * torch.ones_like(mask_row, dtype=torch.float32)
            combined_exp = combined_advantages[i] * torch.ones_like(mask_row, dtype=torch.float32)
            if result_seg > 0:
                boundaries = get_boundaries(mask_row)
                if result_seg < len(boundaries):
                    split_point = boundaries[result_seg]
                    before_mask = (torch.arange(seq_len, device=device) < split_point).float() * mask_row
                    assigned[i] = outcome_exp + self.turn_advantage_coef * turn_exp * before_mask
                else:
                    assigned[i] = combined_exp
            else:
                assigned[i] = combined_exp
        return assigned

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("MTGRPOEnvTrainer does not support returning outputs")
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
        advantages = inputs["advantages"]
        old_logps = inputs["old_per_token_logps"] if self.num_iterations > 1 else per_token_logps.detach()
        coef_1 = torch.exp(per_token_logps - old_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon, 1 + self.epsilon)
        if len(advantages.shape) == 2:
            loss1, loss2 = coef_1 * advantages, coef_2 * advantages
        else:
            loss1, loss2 = coef_1 * advantages.unsqueeze(1), coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(loss1, loss2)
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl
        loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()
        mode = "eval" if self.control.should_evaluate else "train"
        if self.beta != 0.0:
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
            self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
        clip_ratio = ((loss1 < loss2).float() * completion_mask).sum() / completion_mask.sum()
        self._metrics[mode]["clip_ratio"].append(self.accelerator.gather_for_metrics(clip_ratio).mean().item())
        return loss

    def _log_completion_samples(self, prompts, completions, rewards):
        prompts_to_log = gather_object(prompts)
        completions_to_log = gather_object(completions)
        rewards_list = self.accelerator.gather_for_metrics(rewards).tolist()
        if self.accelerator.is_main_process:
            logger.info(f"\n{'='*100}\nSTEP {self.state.global_step} - COMPLETION SAMPLES\n{'='*100}")
            for idx in range(min(2, len(completions_to_log))):
                reward = rewards_list[idx] if idx < len(rewards_list) else 0.0
                logger.info(f"\n{'─'*80}\nSAMPLE {idx+1} | Reward: {reward:.4f}\n{'─'*80}")
                completion = completions_to_log[idx]
                if isinstance(completion, list):
                    for msg in completion:
                        role = msg.get('role', 'unknown').upper()
                        content = msg.get('content', '')[:400]
                        logger.info(f"[{role}]: {content}")
            logger.info(f"\nMean: {sum(rewards_list)/len(rewards_list):.4f}, Max: {max(rewards_list):.4f}, Min: {min(rewards_list):.4f}\n{'='*100}")


# =============================================================================
# TOOL AND REWARD FUNCTION LOADING
# =============================================================================

def load_tool_functions_from_file(file_path: str) -> List[Callable]:
    """
    Load tool functions from a Python file.

    Args:
        file_path: Path to Python file containing TOOL_FUNCTIONS list

    Returns:
        List of callable tool functions
    """
    logger.info(f"Loading tool functions from file: {file_path}")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Tool functions file not found: {file_path}")

    spec = importlib.util.spec_from_file_location("custom_tools", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, 'TOOL_FUNCTIONS'):
        raise ValueError(
            f"File '{file_path}' does not have a TOOL_FUNCTIONS list. "
            "Please define TOOL_FUNCTIONS = [func1, func2, ...] in your file."
        )

    tool_functions = module.TOOL_FUNCTIONS
    if not isinstance(tool_functions, list) or not tool_functions:
        raise ValueError(f"TOOL_FUNCTIONS in {file_path} must be a non-empty list")

    logger.info(f"Loaded {len(tool_functions)} tool functions from {file_path}")
    return tool_functions


def load_reward_functions_from_file(file_path: str) -> Tuple[List[Callable], List[Callable]]:
    """
    Load turn and outcome reward functions from a Python file.

    Args:
        file_path: Path to Python file containing TURN_REWARD_FUNCS and OUTCOME_REWARD_FUNCS lists

    Returns:
        Tuple of (turn_reward_funcs, outcome_reward_funcs)
    """
    logger.info(f"Loading reward functions from file: {file_path}")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Reward functions file not found: {file_path}")

    spec = importlib.util.spec_from_file_location("custom_rewards", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, 'TURN_REWARD_FUNCS') or not hasattr(module, 'OUTCOME_REWARD_FUNCS'):
        raise ValueError(
            f"File '{file_path}' must have both TURN_REWARD_FUNCS and OUTCOME_REWARD_FUNCS lists."
        )

    turn_funcs = module.TURN_REWARD_FUNCS
    outcome_funcs = module.OUTCOME_REWARD_FUNCS

    if not isinstance(turn_funcs, list) or not isinstance(outcome_funcs, list):
        raise ValueError("TURN_REWARD_FUNCS and OUTCOME_REWARD_FUNCS must be lists")

    logger.info(f"Loaded {len(turn_funcs)} turn reward functions, {len(outcome_funcs)} outcome reward functions")
    return turn_funcs, outcome_funcs


# =============================================================================
# DATASET LOADING
# =============================================================================

def format_prompt(question: str, system_prompt: str) -> List[Dict[str, str]]:
    """Format a question into chat messages."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]


def load_triviaqa_dataset(system_prompt: str, split: str = "train", max_samples: Optional[int] = None) -> Dataset:
    """
    Load TriviaQA dataset formatted for MT-GRPO training.

    Args:
        system_prompt: System prompt to use for formatting
        split: Dataset split to load
        max_samples: Maximum number of samples (None for all)

    Returns:
        Formatted dataset
    """
    logger.info(f"Loading TriviaQA dataset (split={split})")

    ds = load_dataset("trivia_qa", "rc.wikipedia", split=split)

    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    def process_example(example):
        question = example["question"]
        answers = example["answer"]["aliases"] if example["answer"] else []
        return {
            "prompt": format_prompt(question, system_prompt),
            "answer": answers,
        }

    ds = ds.map(process_example, remove_columns=ds.column_names)
    logger.info(f"Loaded {len(ds)} samples from TriviaQA")
    return ds


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def main():
    """Main entry point using TrlParser for YAML configuration."""

    # Parse arguments
    parser = TrlParser((ModelConfig, ScriptArguments, GRPOConfig))
    model_args, script_args, training_args = parser.parse_args_and_config()

    logger.info("=" * 70)
    logger.info("Multi-Turn GRPO Training (SageMaker)")
    logger.info("=" * 70)
    logger.info(f"Model: {model_args.model_name_or_path}")
    logger.info(f"Tools script: {script_args.tools_script}")
    logger.info(f"Reward function: {script_args.reward_fn}")
    logger.info(f"Max env steps: {script_args.max_env_steps}")
    logger.info(f"Turn advantage coef: {script_args.turn_advantage_coef}")
    logger.info("=" * 70)

    set_seed(training_args.seed)

    # Load tool functions
    if script_args.tools_script:
        tool_functions = load_tool_functions_from_file(script_args.tools_script)
    else:
        raise ValueError("--tools_script is required for MT-GRPO training")

    # Load reward functions
    if script_args.reward_fn:
        turn_reward_funcs, outcome_reward_funcs = load_reward_functions_from_file(script_args.reward_fn)
    else:
        raise ValueError("--reward_fn is required for MT-GRPO training")

    logger.info(f"Turn reward functions: {[f.__name__ for f in turn_reward_funcs]}")
    logger.info(f"Outcome reward functions: {[f.__name__ for f in outcome_reward_funcs]}")

    # Create environment
    env = MultiTurnToolEnv(
        tools=tool_functions,
        system_prompt=SYSTEM_PROMPT_TEMPLATE,
        max_steps=script_args.max_env_steps,
    )

    # Load dataset
    dataset = load_triviaqa_dataset(env.system_prompt)

    # Load tokenizer
    logger.info(f"Loading tokenizer from {model_args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Load model
    logger.info(f"Loading model from {model_args.model_name_or_path}")
    torch_dtype = getattr(torch, model_args.torch_dtype) if model_args.torch_dtype not in ["auto", None] else "auto"

    from distutils.util import strtobool
    use_deepspeed = strtobool(os.environ.get("ACCELERATE_USE_DEEPSPEED", "false"))

    model_kwargs = {
        "revision": model_args.model_revision,
        "trust_remote_code": model_args.trust_remote_code,
        "attn_implementation": model_args.attn_implementation,
        "torch_dtype": torch_dtype,
    }
    if not use_deepspeed:
        model_kwargs["low_cpu_mem_usage"] = True

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs,
    )

    # Configure training args - disable TRL's vLLM since we handle it ourselves
    training_args.use_vllm = False

    # Initialize trainer
    trainer = MTGRPOEnvTrainer(
        model=model,
        env=env,
        turn_reward_funcs=turn_reward_funcs,
        outcome_reward_funcs=outcome_reward_funcs,
        turn_advantage_coef=script_args.turn_advantage_coef,
        vllm_server_host=training_args.vllm_server_host,
        vllm_server_port=training_args.vllm_server_port,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=dataset,
    )

    # Check for checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint:
            logger.info(f"Resuming from checkpoint: {last_checkpoint}")

    # Train
    start_time = datetime.now()
    logger.info(f"Training started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)

    # Save metrics
    metrics = train_result.metrics
    metrics['train_samples'] = len(dataset)
    trainer.log_metrics('train', metrics)
    trainer.save_metrics('train', metrics)
    trainer.save_state()

    # Save model
    trainer.accelerator.wait_for_everyone()

    if "SM_MODEL_DIR" in os.environ:
        final_model_dir = os.path.join(os.environ["SM_MODEL_DIR"], model_args.model_name_or_path)
    else:
        final_model_dir = os.path.join(training_args.output_dir, "final_model")

    logger.info(f"Saving model to {final_model_dir}")
    trainer.save_model(final_model_dir)

    if trainer.accelerator.is_main_process:
        tokenizer.save_pretrained(final_model_dir)

    end_time = datetime.now()
    logger.info(f"Training completed in {end_time - start_time}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
