# Multi-Turn GRPO Training

This document explains how Multi-Turn Reinforcement Learning differs from Single-Turn RL, and walks through how `mt_grpo_trainer.py` implements it.

## The Story: Why Multi-Turn Matters

Imagine you're training a model to answer trivia questions using Wikipedia search. In **Single-Turn RL**, the model gets one shot:

```
User: "Who wrote Romeo and Juliet?"
Assistant: "William Shakespeare"  ← reward assigned here
```

The model either knows the answer or it doesn't. Simple.

But what if the model needs to *use a tool* to find the answer? Now we need **Multi-Turn RL**:

```
User: "What year was the Eiffel Tower completed?"
Assistant: <reasoning>I should search for this</reasoning>
           <tool>{"name": "wiki_search", "args": {"query": "Eiffel Tower construction"}}</tool>
User: <result>The Eiffel Tower was completed in 1889...</result>
Assistant: <reasoning>The search says 1889</reasoning>
           <answer>1889</answer>  ← reward assigned here
```

The challenge: how do you assign credit? The final answer was correct, but the *search query* was what made it possible. Single-Turn RL can't handle this because it only sees one generation step.

## How Generation Works

### The vLLM Server

Training large models is expensive. Running inference on the same GPUs that are doing gradient updates is wasteful. So we split the work:

- **GPUs 0-6**: Run the training (forward/backward passes, gradient updates)
- **GPU 7**: Run a vLLM server for fast inference

The `VLLMServerClient` class in `mt_grpo_trainer.py` connects to this server:

```python
class VLLMServerClient:
    def __init__(self, host: str, port: int, tokenizer: Any):
        self.base_url = f"http://{host}:{port}"
        # ...
    
    def chat(self, messages_list, sampling_params, ...):
        # Sends requests to vLLM's /v1/chat/completions endpoint
        # Returns generated text + token IDs
```

When training starts, the main process (rank 0) sends prompts to vLLM, gets completions back, then broadcasts them to all other training processes.

### The Multi-Turn Loop

Here's where it gets interesting. The `MultiTurnToolEnv` class manages the conversation loop:

```python
class MultiTurnToolEnv:
    def __init__(self, tools, system_prompt, max_steps=2, ...):
        self.max_steps = max_steps  # How many tool calls allowed
        self.tools = {tool.__name__: tool for tool in tools}
```

The `generate()` method runs the multi-turn loop:

```python
def generate(self, prompts, llm, sampling_params):
    # Initialize state for each prompt
    states = [{"messages": list(m), "completed": False, ...} for m in prompts]
    
    # Keep stepping until all conversations are done
    while not all(s["completed"] for s in states):
        states = self.step(states, llm, sampling_params)
    
    return {"ids": [...], "messages": [...], "mask": [...]}
```

Each `step()` does:

1. **Generate**: Ask vLLM to continue the conversation
2. **Parse**: Check if the model output a `<tool>` or `<answer>` tag
3. **Execute**: If it's a tool call, run the tool and add the result
4. **Track**: Record which tokens came from the model vs the environment

The key insight is the **completion mask**. For each token in the final sequence:
- `1` = model generated this (trainable)
- `0` = environment generated this (not trainable)

```python
# In step():
state["completion_mask"].extend([self.env_mask] * env_response_len)  # Tool results: mask=0
state["completion_mask"].extend([1] * new_completion_len)            # Model output: mask=1
```

This mask ensures we only compute gradients on tokens the model actually produced.

## The Reward Problem

In Single-Turn RL, you have one reward per completion. Easy.

In Multi-Turn RL, you have a sequence of actions:
1. Model writes a search query (Turn 1)
2. Model reads the result and writes an answer (Turn 2)

Which action deserves credit? Both? Just the final one?

### Turn vs Outcome Rewards

`MTGRPOEnvTrainer` uses two types of reward functions:

```python
class MTGRPOEnvTrainer(GRPOTrainer):
    def __init__(self, ..., turn_reward_funcs, outcome_reward_funcs, ...):
        self.turn_reward_funcs = turn_reward_funcs      # Rewards for intermediate steps
        self.outcome_reward_funcs = outcome_reward_funcs # Rewards for final answer
```

**Turn rewards** (from `rewards/triviaqa_reward.py`):
- `tool_execution_reward_func`: Did the tool call succeed without errors?
- `exist_answer_in_search_results`: Does the search result contain the answer?

**Outcome rewards**:
- `exact_match_reward_func`: Is the final answer exactly correct?
- `format_reward_func`: Did the model follow the XML format?

### Position-Aware Advantage Assignment

Here's the clever part. The trainer finds where the tool result appears in the sequence:

```python
def _find_result_positions(self, completion_messages):
    # Find the message index where <result> appears
    for j, msg in enumerate(messages):
        if msg.get('role') == 'assistant':
            if j + 1 < len(messages) and '<result>' in messages[j + 1].get('content', ''):
                result_segment = j + 1
                break
```

Then it assigns different advantages to different parts of the sequence:

```python
def _assign_advantages(self, completion_mask, turn_advantages, outcome_advantages, ...):
    # Before the tool result: use turn_advantage (was the query good?)
    # After the tool result: use outcome_advantage (was the answer correct?)
    
    if result_seg > 0:
        before_mask = (torch.arange(seq_len) < split_point).float()
        assigned[i] = outcome_exp + self.turn_advantage_coef * turn_exp * before_mask
```

This means:
- Tokens before `<result>` get credit for making a good tool call
- Tokens after `<result>` get credit for producing the correct answer

The `turn_advantage_coef` (configurable in YAML) controls how much weight to give the turn-level rewards.

## Putting It All Together

Here's the full training flow:

1. **Load prompts** from TriviaQA dataset
2. **Generate completions** via vLLM (multi-turn loop)
3. **Compute rewards** using turn and outcome functions
4. **Assign advantages** based on position in the conversation
5. **Compute loss** using PPO-style clipped objective
6. **Update model** via gradient descent

The loss function in `compute_loss()` is standard PPO:

```python
coef_1 = torch.exp(per_token_logps - old_logps)
coef_2 = torch.clamp(coef_1, 1 - self.epsilon, 1 + self.epsilon)
per_token_loss = -torch.min(coef_1 * advantages, coef_2 * advantages)
```

But the `advantages` tensor is position-aware, so different tokens get different learning signals.

## Configuration

The YAML files in `hf_recipes/Qwen/` control training:

```yaml
# Key multi-turn settings
max_env_steps: 5          # Max tool calls per conversation
turn_advantage_coef: 1.0  # Weight for turn-level rewards
num_generations: 7        # Completions per prompt (for GRPO grouping)

# vLLM server
vllm_server_host: "0.0.0.0"
vllm_server_port: 8000
```

## Summary

| Aspect | Single-Turn RL | Multi-Turn RL |
|--------|---------------|---------------|
| Generation | One model output | Loop until `<answer>` or max steps |
| Reward | One score per completion | Turn rewards + outcome rewards |
| Credit assignment | Uniform across tokens | Position-aware (before/after tool result) |
| Token masking | All tokens trainable | Only model tokens trainable |

The `mt_grpo_trainer.py` implements all of this in a single file, with the `MultiTurnToolEnv` handling the conversation loop and `MTGRPOEnvTrainer` handling the position-aware reward assignment.
