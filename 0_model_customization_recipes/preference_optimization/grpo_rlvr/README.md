# GRPO + RLVR for Tool Calling

Train language models to reliably call tools and functions using **GRPO (Group Relative Policy Optimization)** with **RLVR (Reinforcement Learning with Verifiable Rewards)**.

## Overview

This recipe enables training models to:
- Generate properly formatted tool calls with correct arguments
- Execute tools and process their responses
- Learn from verifiable rewards based on tool execution outcomes

The training loop:
1. Model generates tool calls from prompts
2. Tools are executed automatically via `run_tool`
3. Reward function evaluates outcomes
4. Model is optimized using GRPO

## Quick Start

```bash
cd sagemaker_code
bash sm_accelerate_grpo_train.sh
```

This uses the default financial tools and accuracy reward function.

---

## Custom Tool Functions

### Structure

Tool functions must:
- Accept typed parameters with clear docstrings
- Return simple string results (for easy comparison with expected answers)
- Be pure functions (deterministic, no side effects)

### Example

```python
def calculate_sum(a: float, b: float) -> str:
    """
    Add two numbers together.
    
    Args:
        a: First number
        b: Second number
    
    Returns:
        String with the sum
    """
    return f"Sum: {a + b}"
```

### Creating Your Tools

1. **Create a Python file** in `sagemaker_code/tools_funcs/`:
   ```bash
   touch sagemaker_code/tools_funcs/my_tools.py
   ```

2. **Define your functions** with clear signatures and docstrings

3. **Export as a list** at the end of the file:
   ```python
   TOOL_FUNCTIONS = [
       calculate_sum,
       calculate_product,
       calculate_average,
   ]
   ```

4. **Use in training** by passing the path:
   ```bash
   bash sm_accelerate_grpo_train.sh \
       --tools_script "${SCRIPT_DIR}/tools_funcs/my_tools.py"
   ```

### Tool Execution

The trainer automatically:
- Converts your functions into tool schemas (name, parameters, descriptions)
- Provides these schemas to the model during generation
- Executes tool calls via `run_tool` when the model generates them
- Passes tool responses back to the model

The `run_tool` function:
- Accepts tool calls as JSON: `{"name": "function_name", "arguments": {...}}`
- Validates the tool name and arguments
- Executes the function and returns the string result
- Handles errors gracefully with descriptive messages

---

## Custom Reward Functions

### Structure

Reward functions must have this signature:

```python
def reward_func(
    completions: List[List[Dict]], 
    answer: List[str], 
    **kwargs
) -> List[float]:
    """
    Calculate rewards for a batch of completions.
    
    Args:
        completions: List of completion message lists, each containing:
                     [{"role": "assistant", "content": "..."}, 
                      {"role": "tool", "content": "..."}]
        answer: List of expected answers (ground truth)
        **kwargs: Additional context from dataset
    
    Returns:
        List of reward values (typically 0.0 to 1.0)
    """
```

### Example: Accuracy Reward

```python
def accuracy_reward(completions: List[List[Dict]], answer: List[str], **kwargs) -> List[float]:
    """Simple exact-match reward."""
    rewards = []
    
    for completion, expected in zip(completions, answer):
        # Extract tool response from completion
        tool_response = None
        for message in completion:
            if message.get("role") == "tool":
                tool_response = message.get("content", "")
        
        # Assign reward
        if tool_response is None:
            reward = 0.0  # No tool call
        elif expected.lower().strip() == tool_response.lower().strip():
            reward = 1.0  # Exact match
        else:
            reward = 0.1  # Attempted but incorrect
        
        rewards.append(reward)
    
    return rewards
```

### Creating Your Reward Function

1. **Create a Python file** in `sagemaker_code/rewards/`:
   ```bash
   touch sagemaker_code/rewards/my_reward.py
   ```

2. **Define your reward function** with the required signature

3. **Export as `reward_func`**:
   ```python
   reward_func = my_custom_reward
   ```

4. **Use in training**:
   ```bash
   bash sm_accelerate_grpo_train.sh \
       --reward_fn "${SCRIPT_DIR}/rewards/my_reward.py"
   ```

### Reward Design Tips

- **Sparse rewards** (0.0 or 1.0): Simple but can be slow to learn
- **Dense rewards** (0.0 to 1.0): Provide intermediate feedback
  - Partial credit for correct tool selection
  - Partial credit for correct argument types
  - Full credit for correct final answer
- **Use dataset context**: Access additional fields via `**kwargs`
- **Log statistics**: Print mean/std of rewards for monitoring

---

## Training Configuration

### Key Parameters

Edit `sm_accelerate_grpo_train.sh` to customize:

```bash
# Tools and rewards
--tools_script "${SCRIPT_DIR}/tools_funcs/my_tools.py" \
--reward_fn "${SCRIPT_DIR}/rewards/my_reward.py" \

# Dataset
--dataset_id_or_path "path/to/dataset.jsonl" \

# Model
--model_name_or_path "Qwen/Qwen2.5-0.5B-Instruct" \

# Training
--num_train_epochs 3 \
--per_device_train_batch_size 2 \
--gradient_accumulation_steps 8 \
--learning_rate 5e-7 \

# Generation
--max_grpo_completion_length 512 \
```

### Dataset Format

Your dataset should be JSONL with:
- `prompt`: The user query/task
- `answer`: Expected tool response (for reward calculation)

Example:
```json
{"prompt": "What is 15 + 27?", "answer": "Sum: 42"}
{"prompt": "Calculate the product of 8 and 9", "answer": "Product: 72"}
```

---

## Examples

### Example 1: Math Tools

**tools_funcs/math_tools.py:**
```python
def add(a: float, b: float) -> str:
    """Add two numbers."""
    return f"Sum: {a + b}"

def multiply(a: float, b: float) -> str:
    """Multiply two numbers."""
    return f"Product: {a * b}"

TOOL_FUNCTIONS = [add, multiply]
```

**rewards/math_reward.py:**
```python
def math_reward(completions, answer, **kwargs):
    rewards = []
    for completion, expected in zip(completions, answer):
        tool_response = None
        for msg in completion:
            if msg.get("role") == "tool":
                tool_response = msg.get("content")
        
        if tool_response == expected:
            rewards.append(1.0)
        elif tool_response is not None:
            rewards.append(0.3)  # Partial credit for trying
        else:
            rewards.append(0.0)
    
    return rewards

reward_func = math_reward
```

**Train:**
```bash
bash sm_accelerate_grpo_train.sh \
    --tools_script "${SCRIPT_DIR}/tools_funcs/math_tools.py" \
    --reward_fn "${SCRIPT_DIR}/rewards/math_reward.py" \
    --dataset_id_or_path "data/math_problems.jsonl"
```

### Example 2: Using Built-in Accuracy Reward

```bash
bash sm_accelerate_grpo_train.sh \
    --tools_script "${SCRIPT_DIR}/tools_funcs/my_tools.py" \
    --reward_function_name "accuracy"
```

---

## Advanced Topics

### Multi-Step Tool Calling

For tasks requiring multiple tool calls:
1. Structure your reward to evaluate the full conversation
2. Check that all necessary tools were called
3. Verify the final answer incorporates all tool responses

### Tool Call Validation

Add validation in your reward function:
```python
def validated_reward(completions, answer, **kwargs):
    rewards = []
    for completion, expected in zip(completions, answer):
        # Check if correct tool was called
        tool_name = None
        tool_response = None
        
        for msg in completion:
            if msg.get("role") == "assistant":
                # Parse tool call from assistant message
                # (implementation depends on your format)
                pass
            if msg.get("role") == "tool":
                tool_response = msg.get("content")
        
        # Reward based on tool selection + response quality
        if tool_name == expected_tool and tool_response == expected:
            reward = 1.0
        elif tool_name == expected_tool:
            reward = 0.5  # Right tool, wrong result
        else:
            reward = 0.0
        
        rewards.append(reward)
    
    return rewards
```

### Debugging

Enable detailed logging in the trainer:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

The trainer prints:
- Loaded tool functions with signatures
- Reward function details
- Per-batch reward statistics

---

## Reference

### Default Files

- **Tools**: `sagemaker_code/tools_funcs/financial_tools_complex.py`
  - 8 complex financial calculation functions
  - Demonstrates comprehensive argument structures
  
- **Reward**: `sagemaker_code/rewards/financial_tools_reward.py`
  - Accuracy-based reward with partial credit
  - Includes tool selection validation

### Training Script

`sm_accelerate_grpo_train.sh` supports:
- `--tools_script PATH`: Custom tool functions file
- `--reward_fn PATH`: Custom reward function file
- `--tool_functions_module MODULE`: Python module path (alternative to file)
- `--reward_function_name NAME`: Built-in reward name (alternative to file)

### Trainer Details

The `GRPOTrainer` (in `grpo_trainer_v2.py`):
- Dynamically loads tools and rewards at runtime
- Converts functions to tool schemas automatically
- Executes tools during generation via `run_tool`
- Supports distributed training with DeepSpeed/Accelerate
- Handles checkpoint resumption and model saving

---

## Tips

1. **Start simple**: Begin with 2-3 tools and exact-match rewards
2. **Iterate on rewards**: Experiment with dense rewards for faster learning
3. **Validate tools**: Test your tool functions independently before training
4. **Monitor rewards**: Watch mean reward per batch to track learning
5. **Use clear docstrings**: The model sees your function docstrings as tool descriptions

---

## Troubleshooting

**"No TOOL_FUNCTIONS found"**
- Ensure your file exports `TOOL_FUNCTIONS = [...]`

**"reward_func must be callable"**
- Ensure your file exports `reward_func = your_function`

**Low rewards throughout training**
- Check that expected answers match tool output format exactly
- Try dense rewards with partial credit
- Verify tools are being called (check logs)

**Model not calling tools**
- Increase `max_grpo_completion_length`
- Check that prompts clearly indicate tool usage
- Verify tool schemas are being generated correctly

---

For more examples, see the `financial_tools_complex.py` and `financial_tools_reward.py` files.
