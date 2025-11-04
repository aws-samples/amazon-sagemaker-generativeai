"""
Custom Reward Functions for SAMA RL Training

This file contains example reward functions that can be used with SAMA RL
for training language models with reinforcement learning (GRPO, PPO, etc.).

Usage:
    from user.reward_functions import length_reward, helpfulness_reward
    
    trainer = GRPO(
        yaml_file="sama_rl/recipes/GRPO/qwen2-0.5b-grpo-config.yaml",
        reward_functions=[length_reward, helpfulness_reward],
        max_steps=100,
        instance_type="ml.p5.48xlarge"
    )
"""

def length_reward(completions, **kwargs):
    """
    Reward function that encourages responses of a target length.
    
    Args:
        completions (List[str]): List of model completions
        **kwargs: Additional arguments (tokenizer, etc.)
    
    Returns:
        List[float]: Reward scores for each completion
    """
    target_length = 400  # Target number of tokens
    tokenizer = kwargs.get('tokenizer')
    rewards = []
    
    for completion in completions:
        if tokenizer:
            num_tokens = len(tokenizer.encode(completion, add_special_tokens=False))
        else:
            # Fallback to word count approximation
            num_tokens = len(completion.split())
        
        # Negative quadratic penalty for deviation from target length
        reward = -(abs(num_tokens - target_length) ** 2) / 1000
        rewards.append(reward)
    
    return rewards


def helpfulness_reward(completions, **kwargs):
    """
    Reward function that encourages helpful and informative responses.
    
    Args:
        completions (List[str]): List of model completions
        **kwargs: Additional arguments
    
    Returns:
        List[float]: Reward scores for each completion
    """
    rewards = []
    
    # Keywords that indicate helpfulness
    helpful_keywords = [
        'explain', 'because', 'reason', 'example', 'specifically', 
        'detail', 'step', 'process', 'method', 'approach'
    ]
    
    for completion in completions:
        completion_lower = completion.lower()
        
        # Base reward
        reward = 0.0
        
        # Reward for helpful keywords
        keyword_count = sum(1 for keyword in helpful_keywords if keyword in completion_lower)
        reward += keyword_count * 0.1
        
        # Reward for reasonable length (not too short, not too long)
        length = len(completion.split())
        if 20 <= length <= 200:
            reward += 0.2
        elif length < 10:
            reward -= 0.3  # Penalty for very short responses
        
        # Penalty for repetitive content
        words = completion_lower.split()
        if len(words) > 0:
            unique_words = len(set(words))
            repetition_ratio = unique_words / len(words)
            if repetition_ratio < 0.7:  # High repetition
                reward -= 0.2
        
        rewards.append(reward)
    
    return rewards


def conciseness_reward(completions, **kwargs):
    """
    Reward function that encourages concise but complete responses.
    
    Args:
        completions (List[str]): List of model completions
        **kwargs: Additional arguments
    
    Returns:
        List[float]: Reward scores for each completion
    """
    rewards = []
    
    for completion in completions:
        word_count = len(completion.split())
        
        # Optimal range: 50-150 words
        if 50 <= word_count <= 150:
            reward = 0.5
        elif 30 <= word_count < 50 or 150 < word_count <= 200:
            reward = 0.2
        elif word_count < 30:
            reward = -0.3  # Too short
        else:
            reward = -0.1 * (word_count - 200) / 100  # Penalty for being too long
        
        rewards.append(reward)
    
    return rewards


def safety_reward(completions, **kwargs):
    """
    Reward function that penalizes unsafe or harmful content.
    
    Args:
        completions (List[str]): List of model completions
        **kwargs: Additional arguments
    
    Returns:
        List[float]: Reward scores for each completion
    """
    rewards = []
    
    # Simple keyword-based safety check (in practice, use more sophisticated methods)
    unsafe_keywords = [
        'violence', 'harm', 'illegal', 'dangerous', 'weapon', 
        'drug', 'suicide', 'hate', 'discrimination'
    ]
    
    for completion in completions:
        completion_lower = completion.lower()
        
        # Start with neutral reward
        reward = 0.0
        
        # Check for unsafe content
        unsafe_count = sum(1 for keyword in unsafe_keywords if keyword in completion_lower)
        if unsafe_count > 0:
            reward -= unsafe_count * 0.5  # Strong penalty for unsafe content
        else:
            reward += 0.1  # Small bonus for safe content
        
        rewards.append(reward)
    
    return rewards


def combined_reward(completions, **kwargs):
    """
    Combined reward function that uses multiple criteria.
    
    Args:
        completions (List[str]): List of model completions
        **kwargs: Additional arguments
    
    Returns:
        List[float]: Reward scores for each completion
    """
    # Get individual reward components
    length_rewards = length_reward(completions, **kwargs)
    helpfulness_rewards = helpfulness_reward(completions, **kwargs)
    safety_rewards = safety_reward(completions, **kwargs)
    
    # Combine with weights
    combined_rewards = []
    for i in range(len(completions)):
        combined = (
            0.3 * length_rewards[i] +
            0.4 * helpfulness_rewards[i] +
            0.3 * safety_rewards[i]
        )
        combined_rewards.append(combined)
    
    return combined_rewards


# Example of how to use multiple reward functions
EXAMPLE_REWARD_FUNCTIONS = [
    length_reward,
    helpfulness_reward,
    safety_reward
]
