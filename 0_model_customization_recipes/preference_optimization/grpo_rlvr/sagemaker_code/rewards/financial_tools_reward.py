"""
Financial Tools Reward Function for GRPO Training

This module provides a reward function for evaluating tool calling accuracy
in financial tool training scenarios.

Usage:
    Pass this file path to --reward_fn when running training:
    ./sm_accelerate_grpo_train.sh --config <config.yaml> --reward_fn rewards/financial_tools_reward.py
"""

from typing import Dict, List


def reward_func(completions: List[List[Dict]], answer: List[str], **kwargs) -> List[float]:
    """
    Reward function for financial tool calling accuracy.

    Reward values:
    - 0.0: No tool call (tool_response is None)
    - 0.1: Some tool call made (any tool response)
    - 1.0: Exact match with expected answer

    Args:
        completions: List of completion message lists
        answer: List of expected answers

    Returns:
        List of reward values
    """
    rewards = []

    for completion, ans in zip(completions, answer):
        ans_str = str(ans).lower().strip()
        reward = 0.0
        tool_response = None

        for message in completion:
            role = message.get("role", "")
            content = message.get("content", "") or ""

            if role == "tool":
                tool_response = content

        # Assign reward based on tool response
        if tool_response is None:
            reward = 0.0  # No tool call
        elif ans_str == tool_response.lower().strip():
            reward = 1.0  # Exact match
        else:
            reward = 0.1  # Some tool call made

        rewards.append(reward)

    print(f"---------------------------------------------")
    print(f"BATCH: {len(rewards)} completions, Mean: {sum(rewards)/len(rewards):.3f}, Rewards: {rewards}")
    print(f"---------------------------------------------")

    return rewards
