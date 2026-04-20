"""
Simplified OpenVLA utilities extracted from the official repository.
This allows training without installing the full OpenVLA package.
"""
import numpy as np
import torch
from typing import List, Tuple


class ActionTokenizer:
    """
    Tokenizes continuous actions into discrete tokens for OpenVLA.
    OpenVLA uses the last 256 tokens in the vocabulary for discretized actions.
    """

    def __init__(self, tokenizer, bins: int = 256, min_action: float = -1.0, max_action: float = 1.0):
        self.tokenizer = tokenizer
        self.bins = bins
        self.min_action = min_action
        self.max_action = max_action
        vocab_size = len(tokenizer)
        self.action_token_begin_idx = vocab_size - bins

    def tokenize(self, actions: np.ndarray) -> List[int]:
        """Convert continuous actions to discrete token IDs."""
        actions = np.clip(np.array(actions), self.min_action, self.max_action)
        normalized = (actions - self.min_action) / (self.max_action - self.min_action)
        bins = (normalized * (self.bins - 1)).astype(np.int64)
        token_ids = self.action_token_begin_idx + bins
        return token_ids.tolist()

    def decode_token_ids_to_actions(self, token_ids: np.ndarray) -> np.ndarray:
        """Convert token IDs back to continuous actions."""
        bins = token_ids - self.action_token_begin_idx
        normalized = bins.astype(np.float32) / (self.bins - 1)
        actions = normalized * (self.max_action - self.min_action) + self.min_action
        return actions


class PurePromptBuilder:
    """
    Simple prompt builder for OpenVLA.
    Format: "In: What action should the robot take to {instruction}?\nOut:"
    """

    def __init__(self, model_family: str):
        self.model_family = model_family

    def build_prompt(self, instruction: str, output: str = "") -> str:
        prompt = f"In: What action should the robot take to {instruction}?\nOut:"
        if output:
            prompt += f" {output}"
        return prompt


class VicunaV15ChatPromptBuilder:
    """Vicuna v1.5 chat prompt builder for OpenVLA v01 models."""

    def __init__(self, model_family: str):
        self.model_family = model_family

    def build_prompt(self, instruction: str, output: str = "") -> str:
        prompt = (
            f"A chat between a curious user and an artificial intelligence assistant. "
            f"The assistant gives helpful, detailed, and polite answers to the user's questions. "
            f"USER: What action should the robot take to {instruction}? ASSISTANT:"
        )
        if output:
            prompt += f" {output}"
        return prompt
