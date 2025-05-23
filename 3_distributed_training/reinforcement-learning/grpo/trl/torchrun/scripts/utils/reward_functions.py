import logging
import numpy as np
import re
from rouge import Rouge
from typing import List, Dict, Any, Union, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize Rouge scorer
rouge = Rouge()


def get_scores(hypotheses: List[str], references: List[str]) -> List[float]:
    """
    Calculate Rouge-L precision scores between hypotheses and references.

    Args:
        hypotheses: List of hypothesis texts
        references: List of reference texts

    Returns:
        List of Rouge-L precision scores
    """
    try:
        return [s["rouge-l"]["p"] for s in rouge.get_scores(hypotheses, references)]
    except Exception as e:
        logger.warning(f"Error calculating Rouge scores: {e}")
        return [0.0] * len(references)


def rouge_reward_func(
    completions: List[Dict[str, Any]], answer: List[str], **kwargs
) -> List[float]:
    """
    Reward function that calculates Rouge-L precision scores between completions and answers.

    Args:
        completions: List of completion dictionaries
        answer: List of reference answers
        **kwargs: Additional keyword arguments

    Returns:
        List of Rouge-L precision scores
    """
    try:
        # Extract completion content
        if isinstance(completions[0], dict) and "content" in completions[0]:
            # Handle case where completions are dictionaries with content field
            completion_texts = [completion["content"] for completion in completions]
        elif (
            isinstance(completions[0], list)
            and isinstance(completions[0][0], dict)
            and "content" in completions[0][0]
        ):
            # Handle case where completions are lists of dictionaries with content field
            completion_texts = [completion[0]["content"] for completion in completions]
        else:
            # Fallback case
            completion_texts = completions

        # Calculate scores
        scores = get_scores(completion_texts, answer)
        return scores
    except Exception as e:
        logger.error(f"Error in rouge_reward_func: {e}")
        return [0.0] * len(answer)


def format_reward_func(completions: List[Dict[str, Any]], **kwargs) -> List[float]:
    """
    Reward function that checks if completions follow specific format patterns.

    Rewards completions that match either of these patterns:
    1. Text enclosed in <think>...</think> tags
    2. Text enclosed in <answer>...</answer> tags

    Each pattern match adds 1.0 to the score, so a completion can score up to 2.0
    if it matches both patterns.

    Args:
        completions: List of completion dictionaries
        **kwargs: Additional keyword arguments

    Returns:
        List of format scores (0.0, 1.0, or 2.0)
    """
    try:
        # Define patterns to match
        pattern1 = r"^<think>.*?</think>$"
        pattern2 = r"^<answer>.*?</answer>$"

        # Extract responses from completions
        if isinstance(completions[0], dict) and "content" in completions[0]:
            # Handle case where completions are dictionaries with content field
            responses = [completion["content"] for completion in completions]
        elif (
            isinstance(completions[0], list)
            and isinstance(completions[0][0], dict)
            and "content" in completions[0][0]
        ):
            # Handle case where completions are lists of dictionaries with content field
            responses = [completion[0]["content"] for completion in completions]
        else:
            # Fallback case
            responses = completions

        # Check for pattern matches
        matches1 = [
            re.match(pattern1, response, re.DOTALL) is not None
            for response in responses
        ]
        matches2 = [
            re.match(pattern2, response, re.DOTALL) is not None
            for response in responses
        ]

        # Convert matches to scores (1.0 for match, 0.0 for no match)
        scores1 = [1.0 if m else 0.0 for m in matches1]
        scores2 = [1.0 if m else 0.0 for m in matches2]

        # Add scores from both patterns
        return list(np.add(scores1, scores2))
    except Exception as e:
        logger.error(f"Error in format_reward_func: {e}")
        return [0.0] * len(completions)


def reward_len(
    completions: List[Dict[str, Any]], target_length: int = 512, **kwargs
) -> List[float]:
    """
    Reward function that scores completions based on their length relative to a target length.

    The score is calculated as min(completion_length, target_length) / target_length,
    which means completions get higher scores as they approach the target length,
    with a maximum score of 1.0 when they reach or exceed the target length.

    Args:
        completions: List of completion dictionaries
        target_length: Target length for completions (default: 512)
        **kwargs: Additional keyword arguments

    Returns:
        List of length-based scores between 0.0 and 1.0
    """
    try:
        # Extract completion content
        if isinstance(completions[0], dict) and "content" in completions[0]:
            # Handle case where completions are dictionaries with content field
            completion_texts = [completion["content"] for completion in completions]
        elif (
            isinstance(completions[0], list)
            and isinstance(completions[0][0], dict)
            and "content" in completions[0][0]
        ):
            # Handle case where completions are lists of dictionaries with content field
            completion_texts = [completion[0]["content"] for completion in completions]
        else:
            # Fallback case
            completion_texts = completions

        # Calculate scores based on length
        return [
            min(len(completion) + 1, target_length) / target_length
            for completion in completion_texts
        ]
    except Exception as e:
        logger.error(f"Error in reward_len: {e}")
        return [0.0] * len(completions)
