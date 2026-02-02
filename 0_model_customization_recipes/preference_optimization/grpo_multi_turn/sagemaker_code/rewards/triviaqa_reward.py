"""
TriviaQA Multi-Turn Reward Functions for MT-GRPO Training

Provides turn-level and outcome-level reward functions for tool-calling agents.

Usage:
    Pass this file path to --reward_fn when running training:
    ./sm_mt_grpo_train.sh --config <config.yaml> --reward_fn rewards/triviaqa_reward.py
"""

import re
from typing import Dict, List, Optional, Any
from types import SimpleNamespace


class XMLParser:
    def __init__(self, fields):
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


PARSER = XMLParser(fields=["reasoning", ("tool", "answer")])
ENV_PARSER = XMLParser(fields=["result"])


def get_last_answer(trajectory: List[Dict[str, str]]) -> Optional[str]:
    for msg in reversed(trajectory):
        if msg['role'] == 'assistant':
            parsed = PARSER.parse(msg['content'])
            if hasattr(parsed, 'answer') and parsed.answer is not None:
                return parsed.answer
    return None


def get_search_results(trajectory: List[Dict[str, str]]) -> Optional[str]:
    for i, msg in enumerate(trajectory):
        if msg['role'] == 'assistant':
            parsed = PARSER.parse(msg['content'])
            if hasattr(parsed, 'tool') and parsed.tool is not None:
                if i + 1 < len(trajectory) and trajectory[i + 1]['role'] == 'user':
                    env_parsed = ENV_PARSER.parse(trajectory[i + 1]['content'])
                    if hasattr(env_parsed, 'result') and env_parsed.result:
                        return env_parsed.result
    return None


def tool_execution_reward_func(completions: List[List[Dict]], **kwargs) -> List[float]:
    def check_execution(trajectory):
        tool_attempts, successful = 0, 0
        for i, msg in enumerate(trajectory):
            if msg['role'] == 'assistant':
                parsed = PARSER.parse(msg['content'])
                if hasattr(parsed, 'tool') and parsed.tool is not None:
                    if i + 1 < len(trajectory) and trajectory[i + 1]['role'] == 'user':
                        tool_attempts += 1
                        resp = ENV_PARSER.parse(trajectory[i + 1]['content'])
                        if hasattr(resp, 'result') and resp.result and not resp.result.startswith("Error:"):
                            successful += 1
        return 0.2 * (successful / tool_attempts) if tool_attempts > 0 else 0.0
    return [check_execution(c) for c in completions]


def exist_answer_in_search_results(completions, answer, **kwargs) -> List[float]:
    rewards = []
    for c, a_list in zip(completions, answer):
        search_result = get_search_results(c)
        if search_result is None:
            rewards.append(0.0)
        else:
            reward = 0.5 if any(str(a).lower() in search_result.lower() for a in a_list) else 0.0
            rewards.append(reward)
    return rewards


def exist_answer_reward_func(completions, answer, **kwargs) -> List[float]:
    rewards = []
    for c, a_list in zip(completions, answer):
        r = get_last_answer(c)
        if r is None:
            rewards.append(0.0)
        else:
            reward = 0.5 if any(str(a).lower() in str(r).lower() for a in a_list) else 0.0
            rewards.append(reward)
    return rewards


def exact_match_reward_func(completions, answer, **kwargs) -> List[float]:
    rewards = []
    for c, a_list in zip(completions, answer):
        r = get_last_answer(c)
        if r is None:
            rewards.append(0.0)
        else:
            r_lower = str(r).lower().strip()
            reward = 1.0 if any(str(a).lower().strip() == r_lower for a in a_list) else 0.0
            rewards.append(reward)
    return rewards


def format_reward_func(completions, **kwargs) -> List[float]:
    def check_format(trajectory):
        model_messages = [msg for msg in trajectory if msg['role'] == 'assistant']
        if not model_messages:
            return 0.0
        format_scores = []
        for msg in model_messages:
            content = msg['content']
            parsed = PARSER.parse(content)
            present = sum(1 for f, alts in PARSER._fields for alt in alts if hasattr(parsed, alt) and getattr(parsed, alt))
            format_score = 0.4 * present / len(PARSER._fields) if PARSER._fields else 0
            if content.strip().startswith("<reasoning>"):
                format_score += 0.3
            if content.strip().endswith("</answer>") or content.strip().endswith("</tool>"):
                format_score += 0.3
            format_scores.append(format_score)
        return 0.2 * (sum(format_scores) / len(format_scores)) if format_scores else 0.0
    return [check_format(c) for c in completions]


def xml_reward_func(completions, **kwargs) -> List[float]:
    def count_xml(trajectory):
        model_messages = [msg for msg in trajectory if msg['role'] == 'assistant']
        if not model_messages:
            return 0.0
        xml_scores = []
        for msg in model_messages:
            content = msg['content']
            score, total_checks = 0, 0
            for _, alternatives in PARSER._fields:
                for alt in alternatives:
                    if content.count(f"<{alt}>") > 0 or content.count(f"</{alt}>") > 0:
                        score += 1 - abs(content.count(f"<{alt}>") - 1)
                        score += 1 - abs(content.count(f"</{alt}>") - 1)
                        total_checks += 2
            xml_scores.append(score / total_checks if total_checks > 0 else 0.0)
        return max(0.0, 0.2 * (sum(xml_scores) / len(xml_scores))) if xml_scores else 0.0
    return [count_xml(c) for c in completions]


TURN_REWARD_FUNCS = [tool_execution_reward_func, exist_answer_in_search_results]
OUTCOME_REWARD_FUNCS = [exist_answer_reward_func, exact_match_reward_func, format_reward_func, xml_reward_func]
