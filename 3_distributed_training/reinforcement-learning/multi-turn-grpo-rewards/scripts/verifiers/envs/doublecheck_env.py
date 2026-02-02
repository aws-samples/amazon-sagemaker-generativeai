from typing import List, Dict, Any

from datasets import Dataset
from trl.trainer.grpo_trainer import RewardFunc

from verifiers.envs.multiturn_env import MultiTurnEnv
from verifiers.prompts import SIMPLE_PROMPT, DOUBLECHECK_FEW_SHOT
from verifiers.rubrics import MathRubric
from verifiers.utils import preprocess_dataset


class DoubleCheckEnv(MultiTurnEnv):
    def __init__(self, 
                 dataset: str = "gsm8k",
                 system_prompt: str = SIMPLE_PROMPT,
                 few_shot: List[Dict[str, str]] = DOUBLECHECK_FEW_SHOT[0],
                 **kwargs):
        
        sampling_args = {
            "skip_special_tokens": False,
            "spaces_between_special_tokens": False,
        }
        super().__init__(dataset=dataset, system_prompt=system_prompt, few_shot=few_shot, sampling_args=sampling_args, **kwargs)
        self.dataset_name = dataset
        self.dataset = preprocess_dataset(
            dataset_name=dataset,
            split="train",
            system_prompt=system_prompt,
            few_shot=few_shot
        )
        self.rubric = MathRubric()

    def get_rubric(self, **kwargs: Any) -> List[RewardFunc]:
        return self.rubric.get_reward_funcs()
    
    def get_dataset(self, **kwargs: Any) -> Dataset:
        return self.dataset
    
    def is_completed(self, messages: List[Dict[str, str]], **kwargs: Any) -> bool:
        return len(messages) > 1 and messages[-2]['content'] == 'Are you sure?'
    
    def env_response(self, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, str]:
        return {'role': 'user', 'content': 'Are you sure?'}