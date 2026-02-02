from typing import List, Dict, Any, Tuple
from datasets import Dataset
from trl.trainer.grpo_trainer import RewardFunc

from verifiers.envs.simple_env import SimpleEnv
from verifiers.parsers import XMLParser
from verifiers.rubrics import MathRubric
from verifiers.prompts import SIMPLE_PROMPT, MATH_FEW_SHOT
from verifiers.utils import preprocess_dataset

class MathEnv(SimpleEnv):
    def __init__(self,
                 dataset: str = "gsm8k",
                 system_prompt: str = SIMPLE_PROMPT,    
                 few_shot: List[Dict[str, str]] = MATH_FEW_SHOT[0],
                 fields: List[str | Tuple[str, ...]] = ["reasoning", "answer"],
                 **kwargs):
        super().__init__(system_prompt=system_prompt, few_shot=few_shot, **kwargs)
        self.parser = XMLParser(fields=fields)
        self.dataset_name = dataset
        self.dataset = preprocess_dataset(
            dataset_name=dataset,
            split="train",
            system_prompt=system_prompt,
            few_shot=few_shot
        ) 
        self.eval_dataset = None
        self.rubric = MathRubric()
    
    def get_dataset(self, **kwargs: Any) -> Dataset:
        return self.dataset
    
    def get_eval_dataset(self, n: int = -1, **kwargs: Any) -> Dataset | None:
        if self.eval_dataset is None:
            self.eval_dataset = preprocess_dataset(
                dataset_name=self.dataset_name,
                split="test",
                system_prompt=self.system_prompt,
                few_shot=self.few_shot
            )
        if n > 0:
            return self.eval_dataset.shuffle().select(range(n)) # type: ignore
        return self.eval_dataset 
    
    def get_rubric(self, **kwargs: Any) -> List[RewardFunc]:
        return self.rubric.get_reward_funcs()
