"""
Inference utilities for deployed SAMA RL models
"""
import boto3
import json
from transformers import AutoTokenizer
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class EndpointInference:
    """Run inference on deployed SageMaker endpoints"""
    
    def __init__(self, endpoint_name: str, base_model_name: str = "Qwen/Qwen2-0.5B-Instruct"):
        """
        Args:
            endpoint_name: SageMaker endpoint name
            base_model_name: Base model name for tokenizer
        """
        self.endpoint_name = endpoint_name
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        self.sagemaker_runtime = boto3.client('sagemaker-runtime')
    
    def generate(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.0, stop_on_repetition: bool = True) -> str:
        """
        Generate completion for a prompt using the HuggingFace LLM endpoint
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stop_on_repetition: Stop generation if repetition detected
            
        Returns:
            Generated completion
        """
        # HuggingFace LLM container expects "inputs" key
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_new_tokens,
                "do_sample": temperature > 0,
                "repetition_penalty": 1.1,  # Penalize repetition
                "no_repeat_ngram_size": 3   # Prevent 3-gram repetition
            }
        }
        
        # Only add temperature if > 0 (HF LLM container requires strictly positive)
        if temperature > 0:
            payload["parameters"]["temperature"] = temperature
        
        try:
            response = self.sagemaker_runtime.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType='application/json',
                Body=json.dumps(payload)
            )
            
            result = json.loads(response['Body'].read().decode())
            
            # HuggingFace LLM container returns different format
            if isinstance(result, list) and len(result) > 0:
                completion = result[0].get("generated_text", "").replace(prompt, "").strip()
            elif isinstance(result, dict):
                completion = result.get("generated_text", "").replace(prompt, "").strip()
            else:
                completion = str(result).strip()
            
            # Post-process to remove repetition
            if stop_on_repetition:
                completion = self._remove_repetition(completion)
            
            return completion
                
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return ""
    
    def _remove_repetition(self, text: str) -> str:
        """Remove repetitive content from generated text"""
        sentences = text.split('. ')
        unique_sentences = []
        seen = set()
        
        for sentence in sentences:
            # Check if sentence is too similar to previous ones
            sentence_clean = sentence.lower().strip()
            if sentence_clean not in seen and len(sentence_clean) > 10:
                unique_sentences.append(sentence)
                seen.add(sentence_clean)
            elif len(unique_sentences) > 0:
                # Stop at first repetition
                break
        
        return '. '.join(unique_sentences)
    
    def run_inference(self, prompt: str, max_new_tokens: int = 2048) -> str:
        """
        Compatible with your reference code pattern
        """
        return self.generate(prompt, max_new_tokens, temperature=0.0)
    
    def batch_inference(self, prompts: List[str], max_new_tokens: int = 512) -> List[str]:
        """
        Run inference on multiple prompts
        
        Args:
            prompts: List of input prompts
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            List of completions
        """
        completions = []
        for prompt in prompts:
            completion = self.generate(prompt, max_new_tokens)
            completions.append(completion)
        return completions
    
    def get_token_count(self, text: str) -> int:
        """Get token count for text"""
        tokens = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        return len(tokens)


def create_inference_model(endpoint_name: str, base_model_name: str = "Qwen/Qwen2-0.5B-Instruct") -> EndpointInference:
    """
    Create inference model from endpoint name
    
    Args:
        endpoint_name: SageMaker endpoint name
        base_model_name: Base model name for tokenizer
        
    Returns:
        EndpointInference instance
    """
    return EndpointInference(endpoint_name, base_model_name)
