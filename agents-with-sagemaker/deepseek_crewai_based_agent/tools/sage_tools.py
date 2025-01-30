import boto3
import json
from crewai.tools import BaseTool
from typing import Callable, Dict, List, Union
from pydantic import Field
from crewai import LLM


def extract_user_content(prompt: Union[List[Dict[str, str]], str]) -> str:
    """
    Extracts the user's content from a CrewAI prompt structure, filtering out system instructions.

    Args:
        prompt: Either a list of message dictionaries (CrewAI format) or a direct string prompt.
               Expected format for list: [{'role': 'user', 'content': 'message'}, ...]

    Returns:
        str: The extracted user content or empty string if no user content found.

    Example:
        >>> messages = [{'role': 'system', 'content': 'system message'},
        ...            {'role': 'user', 'content': 'user query'}]
        >>> content = extract_user_content(messages)
        >>> print(content)
        'user query'
    """
    if isinstance(prompt, list) and all(isinstance(item, dict) for item in prompt):
        for message in prompt:
            if message.get("role") == "user":
                return message.get("content", "").strip()
    return ""


def deepseek_llama_inference(prompt: dict, endpoint_name: str, region: str = "us-east-2") -> dict:
    """
    Performs inference using a DeepSeek LLaMA model deployed on SageMaker.

    This function handles the complete inference pipeline:
    1. Prepares the input prompt for the model
    2. Calls the SageMaker endpoint
    3. Processes the response
    4. Separates reasoning from final answer using </think> delimiter

    Args:
        prompt (dict): Dictionary containing the prompt text under 'prompt' key
        endpoint_name (str): Name of the deployed SageMaker endpoint
        region (str): AWS region where the endpoint is deployed (default: "us-east-2")

    Returns:
        dict: Contains two keys:
            - "Chain of Thought": Model's reasoning process
            - "Final Answer": Model's final response

    Raises:
        RuntimeError: If there's an error calling the SageMaker endpoint
        ValueError: If the response format is unexpected
    """
    import boto3
    import json

    client = boto3.client('sagemaker-runtime', region_name=region)

    user_content = prompt.get("prompt", "").strip()
    payload = {
        "inputs": user_content,
        "parameters": {"max_new_tokens": 2048, "return_full_text": True}
    }

    try:
        response = client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="application/json",
            Body=json.dumps(payload)
        )

        response_body = response['Body'].read().decode()
        result = json.loads(response_body)

        if isinstance(result, list) and "generated_text" in result[0]:
            generated_text = result[0]["generated_text"]
        else:
            raise ValueError("Unexpected response format from SageMaker endpoint.")

        #print(f"DEBUG: Generated text: {generated_text}")

        if "</think>" in generated_text:
            chain_of_thought, final_answer = generated_text.split("</think>", 1)
        else:
            chain_of_thought = generated_text
            final_answer = "Final Answer not explicitly provided."

        return {
            "Chain of Thought": chain_of_thought.strip(),
            "Final Answer": final_answer.strip()
        }

    except Exception as e:
        raise RuntimeError(f"Error while calling SageMaker endpoint: {e}")

    
class DeepSeekSageMakerLLM(LLM):
    """
    CrewAI-compatible LLM implementation for DeepSeek models on SageMaker.

    This class bridges CrewAI's agent system with SageMaker-deployed DeepSeek models,
    handling all necessary format conversions and communication.

    Attributes:
        endpoint_input_name (str): SageMaker endpoint name for model inference
        model (str): Identifier for the model type (set to "sagemaker-deepseek-model")

    Example:
        >>> llm = DeepSeekSageMakerLLM(endpoint="deepseek-endpoint")
        >>> response = llm.call("What is machine learning?")
    """

    def __init__(self, endpoint: str):
        """
        Initialize the DeepSeek SageMaker LLM interface.

        Args:
            endpoint (str): Name of the SageMaker endpoint to use
        """
        super().__init__(model="sagemaker-deepseek-model")
        self.endpoint_input_name = endpoint
        #print(f"DEBUG: LLM initialized with endpoint: {self.endpoint_input_name}")

    def call(self, prompt: Union[List[Dict[str, str]], str], **kwargs) -> str:
        """
        Process a prompt through the DeepSeek model on SageMaker.

        Args:
            prompt: Either a string prompt or list of message dictionaries
            **kwargs: Additional arguments for the inference process

        Returns:
            str: Formatted response with thought process and final answer

        Raises:
            ValueError: If no valid user content can be extracted from prompt
        """
        #print(f"DEBUG: Original LLM prompt: {prompt}")

        user_content = extract_user_content(prompt)
        if not user_content:
            raise ValueError("No valid user content found in the prompt.")

        #print(f"DEBUG: Extracted user content: {user_content}")

        response = deepseek_llama_inference(
            prompt={"prompt": user_content},
            endpoint_name=self.endpoint_input_name,
            region="us-east-2"
        )
        #print(f"DEBUG: LLM response: {response}")
        return f"Thought: Let me explain.\nFinal Answer: {response}"


class CustomTool(BaseTool):
    """
    Generic tool wrapper for CrewAI custom function integration.

    This class allows any Python function to be used as a CrewAI tool while
    maintaining the expected tool interface and configuration options.

    Attributes:
        name (str): Tool identifier for CrewAI
        description (str): Human-readable description of the tool's purpose
        func (Callable): The function to be executed when tool is used

    Example:
        >>> def custom_function(x): return x * 2
        >>> tool = CustomTool(
        ...     name="doubler",
        ...     description="Doubles the input value",
        ...     func=custom_function
        ... )
    """
    
    name: str = Field(description="The name of the tool")
    description: str = Field(description="The description of the tool")
    func: Callable = Field(description="The function to execute")

    def _run(self, *args, **kwargs):
        """Execute the wrapped function with provided arguments."""
        return self.func(*args, **kwargs)

    class Config:
        arbitrary_types_allowed = True