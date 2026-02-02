import inspect
import json
from typing import List, Dict, Any, Callable

from datasets import Dataset
from trl.trainer.grpo_trainer import RewardFunc

from verifiers.envs.multiturn_env import MultiTurnEnv
from verifiers.parsers import XMLParser
from verifiers.prompts import DEFAULT_TOOL_PROMPT_TEMPLATE
from verifiers.rubrics import ToolRubric
from verifiers.utils import preprocess_dataset

def infer_schema_from_function(func: Callable) -> Dict[str, Any]:
    """Infers a tool schema from a function's signature and docstring."""
    sig = inspect.signature(func)
    doc = inspect.getdoc(func) or ""
    
    # Parse docstring sections
    doc_parts = doc.split("\n\n")
    description = doc_parts[0].strip()
    
    # Extract examples if present
    examples = []
    for part in doc_parts:
        if part.startswith("Examples:"):
            examples = [line.strip() for line in part.split("\n")[1:] if line.strip()]
    
    # Build args schema
    args = {}
    for name, param in sig.parameters.items():
        param_doc = ""
        for part in doc_parts:
            if part.strip().startswith("Args:"):
                for line in part.split("\n")[1:]:
                    if line.strip().startswith(f"{name}:"):
                        param_doc = line.strip()[len(name)+1:].strip()
        
        args[name] = {
            "type": str(param.annotation.__name__ if param.annotation != inspect.Parameter.empty else "any"),
            "description": param_doc,
        }
        if param.default != inspect.Parameter.empty:
            args[name]["default"] = param.default
    
    return {
        "name": func.__name__,
        "description": description,
        "args": args,
        "returns": str(sig.return_annotation.__name__ if sig.return_annotation != inspect.Parameter.empty else "any"),
        "examples": examples
    }

def format_tool_descriptions(schemas: List[Dict[str, Any]]) -> str:
    """Formats tool schemas into a user-friendly description string."""
    descriptions = []
    for schema in schemas:
        desc = [f"{schema['name']}: {schema['description']}"]
        
        desc.append("\nArguments:")
        for arg_name, arg_info in schema['args'].items():
            default = f" (default: {arg_info['default']})" if 'default' in arg_info else ""
            desc.append(f"  - {arg_name}: {arg_info['description']}{default}")
        
        if schema['examples']:
            desc.append("\nExamples:")
            for example in schema['examples']:
                desc.append(f"  {example}")
        
        descriptions.append("\n".join(desc))
    
    return "\n\n".join(descriptions)

class ToolEnv(MultiTurnEnv):
    def __init__(self,
                 dataset: str = "gsm8k",
                 tools: List[Callable] = [],
                 system_prompt: str = DEFAULT_TOOL_PROMPT_TEMPLATE,
                 few_shot: List[Dict[str, str]] = [],
                 sampling_args={
                     "stop": ["</tool>", "</answer>"],
                     "include_stop_str_in_output": True
                 },
                 mask_env_response: bool = True,
                 max_steps: int = 10, **kwargs):
        # Infer schemas from tool functions
        self.tool_schemas = [infer_schema_from_function(tool) for tool in tools]
        self.tools = {tool.__name__: tool for tool in tools}
        
        # Format the system prompt with tool descriptions
        tool_descriptions = format_tool_descriptions(self.tool_schemas)
        formatted_prompt = system_prompt.format(tool_descriptions=tool_descriptions)
        
        super().__init__(
            system_prompt=formatted_prompt,
            few_shot=few_shot,
            mask_env_response=mask_env_response,
            sampling_args=sampling_args,
            **kwargs
        )
        self.dataset_name = dataset
        self.dataset = preprocess_dataset(
            dataset_name=dataset,
            split="train",
            system_prompt=formatted_prompt,
            few_shot=few_shot
        )
        self.eval_dataset = None
        self.max_steps = max_steps
        self.rubric = ToolRubric()
        self.llm_parser = XMLParser(fields=["reasoning", ("tool", "answer")])
        self.env_parser = XMLParser(fields=["result"])

    def get_dataset(self, **kwargs: Any) -> Dataset:
        return self.dataset
    
    def get_eval_dataset(self, n: int = -1, **kwargs: Any) -> Dataset | None:
        if self.eval_dataset is None:
            self.eval_dataset = preprocess_dataset(
                dataset_name=self.dataset_name,
                split="validation",
                system_prompt=self.system_prompt,
                few_shot=self.few_shot
            )
        if n > 0:
            return self.eval_dataset.shuffle().select(range(n)) # type: ignore
        return self.eval_dataset
    
    def get_rubric(self, **kwargs: Any) -> List[RewardFunc]:
        return self.rubric.get_reward_funcs()
    
    def _get_step_count(self, messages: List[Dict[str, str]]) -> int:
        """Count the number of tool uses in the message history, excluding few-shot examples."""
        step_count = 0
        
        # Skip messages that are part of few-shot examples
        # We need to determine where the actual conversation starts
        # System message + few-shot examples + user query = start of actual conversation
        conversation_start = 1  # Start after system message
        if self.few_shot:
            # Account for all few-shot messages
            conversation_start += len(self.few_shot)
        
        # Only count tool uses from the actual conversation
        for message in messages[conversation_start:]:
            if message.get("role") == "assistant":
                try:
                    parsed = self.llm_parser.parse(message["content"])
                    if hasattr(parsed, 'tool') and parsed.tool is not None:
                        step_count += 1
                except Exception:
                    pass
        return step_count
    
    def is_completed(self, messages: List[Dict[str, str]], **kwargs: Any) -> bool:
        try:
            # Check if we've hit max steps by counting tool uses in the message history
            step_count = self._get_step_count(messages)
            if step_count >= self.max_steps:
                return True
            
            parsed = self.llm_parser.parse(messages[-1]["content"])
            # Check if we got a valid answer field (not just None from failed parsing)
            return hasattr(parsed, 'answer') and parsed.answer is not None
        except Exception:
            return False

    def call_tool(self, tool_json: str, **kwargs: Any) -> str:
        """Call a tool based on JSON command."""
        try:
            command = json.loads(tool_json)
            if not isinstance(command, dict):
                return "Error: Tool command must be a JSON object"
            
            tool_name = command.get("name")
            if not tool_name:
                return "Error: Tool command must specify 'name'"
            
            if tool_name not in self.tools:
                return f"Error: Unknown tool '{tool_name}'"
            
            tool_func = self.tools[tool_name]
            tool_args = command.get("args", {})
            
            # Call the tool function with arguments
            result = tool_func(**tool_args)
            return str(result)
        except json.JSONDecodeError:
            return "Error: Invalid JSON format"
        except Exception as e:
            return f"Error: {str(e)}"

    def env_response(self, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, str]:
        try:
            parsed = self.llm_parser.parse(messages[-1]["content"])
            # Check if we got a valid tool field (not just None from failed parsing)
            if hasattr(parsed, 'tool') and parsed.tool is not None:
                result = self.call_tool(parsed.tool)
                if len(result.strip()) > 0:
                    return {"role": "user", "content": self.env_parser.format(result=result)}
                else:
                    return {"role": "user", "content": "Error: Tool execution returned empty output."}
        except Exception:
            pass
        return {"role": "user", "content": "Error: Tool command not found or invalid XML format. Please ensure correct formatting."}