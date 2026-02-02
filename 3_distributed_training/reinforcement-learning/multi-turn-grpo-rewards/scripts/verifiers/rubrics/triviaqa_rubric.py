from typing import List, Dict

from verifiers.parsers import XMLParser
from verifiers.rubrics import Rubric
from verifiers.rubrics import ToolRubric

class TrivialQAToolRubric(ToolRubric):
    def __init__(self,
                 parser: XMLParser = XMLParser(fields=["reasoning", ("tool", "answer")]),
                 env_parser: XMLParser = XMLParser(fields=["result"])):
        self.parser = parser
        self.env_parser = env_parser
        self.reward_funcs = [
            self.tool_execution_reward_func,
            self.exist_answer_in_search_results,
            self.exist_answer_reward_func,
            self.exact_match_reward_func,
            self.parser.get_format_reward_func(),
            self.parser.get_xml_reward_func(),
        ]
        
    
    def exist_answer_reward_func(self, completions, answer, **kwargs) -> List[float]:
        """Reward function that checks if the expected answer exists in the completions."""
        
        '''
        
        def exact_match_reward(responses, answers=None):
            """Reward if generated response contains correct answer."""
            rewards = []
            for response, answer in zip(responses, answers):
                reward = 0.0
                for a in answer:
                    if a.lower() in response.lower():
                        reward += 1.0
                        break
                rewards.append(torch.tensor(reward))
            return rewards
                
        '''
        
        responses = [self.get_last_answer(c) for c in completions]
        rewards = []
        
        for r, a_list in zip(responses, answer):
            if r is None:
                rewards.append(0.0)
                continue
                
            r_lower = str(r).lower()
            reward = 0.0
            
            # Check if any of the accepted answers is contained in the response
            for a in a_list:
                if str(a).lower() in r_lower:
                    reward = 0.5
                    break
            
            rewards.append(reward)
            
        return rewards
    
    def exact_match_reward_func(self, completions, answer, **kwargs) -> List[float]:
        """Reward function that checks if the generated answer exactly matches any of the accepted answers."""
    
        responses = [self.get_last_answer(c) for c in completions]
        rewards = []
        
        for r, a_list in zip(responses, answer):
            if r is None:
                rewards.append(0.0)
                continue
                
            r_lower = str(r).lower().strip()
            reward = 0.0
            
            # Check if the response exactly matches any of the accepted answers
            for a in a_list:
                if str(a).lower().strip() == r_lower:
                    reward = 1.0
                    break
            
            rewards.append(reward)
            
        return rewards

    def get_search_results(self, trajectory: List[Dict[str, str]]) -> str | None:
        """
        Extract search results from environment responses in a trajectory.
        
        Args:
            trajectory: List of message dictionaries in the conversation history
            
        Returns:
            The content of the <result> tag or None if not found
        """
        # Find assistant message with tool call followed by user response
        for i, msg in enumerate(trajectory):
            if msg['role'] == 'assistant':
                # Check if this message contains a tool call
                parsed = self.parser.parse(msg['content'])
                if hasattr(parsed, 'tool') and parsed.tool is not None:
                    # Look for the next message which should be the environment response
                    if i + 1 < len(trajectory) and trajectory[i + 1]['role'] == 'user':
                        user_msg = trajectory[i + 1]['content']
                        
                        # Parse the user message to extract the result
                        env_parsed = self.env_parser.parse(user_msg)
                        if hasattr(env_parsed, 'result') and env_parsed.result is not None:
                            return env_parsed.result
        
        return None

    def exist_answer_in_search_results(self, completions, answer, **kwargs) -> List[float]:
        """
        Reward function that checks if the correct answer exists in the search results.
        
        Args:
            completions: List of conversation trajectories
            answer: List of lists of acceptable answers
            
        Returns:
            List of reward values (1.0 if answer exists in search results, 0.0 otherwise)
        """
        rewards = []
        
        for c, a_list in zip(completions, answer):
            # Get search results for this trajectory
            search_result = self.get_search_results(c)
            
            # If no search results, give zero reward
            if search_result is None:
                rewards.append(0.0)
                continue
            
            # Convert to lowercase for case-insensitive matching
            search_result_lower = search_result.lower()
            reward = 0.0
            
            # Check if any of the accepted answers is in the search results
            for a in a_list:
                if str(a).lower() in search_result_lower:
                    reward = 0.5
                    break
            
            rewards.append(reward)
        
        return rewards