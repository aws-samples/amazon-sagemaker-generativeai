SIMPLE_PROMPT = """
Respond in the following format, using careful step-by-step reasoning.

<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

CODE_PROMPT = """\
Given a math problem, use step-by-step reasoning and code execution to solve the problem. 

For each step:
1. Think through your reasoning inside <reasoning> tags
2. Write Python scripts inside <code> tags to work out calculations
   - Functions and variables do not persist across <code> calls and should be redefined each time
   - Scripts should be written in Python 3.10+ syntax, and should run in under 10 seconds
   - Any desired outputs should be printed using print() statements
   - You may import numpy, scipy, and sympy libraries for your calculations
3. You will see the output from print() statements in your code in <output> tags
4. Continue until you can give the final answer inside <answer> tags
"""

DEFAULT_TOOL_PROMPT_TEMPLATE = """\
You have access to the following tools to help solve problems:

{tool_descriptions}

For each step:
1. Think through your reasoning inside <reasoning> tags
2. If needed, use a tool by writing a JSON command inside <tool> tags with:
   - "name": the tool to use
   - "args": the arguments for the tool
3. You will see the tool's output inside <result> tags
4. Continue until you can give the final answer inside <answer> tags

Tools expect specific JSON input formats. Follow the examples carefully.
Do not make up tools or arguments that aren't listed.
"""

DEFAULT_TRIVIALQA_TOOL_PROMPT_TEMPLATE = """\
You have access to the following tools to help solve problems:

{tool_descriptions}

Follow these steps exactly once:
1. Think through your reasoning inside <reasoning> tags
2. Use a tool by writing a JSON command inside <tool> tags with:
   - "name": the tool to use
   - "args": the arguments for the tool
3. You will see the tool's output inside <result> tags
4. Think through the tool's output inside <reasoning> tags
5. Based on your reasoning, provide your final answer inside <answer> tags

Important:
- Use the tool exactly once - DO NOT attempt to call the tool again even if the first search doesn't give helpful results
- You must work with both your prior knowledge and whatever information the single tool call provides
- If the tool doesn't return useful information, rely on your prior knowledge to answer
- Tools expect specific JSON input formats. Follow the examples carefully
- Do not make up tools or arguments that aren't listed
- After getting the tool result, analyze it in a reasoning step before giving your answer
- Your answer should match the expected ground-truth
"""