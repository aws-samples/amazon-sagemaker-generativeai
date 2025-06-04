"""
Preprocessing Utilities for AI Assistant Conversation Data

This module provides utilities for preprocessing conversation data between users and AI assistants.
It focuses on converting data from GlaiveAI format to OpenAI format, handling tool calls,
and ensuring proper message structure.

The module includes functions for:
- Converting GlaiveAI formatted data to OpenAI format
- Parsing chat strings into structured message lists
- Handling tool calls and responses
- Filtering invalid or malformed conversation entries
"""

from typing import Dict, Any, List
from datasets import Dataset
import re
import json
import logging
import uuid

from utils.function_extraction_utils import (
    get_tool_calls_from_response,
    extract_functions_from_system_msg,
    FunctionCallFormatError,
    DatasetFormat,
    FunctionFormatError,
)
from utils.data_format import (
    GLAIVEAI_SYSTEM_NO_TOOLS,
    GLAIVEAI_SYSTEM_WITH_TOOLS,
    GLAIVEAI_TOOL_CALL_INDICATORS,
    GLAIVEAI_TOOL_CALL_PREFIX,
    GLAIVEAI_EOS,
    DEFAULT_SYSTEM_PROMPT,
    GlaiveAIRoleTags,
    MessageType,
)


class InvalidSystemPromptError(Exception):
    """Raised when an invalid system prompt is found."""

    pass


class InvalidRoleError(Exception):
    """Raised when an invalid role is found in a message."""

    pass


class TagsNotFoundError(Exception):
    """Raised when none of the expected tags are found in the chat string."""

    pass


def _glaive_to_openai(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a single example from GlaiveAI format to OpenAI format.

    Args:
        example: Dictionary containing 'system' and 'chat' keys in GlaiveAI format

    Returns:
        Dictionary with 'messages' and 'tools' keys in OpenAI format

    Raises:
        InvalidSystemPromptError: If the system prompt doesn't match expected formats
    """
    messages = []
    tools = None
    if GLAIVEAI_SYSTEM_WITH_TOOLS in example["system"]:
        try:
            tools = extract_functions_from_system_msg(
                example["system"], format=DatasetFormat.GLAIVE
            )
        except FunctionFormatError as e:
            logging.info(f"Error processing example {example['system']} : {e}")
            return {"messages": None, "tools": None}
        # Convert to string for compatiblity with PyArrow
        tools = json.dumps(tools)
    elif GLAIVEAI_SYSTEM_NO_TOOLS not in example["system"]:
        # If an unexpected system prompt is found, raise an error to investigate
        raise InvalidSystemPromptError(
            f"System prompt {example['system']} does not match expected prefixes"
        )

    messages.append({"role": "system", "content": DEFAULT_SYSTEM_PROMPT})
    try:
        chat_messages = chat_str_to_messages(example["chat"])
    except (FunctionCallFormatError, TagsNotFoundError, json.JSONDecodeError) as e:
        # For chat data format errors, propagate None types for filtering later
        return {"messages": None, "tools": None}

    messages.extend(chat_messages)
    processed_example = {"messages": messages, "tools": tools}
    return processed_example


def combine_multiple_entries(assistant_content: str) -> str:
    """
    Combine multiple assistant entries that may have been split by tool call prefixes.

    Args:
        assistant_content: The assistant's response content

    Returns:
        Combined assistant content with proper tool call formatting
    """
    if (
        assistant_content.startswith(GLAIVEAI_TOOL_CALL_PREFIX)
        or GLAIVEAI_TOOL_CALL_PREFIX not in assistant_content
    ):
        return assistant_content
    else:
        assistant_tag = GlaiveAIRoleTags.ASSISTANT.value
        fn_call_pattern = r"([\s\S]*?){}\s+{}([\s\S]*)".format(
            re.escape(assistant_tag), re.escape(GLAIVEAI_TOOL_CALL_PREFIX)
        )
        function_call_match = re.search(fn_call_pattern, assistant_content, re.DOTALL)
        if function_call_match:
            content1 = function_call_match.group(1).strip()
            content2 = function_call_match.group(2).strip()
            assistant_content = content1 + GLAIVEAI_TOOL_CALL_PREFIX + content2
    return assistant_content


def chat_str_to_messages(chat: str) -> List[MessageType]:
    """
    Convert a chat string with GlaiveAI format tags to a list of structured messages.

    Args:
        chat: String containing the chat conversation with GlaiveAI role tags

    Returns:
        List of message dictionaries in OpenAI format

    Raises:
        TagsNotFoundError: If no user/assistant/tool messages are found
        FunctionCallFormatError: If function calls are not properly formatted
    """
    user_tag = GlaiveAIRoleTags.USER.value
    assistant_tag = GlaiveAIRoleTags.ASSISTANT.value
    tool_tag = GlaiveAIRoleTags.TOOL.value
    # Regex pattern to extract user, assistant and tool messages.
    tag_pattern = re.compile(
        rf"(?:{user_tag}\s*(?P<user>.*?)\s*(?={assistant_tag}|$)|{assistant_tag}\s*(?P<assistant>.*?)\s*(?={tool_tag}|{user_tag}|$)|{tool_tag}\s*(?P<function_response>.*?)\s*(?={tool_tag}|{assistant_tag}|$))",
        re.DOTALL,
    )

    matches = tag_pattern.finditer(chat)
    # If no matches found, raise an error
    if not matches:
        raise TagsNotFoundError(f"No user/assistant/tool message found in {chat}")
    messages = []
    # Keep track of the tool call ids and function names in the previous assistant response
    previous_tool_calls_info = []
    # Loop through all matches and extract the respective roles and content
    for match in matches:
        if match.group("user"):
            user_content = match.group("user").strip()
            msg = {"role": "user", "content": user_content}
        elif match.group("assistant"):
            assistant_content = match.group("assistant").strip()
            assistant_content = combine_multiple_entries(assistant_content)

            # Glaive dataset is full of single function calls.
            # We extract the function call and place it in the tool_calls field
            openai_fmt_tool_calls = []
            if GLAIVEAI_TOOL_CALL_PREFIX in assistant_content:
                # Get the function calls from the response.
                # We convert to JSON and then back to string to ensure the format is correct
                assistant_content, tool_calls = get_tool_calls_from_response(
                    assistant_content,
                    GLAIVEAI_TOOL_CALL_INDICATORS,
                    format=DatasetFormat.GLAIVE,
                )
                if assistant_content is None:
                    assistant_content = ""
                for i, tool_call in enumerate(tool_calls):
                    # Generate a short UUID for the tool call id
                    tool_id = str(uuid.uuid4())[:9]
                    openai_fmt_tool_call = {
                        "type": "function",
                        "function": {
                            "name": tool_call["name"],
                            # arguments field is stringified with single quotes
                            "arguments": json.dumps(tool_call["arguments"]),
                            "id": tool_id,
                        },
                    }
                    openai_fmt_tool_calls.append(openai_fmt_tool_call)
            # Remove the eos token if present
            assistant_content = assistant_content.replace(GLAIVEAI_EOS, "")
            msg = {
                "role": "assistant",
                "content": assistant_content,
                "tool_calls": openai_fmt_tool_calls,
            }
            previous_tool_calls_info = []
            for i, tool_call in enumerate(openai_fmt_tool_calls):
                tool_call_id = tool_call["function"].get("id")
                if tool_call_id is None or tool_call_id == "":
                    tool_call_id = str(uuid.uuid4())[:9]
                previous_tool_calls_info.append(
                    (tool_call_id, tool_call["function"]["name"])
                )
        elif match.group("function_response"):
            function_response = match.group("function_response").strip()
            role = "tool"
            # Skip this element if no tool call id is found
            if not len(previous_tool_calls_info):
                continue
            tool_call_id, tool_call_name = previous_tool_calls_info.pop(0)
            msg = {
                "role": role,
                "content": function_response,
                "name": tool_call_name,
                "tool_call_id": tool_call_id,
            }
        else:
            # Sometimes, the input can be malformed with no content in the captured group.
            # Example: 'USER: \n'. Skip these entries
            continue
        messages.append(msg)
    return messages


def filter_func(example: Dict[str, Any]) -> bool:
    """
    Filter function to remove invalid conversation examples.

    Args:
        example: Dictionary containing 'messages' key with conversation messages

    Returns:
        bool: True if the example is valid, False otherwise
    """
    messages = example["messages"]
    is_good_entry = True
    j = 0
    while j + 1 < len(messages):
        # Sometimes,a single message has the same assistant response repeated. We remove these entries along with the ones where we have consecutive assistant responses
        if (
            messages[j]["role"] == messages[j + 1]["role"]
            or GlaiveAIRoleTags.ASSISTANT.value in messages[j]["content"]
        ):
            is_good_entry = False
            break

        j += 1
    return is_good_entry


def glaive_to_openai(ds: Dataset) -> Dataset:
    """
    Convert a dataset from GlaiveAI format to OpenAI format.

    Args:
        ds: Dataset in GlaiveAI format

    Returns:
        Dataset in OpenAI format with invalid entries filtered out
    """
    ds = ds.map(_glaive_to_openai)
    ds = ds.filter(lambda x: x["messages"] is not None)
    ds = ds.filter(filter_func)
    return ds
