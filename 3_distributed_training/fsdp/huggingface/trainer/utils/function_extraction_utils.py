"""
Function Extraction Utilities

This module provides utilities for extracting and parsing function calls and responses
from AI assistant interactions. It handles different formats including GlaiveAI,
and OpenAI.

The module includes functions for:
- Extracting content between specific tags
- Parsing function calls from different formats
- Extracting function definitions from system messages
- Processing tool calls and results

It also defines custom exceptions for various error conditions during extraction and parsing.
"""

import json
import re
from typing import List, Dict, Any, Tuple, Union, Optional
from enum import Enum

from utils.data_format import (
    IndicatorTags,
    ToolCallType,
    DatasetFormat,
    check_tool_calls_format,
)


class FunctionCallFormatError(Exception):
    """Raised when a function call is expected but not found/ in a wrong format in the assistant response."""

    pass


class FunctionResponseFormatError(Exception):
    """Raised when a function response is not found/ in a wrong format in the given content."""

    pass


class PatternNotFoundError(Exception):
    """Raised when no content is not found based on the given string and tags."""

    pass


class FunctionFormatError(Exception):
    """Raised when function definition is in the wrong format in the given string."""

    pass


def extract_segment_between_tags(
    string: str, indicator_tags: IndicatorTags
) -> Tuple[Optional[str], str]:
    """
    Extract content between specified tags in a string.

    Args:
        string: The input string to search in
        indicator_tags: IndicatorTags object containing start and end tags

    Returns:
        Tuple containing:
            - Optional prefix before the start tag (None if string starts with start tag)
            - Content between the start and end tags

    Raises:
        PatternNotFoundError: If the tags are not found in the string
    """
    string = string.strip()
    escaped_tags = [re.escape(tag) for tag in indicator_tags]

    if string.startswith(indicator_tags.start):
        pattern = r"{}([\s\S]*?){}".format(*escaped_tags)
        extract_prefix = False
    else:
        pattern = r"([\s\S]*?){}([\s\S]*?){}".format(*escaped_tags)
        extract_prefix = True

    pattern_match = re.search(pattern, string)
    if not pattern_match:
        raise PatternNotFoundError(
            f"No content found in the string {string} with the given tags {indicator_tags}"
        )
    prefix, special_content = None, None
    if extract_prefix:
        prefix = pattern_match.group(1).strip()
        special_content = pattern_match.group(2).strip()
    else:
        prefix = None
        special_content = pattern_match.group(1).strip()
    return prefix, special_content


def _extract_functions_from_system_msg_glaive(system_str: str) -> List[Dict[str, Any]]:
    """
    Extract function definitions from a GlaiveAI system message.

    Args:
        system_str: The system message string containing function definitions

    Returns:
        List of function definitions in OpenAI tools format

    Raises:
        FunctionFormatError: If the function definitions are not in valid JSON format
    """
    # Extracting the functions using regex
    functions_match = re.findall(r"\{.*?\}(?=\s*\{|\s*$)", system_str, re.DOTALL)
    functions = []

    for fn in functions_match:
        try:
            # Convert string representation of dictionary to actual dictionary
            fn_dict = json.loads(fn)
            functions.append(fn_dict)
        except json.JSONDecodeError:
            # In case the string is not a valid JSON, raise an error
            raise FunctionFormatError(
                f"Tool list not in the correct format in : {system_str}"
            )

    # Some functions may not have parameters. Fix them
    for fn in functions:
        if not fn["parameters"]:
            fn["parameters"] = {
                "type": "object",
                "properties": {},
                "required": [],
            }
    # Bring it into the OpenAI tools format
    functions = [{"type": "function", "function": fn} for fn in functions]
    return functions


def extract_functions_from_system_msg(
    system_str: str,
    format: DatasetFormat,
) -> List[Dict[str, Any]]:
    """
    Extract function definitions from a system message based on the dataset format.

    Args:
        system_str: The system message string containing function definitions
        format: The dataset format (GLAIVE)

    Returns:
        List of function definitions in the appropriate format

    Raises:
        NotImplementedError: If the specified format is not supported
    """
    if format == DatasetFormat.GLAIVE:
        return _extract_functions_from_system_msg_glaive(system_str)
    else:
        raise NotImplementedError(
            f"Function extraction for format {format} not implemented"
        )


def _parse_function_calls_glaive(string: str) -> List[Dict[str, Any]]:
    """
    Parse function calls from a GlaiveAI format string.

    Args:
        string: JSON string containing function calls

    Returns:
        List of parsed function call dictionaries

    Raises:
        json.JSONDecodeError: If the string is not valid JSON
    """
    # Remove single quotes used for the arguments field.
    string = string.replace("'", "")
    # Parse the string into a list of JSONs
    json_list = json.loads(string)
    if isinstance(json_list, dict):
        json_list = [json_list]
    return json_list


def _parse_function_calls_openai(string: str) -> List[Dict[str, Any]]:
    """
    Parse function calls from an OpenAI format string.

    Args:
        string: JSON string containing function calls

    Returns:
        List of parsed function call dictionaries with arguments parsed as JSON

    Raises:
        json.JSONDecodeError: If the string is not valid JSON
        FunctionCallFormatError: If the function call format is invalid
    """
    # Parse the string into a list of JSONs
    json_list = json.loads(string)
    if isinstance(json_list, dict):
        json_list = [json_list]
    for json_obj in json_list:
        if "function" not in json_obj or "arguments" not in json_obj["function"]:
            raise FunctionCallFormatError(
                f"Function call not in the correct format in : {string}"
            )
        json_obj["function"]["arguments"] = json.loads(
            json_obj["function"]["arguments"]
        )
    return json_list


def parse_function_calls(string: str, format: DatasetFormat) -> List[Dict[str, Any]]:
    """
    Parse function calls from a string based on the dataset format.

    Args:
        string: JSON string containing function calls
        format: The dataset format (GLAIVE or other formats use OpenAI parsing)

    Returns:
        List of parsed function call dictionaries
    """
    if format == DatasetFormat.GLAIVE:
        return _parse_function_calls_glaive(string)
    else:
        return _parse_function_calls_openai(string)


def get_tool_calls_from_response(
    raw_response: str, tool_call_tags: IndicatorTags, format: DatasetFormat
) -> Tuple[str, List[ToolCallType]]:
    """
    Extract tool calls from an assistant response.

    Args:
        raw_response: The assistant's response text
        tool_call_tags: IndicatorTags object containing start and end tags for tool calls
        format: The dataset format to use for parsing

    Returns:
        Tuple containing:
            - Response text before the tool calls
            - List of parsed tool call objects

    Raises:
        FunctionCallFormatError: If tool calls cannot be found or are in an invalid format
    """
    try:
        response_text, tool_calls_str = extract_segment_between_tags(
            raw_response, tool_call_tags
        )
        tool_calls = parse_function_calls(tool_calls_str, format)
    except (PatternNotFoundError, json.JSONDecodeError) as e:
        # Propagate a custom exception for use later
        raise FunctionCallFormatError(f"Tool calls could not be found : {e}")

    if not check_tool_calls_format(tool_calls, format):
        raise FunctionCallFormatError("Tool call is not in the correct format")
    return response_text, tool_calls


def parse_tool_result(string: str, tool_result_tags: Tuple[str, str]) -> Dict[str, Any]:
    """
    Parse tool result from a string.

    Args:
        string: The string containing tool result
        tool_result_tags: Tuple containing start and end tags for tool result

    Returns:
        Dictionary containing the parsed tool result

    Raises:
        FunctionResponseFormatError: If tool result cannot be found or is not valid JSON
    """
    try:
        _, tool_result_str = extract_segment_between_tags(string, tool_result_tags)
        result = json.loads(tool_result_str)
    except (PatternNotFoundError, json.JSONDecodeError) as e:
        # Propagate a custom exception for use later
        raise FunctionResponseFormatError(f"Tool result could not be found : {e}")
    return result
