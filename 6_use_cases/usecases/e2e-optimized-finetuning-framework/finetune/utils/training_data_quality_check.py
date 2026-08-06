import re
from typing import List, Optional, Tuple

from finetune.utils.logging_util import get_logger

logger = get_logger(__name__)


class NumberAlignmentProcessor:
    def __init__(self):
        pass

    @staticmethod
    def extract_numbers(text: str) -> List[str]:
        """
        Extracts numbers from a string, removing currency and percentage signs.

        Args:
            text (str): The input text to extract numbers from.

        Returns:
            List[str]: A list of extracted numbers as strings.
        """
        pattern = r"[$€£¥]?\d+(?:,\d+)*(?:\.\d+)?(?:[KMBT]|\.\d+[KMBT])?(?:%|\b)"
        try:
            matches = re.findall(pattern, text)
            # Remove dollar sign ($) and pecentage sign (%) from numeric values
            extracted_numbers = [match.lstrip("$").rstrip("%") for match in matches]
            logger.info(f"Extracted numbers: {extracted_numbers}")
            return extracted_numbers
        except Exception as e:
            logger.exception("Error extracting numbers: %s", e)
            return []

    @staticmethod
    def extract_numbers_with_sign(text: str) -> List[str]:
        """
        Extracts numbers with their signs (including $, %, and KMBT) from a string.

        Args:
            text (str): The input text to extract numbers from.

        Returns:
            List[str]: A list of extracted numbers with signs.
        """
        pattern1 = r"[-+]?\$?\d{1,2},\d{1,2}"  # 1. For numbers like 1,75
        pattern2 = r"[-+]?\$?\d+(?:\.\s?\d+)?(?:[KMBT])?(?:%?)?"  # 2. For standard numbers with comma separators and optional KMBT and % signs
        pattern3 = r"[-+]?\$?\d+\.\d+\.\d+[KMBT]"  # 3. For numbers like 908.3.2K
        combined_pattern = f"{pattern1}|{pattern3}|{pattern2}"

        try:
            matches = re.findall(combined_pattern, text)
            # Filter out any empty matches and standalone letters
            matches = [match for match in matches if match and not match.isalpha()]
            # Remove dollar sign ($), plus sign (+) and percentage sign (%) from numeric values
            extracted_numbers = [
                match.replace("$", "").replace("+", "").replace("%", "") for match in matches
            ]
            logger.info(f"Extracted numbers with signs: {extracted_numbers}")
            return extracted_numbers
        except Exception as e:
            logger.exception("Error extracting numbers with signs: %s", e)
            return []

    @staticmethod
    def compare_lists(
        list1: List[str], list2: List[str]
    ) -> Tuple[bool, Optional[Tuple[List[str], List[str]]]]:
        """
        Compares two lists and returns whether they are equal or if there are differences.

        Args:
            list1 (List[str]): First list to compare.
            list2 (List[str]): Second list to compare.

        Returns:
            Tuple[bool, Optional[Tuple[List[str], List[str]]]]: True if lists are equal,
            otherwise False along with the differences.
        """
        set1 = set(list1)
        set2 = set(list2)

        if set1 == set2:
            logger.info("Lists are identical.")
            return True, None
        else:
            diff_elements_list1 = list(set1 - set2)
            diff_elements_list2 = list(set2 - set1)
            logger.info(f"Lists differ: {diff_elements_list1} vs {diff_elements_list2}")
            return False, (diff_elements_list1, diff_elements_list2)
