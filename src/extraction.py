from typing import Optional, Dict
import re
from .interfaces import HelpersInterface, ExtractorInterface

class Extractor(ExtractorInterface):

    def __init__(self, utils: HelpersInterface):
        self.utils = utils
        
    def extract_final_answer(self, response: str) -> Optional[float]:
        """Extract the final numerical answer from the response."""
        response = self.utils.clean_response(response)

        # Define regex patterns to capture the answer
        patterns = [
            r'FINAL_?ANSWER:\s*(-?\d+\.?\d*%?)',
            r'\\boxed{([^}]+)}',
            r'answer(?:\s+is)?:\s*(-?\d+\.?\d*%?)',
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                answer = self.utils.parse_number(match.group(1))
                if answer is not None:
                    return answer

        # Fallback: Extract the last numerical value that's likely the final answer
        matches = re.findall(r'-?\d+\.?\d*%?', response)
        for value in reversed(matches):
            answer = self.utils.parse_number(value)
            if answer is not None:
                return answer

        return None

    def compare_answers(self, calculated: Optional[float], expected: str) -> Dict[str, Optional[float]]:
        """Compare calculated and expected answers with flexible precision."""
        if calculated is None:
            return {
                "is_correct": False,
                "exact_match": False,
                "close_match": False,
                "error": None
            }

        expected = self.utils.clean_response(expected)
        expected_val = self.utils.parse_number(expected)

        if expected_val is None:
            return {
                "is_correct": False,
                "exact_match": False,
                "close_match": False,
                "error": None
            }

        error = abs(calculated - expected_val)
        exact_match = error < 0.0001
        close_match = error < 0.01  # 1% tolerance

        return {
            "is_correct": exact_match or close_match,
            "exact_match": exact_match,
            "close_match": close_match,
            "error": error
        }
