from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List

class ExtractorInterface(ABC):

    @abstractmethod
    def extract_final_answer(self, response: str) -> Optional[float]:
        pass

    @abstractmethod
    def compare_answers(self, calculated: Optional[float], expected: str) -> Dict[str, Optional[float]]:
        pass

class EvaluatorInterface(ABC):
    @abstractmethod
    def evaluate_dataset(self, num_samples: int = None):
        pass

class HelpersInterface(ABC):

    @abstractmethod
    def load_dataset(self) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def format_table(self, table_data: List[List[str]]) -> str:
        pass

    @abstractmethod
    def prepare_context(self, entry: Dict[str, Any]) -> str:
        pass

    @abstractmethod
    def get_qa(self, entry: Dict) -> Dict:
        pass

    @abstractmethod
    def print_running_accuracy(self, i, total, stats):
        pass

    @abstractmethod
    def print_non_exact_match(self, i, question, expected, calculated, comparison):
        pass
    
    @abstractmethod
    def save_results(self, results, stats, save_dir="results"):
        pass

    @abstractmethod
    def clean_response(self, response: str) -> str:
        pass

    @abstractmethod
    def parse_number(self, value: str) -> Optional[float]:
        pass

    @abstractmethod
    def standardize_percentage(self, value: Optional[float]) -> str:
        pass

class ModelHandlerInterface(ABC):
    @abstractmethod
    def generate_response(self, prompt: str, stream: bool = False):
        pass
    