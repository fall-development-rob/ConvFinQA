import json
from typing import Dict, List, Any, Optional
from pathlib import Path
import time
import pandas as pd
from .interfaces import HelpersInterface

class Helpers(HelpersInterface):
    def __init__(self, filepath):
        self.filepath = filepath

    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load and parse the ConvFinQA dataset."""
        with open(self.filepath, 'r') as f:
            data = json.load(f)
        return data

    def format_table(self, table_data: List[List[str]]) -> str:
        """Convert table data into a readable string format."""
        df = pd.DataFrame(table_data)
        table_str = df.to_string(index=False, header=False)
        return f"Table data:\n{table_str}"

    def prepare_context(self, entry: Dict[str, Any]) -> str:
        """Prepare the context by combining pre-text, table, and post-text."""
        pre_text = " ".join(entry.get("pre_text", []))
        post_text = " ".join(entry.get("post_text", []))
        table = self.format_table(entry.get("table", []))
        
        return f"""Context:
        Pre-text: {pre_text}
        
        {table}
        
        Post-text: {post_text}"""

    def get_qa(self, entry: Dict) -> Dict:
        """Extract QA data from entry regardless of structure."""
        if "qa" in entry:
            return entry["qa"]
        for key in ["qa_0", "qa_1", "qa_2"]:
            if key in entry:
                return entry[key]
        raise KeyError("No QA data found in entry")

    def print_running_accuracy(self, i, total, stats):
        """Print the running accuracy statistics."""
        total_correct = stats["exact_matches"] + stats["close_matches"]
        print(f"\nRunning Accuracy ({i}/{total}):")
        print(f"Exact Matches: {stats['exact_matches']}/{stats['processed']} ({stats['exact_matches']/stats['processed']:.1%})")
        print(f"Close Matches: {stats['close_matches']}/{stats['processed']} ({stats['close_matches']/stats['processed']:.1%})")
        print(f"Total Correct: {total_correct}/{stats['processed']} ({total_correct/stats['processed']:.1%})")

    def print_non_exact_match(self, i, question, expected, calculated, comparison):
        """Print details for non-exact matches."""
        print(f"\nQuestion {i}:")
        print(f"Q: {question}")
        print(f"Expected: {expected}")
        standardized_calculated = self.standardize_percentage(calculated) if '%' in expected else calculated
        print(f"Actual: {standardized_calculated}")
        if comparison.get("close_match"):
            print("Close match (within tolerance)")
        if comparison.get("error") is not None:
            print(f"Error margin: {comparison['error']:.4f}")
        print("-" * 50)

    def save_results(self, results, stats, save_dir="results"):
        """Save the evaluation results and statistics to a JSON file."""
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)

        total = len(results)
        data_to_save = {
            "total_questions": total,
            "exact_matches": stats['exact_matches'],
            "close_matches": stats['close_matches'],
            "total_correct": stats["exact_matches"] + stats["close_matches"],
            "accuracy": (stats["exact_matches"] + stats["close_matches"]) / total if total else 0,
            "detailed_results": results
        }

        with open(save_path / f"{timestamp}_results.json", "w") as f:
            json.dump(data_to_save, f, indent=2)

    def clean_response(self, response: str) -> str:
        """Remove unnecessary characters and trim whitespace."""
        return response.replace('\\', '').replace('$', '').replace(',', '').strip()

    def parse_number(self, value: str) -> Optional[float]:
        """
        Convert a string to a float, handling percentages.
        Returns None if conversion fails.
        """
        value = value.strip()
        try:
            if value.endswith('%'):
                return float(value.rstrip('%')) / 100
            return float(value)
        except ValueError:
            return None
               
    def standardize_percentage(self, value: Optional[float]) -> str:
        """Format the float as a percentage string with one decimal place."""
        if value is None:
            return "N/A"
        percentage = value * 100 if abs(value) < 1 else value
        return f"{percentage:.1f}%"