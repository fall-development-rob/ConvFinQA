import time
import json
from .prompt import create_prompt
from .interfaces import ModelHandlerInterface, HelpersInterface, ExtractorInterface, EvaluatorInterface

class Evaluator(EvaluatorInterface):
    def __init__(self, model_handler: ModelHandlerInterface, utils: HelpersInterface, extract: ExtractorInterface):
        self.model = model_handler
        self.utils = utils
        self.extract = extract

    def evaluate_dataset(self, num_samples: int = None):
        """Evaluate the dataset and compute accuracy scores."""
        num_samples = 5
        dataset = self.utils.load_dataset()

        if num_samples:
            dataset = dataset[:num_samples]

        results = []
        running_stats = {
            "processed": 0,
            "exact_matches": 0,
            "close_matches": 0
        }

        total_entries = len(dataset)
        print(f"Evaluating {total_entries} questions...")

        for i, entry in enumerate(dataset, start=1):
            try:
                qa_data = self.utils.get_qa(entry)
                question = qa_data["question"]
                expected = qa_data["answer"]

                # Generate model response using the ModelHandler
                prompt = create_prompt(self.utils.prepare_context(entry), question)
                response_text = self.model.generate_response(prompt)

                calculated = self.extract.extract_final_answer(response_text)
                comparison = self.extract.compare_answers(calculated, expected)

                # Update statistics
                running_stats["processed"] += 1
                if comparison.get("exact_match"):
                    running_stats["exact_matches"] += 1
                elif comparison.get("close_match"):
                    running_stats["close_matches"] += 1

                # Store the result
                results.append({
                    "question": question,
                    "expected": expected,
                    "calculated": calculated,
                    "comparison": comparison,
                    "response": response_text
                })

                # Print running accuracy
                self.utils.print_running_accuracy(i, total_entries, running_stats)

                # Print details if not an exact match
                if not comparison.get("exact_match"):
                    self.utils.print_non_exact_match(i, question, expected, calculated, comparison)

                time.sleep(1)

            except Exception as e:
                print(f"\nError processing entry {i}")
                print(f"Error type: {type(e).__name__}")
                print(f"Error details: {str(e)}")
                entry_preview = json.dumps(entry, indent=2)[:200] + "..."
                print(f"Entry structure: {entry_preview}")
                continue

        # Final statistics
        print("\n=== Final Results ===")
        total = len(results)
        if total:
            total_correct = running_stats["exact_matches"] + running_stats["close_matches"]
            print(f"Total Questions: {total}")
            print(f"Exact Matches: {running_stats['exact_matches']} ({running_stats['exact_matches']/total:.1%})")
            print(f"Close Matches: {running_stats['close_matches']} ({running_stats['close_matches']/total:.1%})")
            print(f"Total Correct: {total_correct} ({total_correct/total:.1%})")
        else:
            print("No results to display.")

        # Save the results
        self.utils.save_results(results, running_stats)

        return results
