from src.extraction import Extractor
from src.evaluation import Evaluator
from src.helper import Helpers
from src.model_handler import ModelHandler

def main():
    dataset_path = 'data/train.json'
    num_samples = 5  

    # Initialize the ModelHandler with desired configurations
    model_handler = ModelHandler(model_name="llama3", temperature=0.1)
    utils = Helpers(dataset_path)
    extractor = Extractor(utils)
    evaluator = Evaluator(model_handler, utils, extractor)

    # Evaluate the dataset
    evaluator.evaluate_dataset(num_samples)

if __name__ == "__main__":
    main()
