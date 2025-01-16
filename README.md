# ConvFinQA

This code is for a llm powered prototype that answers questions using financial statement.

## Usage

1. Ensure you have all the necessary packages installed by running poetry:

```bash
poetry install
```

2. Ensure ollama is downloaded, installed and running in your local machine. Use the command `ollama serve` to run it. 

3. Pull the LLM from the ollama server. In this instance we are using llama3.

```bash
ollama pull llama3
```

5. Evaluate the performance of the language model by running `main.py`. 

