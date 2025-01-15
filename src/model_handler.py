from ollama import Client
from .interfaces import ModelHandlerInterface

class ModelHandler(ModelHandlerInterface):
    def __init__(self, model_name: str = "llama3", temperature: float = 0.1):
        self.client = Client()
        self.model_name = model_name
        self.temperature = temperature

    def generate_response(self, prompt: str, stream: bool = False):
        """Generate a response from the model based on the given prompt."""
        try:
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                stream=stream,
                options={"temperature": self.temperature}
            )
            return response['response']
        except Exception as e:
            raise RuntimeError(f"Model generation failed: {e}")
