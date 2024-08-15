from src.core.predictors.openai_predictor import OpenAIPredictor
from src.core.predictors.huggingface_predictor import HuggingFacePredictor

class PredictionManager:
    """Manager class to handle predictions for a single model type."""

    def __init__(self, service, model_name: str, **kwargs):
        """
        Initializes the PredictionManager with the appropriate prediction class based on the model_name.

        Args:
            model_name (str): The name of the model ('openai' or the Hugging Face model name).
            **kwargs: Additional keyword arguments such as 'openai_api_key' and 'device'.
        """
        if service.lower() == 'openai':
            api_key = kwargs.get('api_key')
            if not api_key:
                raise ValueError("API key is required for OpenAI model.")
            self.predictor = OpenAIPredictor(model_name=model_name, api_key=api_key)
        else:
            device = kwargs.get('device', 'cuda')
            self.predictor = HuggingFacePredictor(model_name=model_name, device=device)

    def predict(self, messages, temperature: float = 1.0, **kwargs) -> str:
        """Generates a prediction using the initialized predictor.
        
        Args:
            input_data (str): The input data for which the prediction should be generated.
            **kwargs: Additional arguments for prediction (e.g., max_tokens, temperature).
        
        Returns:
            str: The prediction result.
        """
        return self.predictor.predict(messages, **kwargs)