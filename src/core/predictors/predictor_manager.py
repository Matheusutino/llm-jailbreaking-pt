from typing import List, Dict, Union
from src.core.predictors.openai_predictor import OpenAIPredictor
from src.core.predictors.maritaca_ai import MaritacaAIPredictor
from src.core.predictors.llama_predictor import LlamaCppPredictor
from src.core.predictors.huggingface_predictor import HuggingFacePredictor
from src.core.predictors.gemini_predictor import GeminiPredictor

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
            self.predictor = OpenAIPredictor(model_name=model_name, api_key=api_key)
        elif service.lower() == 'maritaca_ai':
            api_key = kwargs.get('api_key')
            self.predictor = MaritacaAIPredictor(model_name=model_name, api_key=api_key)
        elif service.lower() == 'llama_cpp':
            device = kwargs.get('device', 'gpu')
            self.predictor = LlamaCppPredictor(model_name=model_name, device=device)
        elif service.lower() == 'huggingface':
            device = kwargs.get('device', 'gpu')
            self.predictor = HuggingFacePredictor(model_name=model_name, device=device)
        elif service.lower() == 'gemini':
            api_key = kwargs.get('api_key')
            self.predictor = GeminiPredictor(model_name=model_name, api_key=api_key)

    def predict(self, messages: Union[str, List[Dict[str, str]]], temperature: float = 0.3, **kwargs) -> str:
        """Generates a prediction using the initialized predictor.
        
        Args:
            input_data (str): The input data for which the prediction should be generated.
            **kwargs: Additional arguments for prediction (e.g., max_tokens, temperature).
        
        Returns:
            str: The prediction result.
        """
        return self.predictor.predict(messages, **kwargs)