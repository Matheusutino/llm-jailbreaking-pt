import os
from openai import OpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential
)
from typing import List, Dict
from dotenv import load_dotenv, find_dotenv
from src.core.predictors.base import PredictionModel

load_dotenv(find_dotenv())

class MaritacaAIPredictor(PredictionModel):
    """Prediction model implementation for OpenAI."""

    def __init__(self, model_name: str, base_url: str = 'https://chat.maritaca.ai/api', api_key: str = None):
        """
        Initializes the OpenAIPredictor with an API key and another parameter.
        """
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key or os.environ.get('MARITACA_AI_API_KEY')
        self.client = OpenAI(
            api_key=self.api_key,
            base_url = base_url
        )

    @retry(wait=wait_random_exponential(min=2, max=5), stop=stop_after_attempt(5))
    def predict(self, messages: List[Dict[str, str]], max_tokens: int = 1024, temperature: float = 0.3):
        """
        Predicts using OpenAI's API.

        Args:
            input_data (str): The input data to be processed for prediction.

        Returns:
            str: The prediction result.

        Raises:
            Exception: If there is an error with the API call.
        """
        try:
            response = self.client.chat.completions.create(
                messages=messages,
                model=self.model_name,
                max_tokens = max_tokens,
                temperature=temperature
            )   
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"OpenAI API error: {e}")