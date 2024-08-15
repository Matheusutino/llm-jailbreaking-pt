import os
from openai import OpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    before_sleep_log
)
from src.core.predictors.base import PredictionModel

class OpenAIPredictor(PredictionModel):
    """Prediction model implementation for OpenAI."""

    def __init__(self, model_name: str, api_key: str):
        """
        Initializes the OpenAIPredictor with an API key and another parameter.
        """
        self.model_name = model_name
        self.client = OpenAI(
            api_key=api_key,
        )

    @retry(wait=wait_random_exponential(min=2, max=5), stop=stop_after_attempt(5))
    def predict(self, messages, temperature: float = 1.0):
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
                temperature=temperature
            )   
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"OpenAI API error: {e}")