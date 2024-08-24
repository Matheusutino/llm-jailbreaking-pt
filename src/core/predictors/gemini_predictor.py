import os
import google.generativeai as genai
from src.core.predictors.base import PredictionModel
from dotenv import load_dotenv, find_dotenv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential
)
from google.generativeai.types import HarmCategory, HarmBlockThreshold


load_dotenv(find_dotenv())

class GeminiPredictor(PredictionModel):
    def __init__(self, model_name: str, api_key: str = None):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get('GEMINI_API_KEY')
        genai.configure(api_key=self.api_key)
    

    @retry(wait=wait_random_exponential(min=2, max=5), stop=stop_after_attempt(5))
    def predict(self, messages: str, max_tokens: int = 1024, temperature: float = 0.3):
        try:
            generation_config = genai.types.GenerationConfig(
                candidate_count=1,
                max_output_tokens=max_tokens,
                temperature=temperature,
            )
            
            model = genai.GenerativeModel(
                model_name="gemini-1.5-flash",
                generation_config=generation_config,
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                }
            )
            output = model.generate_content(messages)

            return output.text
        except Exception as e:
            raise RuntimeError(f"Error generating text: {e}")