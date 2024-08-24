import os
from dotenv import load_dotenv, find_dotenv
from src.core.messages.message_manager import MessageManager
from src.core.predictors.predictor_manager import PredictionManager
from src.core.utils import load_yaml

load_dotenv(find_dotenv())

class Evaluation:
    def __init__(self, service: str = 'openai', model_name: str = 'gpt-4o-mini'):
        """
        Initialize the Evaluation class.

        Args:
            service (str): The service name (default is 'openai').
            model_name (str): The model name (default is 'gpt-4o-mini').
            api_key (str): The API key for the service (default is None).
        """
        self.service = service
        self.model_name = model_name
        self.api_key = os.environ.get('OPENAI_API_KEY')
        self.message_configs = load_yaml('configs/message.yaml')

        self.prediction_manager = PredictionManager(service=self.service, model_name=self.model_name, api_key=self.api_key)
        self.message_manager = MessageManager()
        

    def evaluate_result(self, result: str, message_type: str = 'openai') -> str:
        """
        Evaluate prediction.

        Args:
            result (str): The result from predictions.
            message_type (str): The type of message (e.g., 'openai').
            message_configs (dict): The message configuration dictionary.

        Returns:
            str: The evaluation result.
        """
        prompt_evaluate = self.message_configs.get('evaluate_response_prompt').format(text=result)
        specialist_evaluate = self.message_configs.get('evaluate_response_specialist')

        messages_evaluate = self.message_manager.generate_message(message_type, prompt_evaluate, specialist_evaluate)
        result_evaluate = self.prediction_manager.predict(messages=messages_evaluate)

        return result_evaluate
