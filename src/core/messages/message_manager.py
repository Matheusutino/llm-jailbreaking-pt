from src.core.messages.openai_message import OpenAIMessageGenerator
from src.core.messages.huggingface_message import HuggingFaceMessageGenerator

class MessageManager:
    """Manager class to handle message generation for different model types."""

    def __init__(self):
        """Initializes the MessageManager with supported message generator classes."""
        self.message_generators = {
            'openai': OpenAIMessageGenerator(),
            'huggingface': HuggingFaceMessageGenerator()
        }

    def generate_message(self, model_type: str, prompt: str, specialist: str = None) -> list:
        """Generates a message according to the specified model type.
        
        Args:
            model_type (str): The type of model ('openai' or 'huggingface').
            prompt (str): The input text for which the message should be generated.
            specialist (str, optional): The specialist content to include in the message.
        
        Returns:
            list: The formatted message as a list of dictionaries.
        """
        message_generator = self.message_generators.get(model_type.lower())
        if not message_generator:
            raise ValueError("Unsupported model type. Choose between 'openai' or 'huggingface'.")

        return message_generator.generate_message(prompt, specialist)