from abc import ABC, abstractmethod

class MessageGenerator(ABC):
    """Abstract base class for generating messages."""

    @abstractmethod
    def generate_message(self, prompt: str, specialist: str = None) -> list:
        """Generates the expected message format for a specific model.
        
        Args:
            prompt (str): The input text for which the message should be generated.
            specialist (str, optional): The specialist content to include in the message.
        
        Returns:
            list: The formatted message as a list of dictionaries.
        """
        pass