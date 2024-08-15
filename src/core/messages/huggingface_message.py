from src.core.messages.base import MessageGenerator

class HuggingFaceMessageGenerator(MessageGenerator):
    """Implementation of the MessageGenerator interface for Hugging Face models."""

    def generate_message(self, prompt: str, specialist: str = None) -> str:
        """Generates a formatted message for Hugging Face.
        
        Args:
            prompt (str): The input text for which the message should be generated.
            specialist (str, optional): The specialist content to include in the message.
        
        Returns:
            str: The formatted message as a string.
        """
        # Customize the message format for Hugging Face as needed
        return f"How would you respond to this text? {prompt}"