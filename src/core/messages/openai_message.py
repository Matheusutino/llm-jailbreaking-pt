from src.core.messages.base import MessageGenerator

class OpenAIMessageGenerator(MessageGenerator):
    """Implementation of the MessageGenerator interface for OpenAI models."""

    def generate_message(self, prompt: str, specialist: str = None) -> list:
        """Generates a formatted message for OpenAI.
        
        Args:
            prompt (str): The input text for which the message should be generated.
            specialist (str, optional): The specialist content to include in the message.
        
        Returns:
            list: The formatted message as a list of dictionaries.
        """
        messages = []   

        if specialist:
            messages.append({"role": "system", "content": specialist})

        messages.append({"role": "user", "content": prompt})

        return messages