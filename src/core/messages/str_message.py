from src.core.messages.base import MessageGenerator

class StrMessageGenerator(MessageGenerator):
    """
    A message generator that returns the input prompt as the message.

    This class implements the `generate_message` method to simply return the 
    provided prompt string without any modifications. It is useful in scenarios 
    where the prompt itself is the desired message for further processing.

    Args:
        prompt (str): The input string to be returned as the message.

    Returns:
        str: The generated message, which is the same as the input prompt.
    """

    def generate_message(self, prompt: str, specialist = None) -> str:
        return prompt
