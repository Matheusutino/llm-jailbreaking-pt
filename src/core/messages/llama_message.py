from src.core.messages.base import MessageGenerator

class LlamaMessageGenerator(MessageGenerator):
    """Implementation of the MessageGenerator interface for Llama 3.1."""

    def generate_message(self, prompt: str, specialist: str = None) -> str:
        """Generates a formatted message for Llama.
        
        Args:
            prompt (str): The input text for which the message should be generated.
            specialist (str, optional): The specialist content to include in the message.
        
        Returns:
            str: The formatted message as a string.
        """
        message = "<|begin_of_text|>"

        if specialist:
            message += "<|start_header_id|>system<|end_header_id|>"
            message += specialist + "<|eot_id|>"

        message += "<|start_header_id|>user<|end_header_id|>"
        message += prompt + "<|eot_id|>"
        message += "<|start_header_id|>assistant<|end_header_id|>"

        return message