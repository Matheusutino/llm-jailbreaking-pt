import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from src.core.predictors.base import PredictionModel

class HuggingFacePredictor(PredictionModel):
    """Prediction model implementation for Hugging Face."""

    def __init__(self, model_name: str, device: str = 'cpu'):
        """
        Initializes the HuggingFacePredictor with a model name and another parameter.

        Args:
            model_name (str): The name of the model to use.
            device (str): Device to use for inference, 'cpu' or 'cuda'.
        """
        self.device = device
        self.model_name = model_name
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map='cuda' if self.device == 'gpu' else 'cpu',
            torch_dtype='auto',
            # low_cpu_mem_usage=True,
            # load_in_8bit=True,
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.pipeline = pipeline(
            'text-generation',
            model=self.model,
            tokenizer=self.tokenizer,
            # device = 0 if self.device == 'cuda' else -1,
        )

    def predict(self, messages, max_tokens: int = 1024, temperature: float = 0.3, do_sample:bool = True):
        """
        Generates text based on the input text using the specified model.

        Args:
            input_text (str): The input text to generate from.
            max_tokens (int): Maximum number of tokens to generate.
            temperature (float): Sampling temperature. Lower values make the output more deterministic.
            do_sample (bool): Whether to use sampling or greedy decoding.

        Returns:
            str: The generated text.
        """
        generation_args = {
            'max_new_tokens': max_tokens,
            'temperature': temperature,
            'do_sample': do_sample
        }
        try:
            output = self.pipeline(messages, **generation_args)
            return output[0]['generated_text']
        except Exception as e:
            raise RuntimeError(f"Error generating text: {e}")