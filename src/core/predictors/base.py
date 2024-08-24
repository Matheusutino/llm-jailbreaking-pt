from abc import ABC, abstractmethod
from typing import List, Dict, Union

class PredictionModel(ABC):
    """Abstract base class for prediction models."""

    @abstractmethod
    def predict(self, messages: Union[str, List[Dict[str, str]]], max_tokens: int, temperature: float) -> str:
        """
        Abstract method to perform prediction.

        Args:
            messages (Union[str, List[Dict[str, str]]]): The input data for prediction. 
                It can be a single string or a list of dictionaries, where each dictionary
                represents a message with string keys and values.

        Returns:
            str: The prediction result.
        """
        pass
