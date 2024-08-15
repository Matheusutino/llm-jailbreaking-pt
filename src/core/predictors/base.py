from abc import ABC, abstractmethod

class PredictionModel(ABC):
    """Abstract base class for prediction models."""

    @abstractmethod
    def predict(self, input_data: str):
        """
        Abstract method to perform prediction.

        Args:
            input_data (str): The input data to be processed for prediction.

        Returns:
            str: The prediction result.
        """
        pass