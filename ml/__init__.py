from abc import ABC, abstractmethod


class Model(ABC):
    """
    model with generic operations
    """

    @abstractmethod
    def fit(self):
        """
        Fit model and find parameters value
        """
        pass

    @abstractmethod
    def predict(self):
        """
        Predict labels on a given dataset
        """
        pass
