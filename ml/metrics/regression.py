import numpy as np


class RegressionMetrics:
    def __init__(self, y: np.array, y_hats: np.array) -> None:
        self.y_hats = y_hats
        self.y = y

    def get_mse(self) -> float:
        """
        We follow the next formula:
        accuracy = correct predictions / total predictions * 100
        """
        return np.mean(np.power(self.y - self.y_hats, 2))
