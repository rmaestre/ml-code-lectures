import numpy as np


class ClassificationMetrics:
    def __init__(self, y: np.array, y_hats: np.array) -> None:
        self.y_hats = y_hats
        self.y = y

    def get_accuracy(self) -> float:
        """
        We follow the next formula:
        accuracy = correct predictions / total predictions * 100
        """
        total = len(self.y)
        correct = np.count_nonzero(self.y == self.y_hats)
        return (correct / total) * 100
