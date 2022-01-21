from typing import Tuple
import numpy as np

from ml import Model
from ml.models.decision_tree import DecisionTree


class Gbm(Model):
    """
    Gradient boost machine for regression and classifications
    """

    def __init__(
        self, n_trees: int, is_classification: bool = True, random_state: int = 12
    ) -> None:
        """
        Init parameters
        """
        np.random.seed(random_state)
        self.n_trees = n_trees
        self.is_classification = is_classification
        self.trees = {}
        self.f0 = None

    def fit(
        self,
        X: np.array,
        y: np.array,
        learning_rate: float = 0.01,
        max_depth: int = 1,
        batch_size: int = 10,
        epochs: int = 200,
    ) -> None:
        """
        Fit GBM model by using weak learner trees
        """
        self.learning_rate = learning_rate
        self.f0 = np.mean(y)
        fi = self.f0
        for i in range(self.n_trees):
            self.trees[i] = DecisionTree(
                is_classification=self.is_classification, max_depth=max_depth
            )
            pseudo_residuals = y - fi
            self.trees[i].fit(X, pseudo_residuals)
            fi += learning_rate * self.trees[i].predict(X)

    def predict(self, X: np.array) -> np.array:
        """
        Predict a batch of examples
        """
        summ = self.f0 + self.learning_rate * np.sum(
            [self.trees[i].predict(X) for i in self.trees], axis=0
        )
        return summ
