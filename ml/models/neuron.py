from typing import Tuple
import numpy as np

from ml import Model


class Neuron(Model):
    """
    Single neuron ml model
    """

    def __init__(
        self, n_parameters: int, bias: bool = True, random_state: int = 12
    ) -> None:
        """
        Init paramters, including bias as optional
        """
        np.random.seed(12)
        # N weigths randomly created
        self.n_parameters = n_parameters
        self.bias = bias
        self.weights = np.random.normal(
            0, 1, n_parameters + 1 if self.bias else n_parameters
        )

    def get_minibatches(self, X: np.array, y: np.array, batch_size: int) -> Tuple:
        """
        Generate batches creating windows on the complete examples
        """
        batches = []
        n = len(X)
        for index in range(0, n + 1, batch_size):
            if index == 0:
                ant = 0
            else:
                batches.append({"features": X[ant:index], "labels": y[ant:index]})
                ant = index
        return batches

    def fit(
        self,
        X: np.array,
        y: np.array,
        learning_rate: float = 0.001,
        batch_size: int = 10,
        epochs: int = 200,
    ) -> None:
        """
        Fit Neuron model
        """
        mini_batches = self.get_minibatches(X, y, batch_size)
        for e in range(epochs):
            for mini_batch in mini_batches:
                # Get matrix of features and labels
                # Insert bias term on minibatch features
                if self.bias:
                    features = np.insert(
                        mini_batch["features"],
                        mini_batch["features"].shape[1],
                        1,
                        axis=1,
                    )
                else:
                    features = mini_batch["features"]
                # Get true labels
                y = mini_batch["labels"]
                # Get predictions
                y_hat = self.predict(mini_batch["features"])
                # Get errors
                errors = y_hat - y
                # Updates
                gradients = np.asarray(
                    [np.sum(e) / len(mini_batch) for e in errors * features.T]
                )
                # Update weigths
                self.weights -= learning_rate * gradients

    def sigmoid(self, x: np.array) -> np.array:
        """
        Sigmoid function returning probability to get the "1" class
        """
        return 1 / (1 + np.exp(-x))

    def predict(self, X: np.array) -> np.array:
        """
        Predict a batch of examples
        """
        if self.bias:
            return self.sigmoid(
                np.dot(self.weights, np.insert(X, X.shape[1], 1, axis=1).T)
            )
        else:
            return self.sigmoid(np.dot(self.weights, X.T))
