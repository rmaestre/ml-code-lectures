from typing import Tuple, Dict
import numpy as np

from ml import Model
from ml.models.decision_tree import DecisionTree

from ml.metrics.classification import ClassificationMetrics
from ml.metrics.regression import RegressionMetrics


class RandomForest(Model):
    """
    Simple Random Forest ml model
    """

    def __init__(
        self,
        n_trees: int,
        max_depth: int,
        batch_size: int,
        is_classification: bool,
        random_state: int = 12,
    ) -> None:
        """
        Init paramters
        """
        np.random.seed(random_state)
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.batch_size = batch_size
        self.is_classification = is_classification
        self.random_state = random_state
        self.forest = []

    def mode(self, x: np.array) -> int:
        """
        Calculate the mode of an array values
        """
        vals, counts = np.unique(x, return_counts=True)
        return vals[np.argmax(counts)]

    def fit(self, X: np.array, y: np.array) -> None:
        """
        Fit Random Forest
        """
        for tree in range(self.n_trees):
            # Select random batch
            batch_index = np.random.choice(X.shape[0], self.batch_size)
            # Calculate batch
            X_batch = X[batch_index]
            y_batch = y[batch_index]
            # Select random features
            random_features = np.random.randint(X.shape[1]) + 1
            features_index = np.random.choice(
                X.shape[1], random_features, replace=False
            )
            # Select a random depth
            random_depth = np.random.randint(self.max_depth) + 1
            random_depth = np.max([2, random_depth])
            # Fit Deccision Tree
            print(
                "Training %s tree in the forest with %s features and %s max depth"
                % (tree, random_features, random_depth)
            )
            self.forest.append(
                DecisionTree(
                    self.is_classification, random_depth, random_features_on_split=True
                )
            )
            self.forest[-1].fit(X_batch[:, features_index], y_batch)

    def predict(self, X: np.array) -> np.array:
        """
        Predict a batch of examples
        """
        # Collect all predictions in the batch
        predictions = []
        if self.forest is not None:
            for tree in self.forest:
                predictions.append(tree.predict(X))
        predictions = np.asarray(predictions)
        # Select kind of prediction
        y_hats = []
        for column in predictions.T:
            if self.is_classification:
                y_hats.append(self.mode(column))
            else:
                y_hats.append(np.mean(column))
        return np.asarray(y_hats)

    def feature_importance(self, X: np.array, y: np.array) -> Tuple[np.array]:
        """
        Feature importance calculated on column (feature) permutation method
        """
        if len(self.forest) == 0:
            print("Please, fit the model before")
            return None
        else:
            errors_permuted_features = {key: 0 for key in range(X.shape[1])}
            # Calculate the error produced
            y_hats = self.predict(X)
            metric = None
            error = None
            if self.is_classification:
                metric = ClassificationMetrics(y, y_hats)
                error = metric.get_accuracy()
            else:
                metric = RegressionMetrics(y, y_hats)
                error = metric.get_mse()

            # Feature permutation
            for feature in range(X.shape[1]):
                # Preparation to permute feature
                for k in range(5):
                    X_perm = X.copy()
                    # select ramdon feature permutation
                    np.random.shuffle(X_perm[:, feature])
                    y_hats_perm = self.predict(X_perm)
                    # Get permutation error
                    error_perm = None
                    if self.is_classification:
                        metric = ClassificationMetrics(y, y_hats_perm)
                        error_perm = metric.get_accuracy()
                    else:
                        metric = RegressionMetrics(y, y_hats_perm)
                        error_perm = metric.get_mse()
                    errors_permuted_features[feature] += error_perm
                errors_permuted_features[feature] /= 5
                errors_permuted_features[feature] -= error
            return errors_permuted_features
