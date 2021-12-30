from typing import Tuple
import numpy as np

from ml import Model


class TreeNode:
    def __init__(
        self,
        X: np.array,
        y: np.array,
        deep: int,
        is_classification: bool,
        max_depth: int = 20,
        min_rows_to_split: int = 3,
        random_features_on_split: bool = False,
    ) -> None:
        self.X = X
        self.y = y
        self.left = None
        self.right = None
        self.split_point = None
        self.feature = None
        self.deep = deep
        self.is_classification = is_classification
        self.max_depth = max_depth
        self.min_rows_to_split = min_rows_to_split
        self.random_features_on_split = random_features_on_split
        # print("Deep %d" % deep)

    def mode(self, x: np.array) -> int:
        """
        Calculate the mode of an array values
        """
        vals, counts = np.unique(x, return_counts=True)
        return vals[np.argmax(counts)]

    def get_impurities(
        self,
        is_classification: bool,
        X: np.array,
        y: np.array,
        feature: int,
        split: float,
    ) -> Tuple[float, float, float, float, float]:
        """
        Calculate impurities for split, rigth and left nodes; given dataset,
        feature and the numerical split (or threshold).
        """
        if is_classification:
            return self.get_impurities_classification(X, y, feature, split)
        else:
            return self.get_impurities_regression(X, y, feature, split)

    def get_impurities_classification(self, X, y, feature, split):
        """
        Calculate impurities for a given split point, data and a feature
        """
        split_mask = X[:, feature] < split
        # Left split
        n_left = sum(split_mask == 1)
        left_split_ones = sum(y[split_mask == 1] == 1)
        left_split_zero = sum(y[split_mask == 1] == 0)
        impurity_left = 1 - (
            np.power(left_split_ones / n_left, 2)
            + np.power(left_split_zero / n_left, 2)
        )
        # Rigth split
        n_rigth = sum(split_mask == 0)
        rigth_split_ones = sum(y[split_mask == 0] == 1)
        rigth_split_zero = sum(y[split_mask == 0] == 0)
        impurity_rigth = 1 - (
            np.power(rigth_split_ones / n_rigth, 2)
            + np.power(rigth_split_zero / n_rigth, 2)
        )
        # Weights impurities
        impurity_left *= n_left / (n_left + n_rigth)
        # Check if no data in partition
        if np.isnan(impurity_left):
            impurity_left = 0
        # Check if no data in partition
        impurity_rigth *= n_rigth / (n_left + n_rigth)
        if np.isnan(impurity_rigth):
            impurity_rigth = 0
        # Calculate overall split impurity
        impurity_split = impurity_left + impurity_rigth

        # Return imputiries values
        return feature, impurity_split, impurity_rigth, impurity_left, split

    def get_impurities_regression(self, X, y, feature, split):
        """
        Calculate impurities using MSE creteria
        """
        # create split mask on dataset
        split_mask = X[:, feature] < split
        n_left = len(y[split_mask])
        n_right = len(y[~split_mask])
        # calculate MSE on left nd right
        avg_left = np.mean(y[split_mask])
        avg_right = np.mean(y[~split_mask])
        left_powers = np.power(y[split_mask] - avg_left, 2)
        right_powers = np.power(y[~split_mask] - avg_right, 2)
        # mse on nodes
        mse_left = np.sum(left_powers) / n_left
        mse_right = np.sum(right_powers) / n_right
        # total mse
        mse = (np.sum(left_powers) + np.sum(right_powers)) / (n_left + n_right)
        # return values
        return feature, mse, mse_right, mse_left, split

    def split(self):
        """
        Split data based on recursively Impurity criteria
        """
        n_rows = self.X.shape[0]
        n_features = self.X.shape[1]
        # Calculate all splits posibilities for every feature
        best_split = {
            "feature": 0,
            "impurity": np.Inf,
            "impurity_right": 0,
            "impurity_left": 0,
            "split": None,
        }
        features_list = range(n_features)
        for feature in features_list:
            # Sort by feature
            sorted_index = self.X[:, feature].argsort()
            self.X = self.X[sorted_index]
            self.y = self.y[sorted_index]
            # Loop over features averaging values (the thresholds)
            splits = []
            for ind in range(n_rows - 1):
                splits.append((self.X[ind, feature] + self.X[ind + 1, feature]) / 2)
            # Calculate partitions on this feature
            for split in np.unique(splits):
                (
                    feature,
                    impurity_split,
                    impurity_rigth,
                    impurity_left,
                    split,
                ) = self.get_impurities(
                    self.is_classification, self.X, self.y, feature, split
                )
                # Save best split point
                if impurity_split < best_split["impurity"]:
                    best_split["feature"] = feature
                    best_split["impurity"] = impurity_split
                    best_split["impurity_right"] = impurity_rigth
                    best_split["impurity_left"] = impurity_left
                    best_split["split"] = split
        self.split_point = best_split

        # Recursive calls
        # On right
        mask_right = self.X[:, best_split["feature"]] >= best_split["split"]

        if (
            best_split["impurity_right"] > 0
            and self.deep <= self.max_depth
            and len(np.unique(self.X[mask_right], axis=0)) > self.min_rows_to_split
        ):
            self.right = TreeNode(
                self.X[mask_right],
                self.y[mask_right],
                self.deep + 1,
                self.is_classification,
                self.max_depth,
            )
            self.right.split()
        else:
            if self.is_classification:
                self.split_point["class_rigth"] = self.mode(self.y[mask_right])
            else:
                self.split_point["class_rigth"] = np.mean(self.y[mask_right])
        # On left
        mask_left = self.X[:, best_split["feature"]] < best_split["split"]
        if (
            best_split["impurity_left"] > 0
            and self.deep <= self.max_depth
            and len(np.unique(self.X[mask_left], axis=0)) > self.min_rows_to_split
        ):
            self.left = TreeNode(
                self.X[mask_left],
                self.y[mask_left],
                self.deep + 1,
                self.is_classification,
                self.max_depth,
            )
            self.left.split()
        else:
            if self.is_classification:
                self.split_point["class_left"] = self.mode(self.y[mask_left])
            else:
                self.split_point["class_left"] = np.mean(self.y[mask_left])


class DecisionTree(Model):
    """
    Classification tree implementation
    """

    def __init__(
        self,
        is_classification: bool,
        max_depth: int = 20,
        random_state: int = 12,
    ) -> None:
        self.random_state = random_state
        self.is_classification = is_classification
        self.max_depth = max_depth

    def fit(self, X: np.array, y: np.array):
        """
        Fit Classification Tree model
        """
        # Recursive split process
        self.root = TreeNode(
            X,
            y,
            deep=0,
            is_classification=self.is_classification,
            max_depth=self.max_depth,
        )
        self.root.split()

    def predict(self, X) -> np.array:
        """
        Predict a batch of examples
        """
        y_hats = []
        # Loop over rows
        for i, x in enumerate(X):
            pointer = self.root  # default pointer on root
            while True:
                if x[pointer.split_point["feature"]] < pointer.split_point["split"]:
                    # Go left
                    if pointer.left is None:
                        y_hats.append(pointer.split_point["class_left"])
                        break
                    else:
                        pointer = pointer.left
                else:
                    # Go right
                    if pointer.right is None:
                        y_hats.append(pointer.split_point["class_rigth"])
                        break
                    else:
                        pointer = pointer.right
        return y_hats
