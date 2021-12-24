from typing import Union
import numpy as np

from ml import Model


class Kmeans(Model):
    """
    Kmeans ml model
    """

    def __init__(self) -> None:
        """
        Init paramters
        """
        self.centroids = None
        self.error = None
        self.inertia = np.Inf

    def distance(self, x: np.array, y: np.array) -> float:
        """
        Calculate distance between one vector and
        a n-dimensional centroid vector
        """
        return np.sum(np.square(x - y), axis=1)

    def set_randomd_centroids(
        self, X: np.array, n_centroids: int, random_state: int
    ) -> None:
        """
        Set centroids
        """
        np.random.seed(random_state)
        self.centroids = X[np.random.choice(X.shape[0], n_centroids)]

    def predict(self, X: np.array) -> np.array:
        """
        Predict centroid membership of a given X dataset
        """
        membership = np.asarray(
            [np.argmin(self.distance(x, self.centroids)) for x in X]
        )
        return membership

    def fit(
        self, X: np.array, n_centers: int, n_iterations: int, random_state=12
    ) -> None:
        """
        Fit kmeans model, meaning to find correct values for centers minimizing error
        """
        # Get centroids
        if self.centroids is None:
            self.set_randomd_centroids(X, n_centers, random_state)
        # Iterations
        for _ in range(n_iterations):
            # loop over points and find membership, using euclidean distance
            membership = [np.argmin(self.distance(x, self.centroids)) for x in X]
            # For every row in X, calculate closest centroid
            centroid_summ_all_points = {k: None for k in range(n_centers)}
            centroid_counts = {k: 0 for k in range(n_centers)}
            # Loop over each X and its closest "centroid" or "label"
            for i, centroid_id in enumerate(membership):
                if centroid_summ_all_points[centroid_id] is None:
                    # Assign point to closest centroid
                    centroid_summ_all_points[centroid_id] = X[i].copy()
                else:
                    # Summ of all points corresponding to a
                    centroid_summ_all_points[centroid_id] += X[i]
                centroid_counts[centroid_id] += 1
            # Check no-membership to avoid raise execeptions
            if 0 in [v for v in centroid_counts.values()]:
                raise Exception("At least one centroid has not a member assigment")
            # Calculate new centroids
            self.centroids = np.asarray(
                [
                    centroid_summ_all_points[k] / centroid_counts[k]
                    for k in centroid_summ_all_points
                ]
            )

    def intertia(X: np.array, membership: np.array) -> float:
        """
        Intertia measurement of cluster performance

        X = [[1,2], [2,3], [1,5], ...]
        membership = [0,1,0,...]
        self.centroids
        """
        return np.Inf
