from typing import Tuple
import numpy as np
from sklearn.datasets import make_blobs, make_regression


class Blobs:
    def generate(
        self,
        n_samples: int,
        n_centers: int,
        random_state: int,
        cluster_std: np.array = None,
        n_features: int = 2,
    ) -> Tuple[np.array, np.array]:
        """
        Create N blobs fom sklearn.dataset generators
        """
        # Check if std is none, if true start by dafault
        if cluster_std is None:
            cluster_std = [0.1 for i in range(n_centers)]
        if n_centers != len(cluster_std):
            raise Exception(
                "Number of centers (%d) and its std dimension (%s) are not equal"
                % (n_centers, len(cluster_std))
            )
        # Generate data
        X, y = make_blobs(
            n_features=n_features,
            n_samples=n_samples,
            centers=n_centers,
            random_state=random_state,
            cluster_std=cluster_std,
        )
        return X, y

    def generate_real_blob(self) -> Tuple[np.array, np.array]:
        """
        Another blob generation without parameter
        """
        X, y = make_blobs(
            n_samples=100,
            centers=2,
            random_state=123,
            cluster_std=[100, 100],
        )
        return X, y

    def generate_line(
        self, n_samples: int, n_features: int, noise: int, random_state: int
    ) -> Tuple[np.array, np.array]:
        """
        Generate a line as a regression dataset
        """
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            noise=noise,
            random_state=random_state,
        )
        return X, y
