import numpy as np

import ml.datasets.blobs as blobs
from ml.models.kmeans import Kmeans


def get_data():
    dataset = blobs()
    X, y = dataset.generate(
        n_samples=100, n_centers=3, random_state=1234, cluster_std=[1.2, 2.4, 2.0]
    )
    return X, y


def test_init():
    X, _ = get_data()

    kmeans = Kmeans()
    kmeans.fit(X, n_centers=3, n_iterations=3)

    # Assert centroids length
    assert len(kmeans.centroids) == 3
    # Assert not None centroids
    for centroid in kmeans.centroids:
        assert centroid is not None
        # Assert centroid has value
        for coord in centroid:
            assert coord != 0.0


def test_distance():
    X, _ = get_data()

    kmeans = Kmeans()
    kmeans.fit(X, n_centers=3, n_iterations=3)

    # Get one feature
    distances = kmeans.distance(X[0], kmeans.centroids)

    assert distances.shape[0] == 3
    assert distances[0] != distances[1]
    assert distances[0] != distances[2]
    assert distances[1] != distances[2]


def test_predict():
    X, _ = get_data()

    kmeans = Kmeans()
    kmeans.fit(X, n_centers=3, n_iterations=3)

    y_hats = kmeans.predict(X)
    # Same y_hats length tha X
    assert y_hats.shape[0] == X.shape[0]
    # Labels are the same than the centroids number
    assert np.unique(y_hats).shape[0] == 3
