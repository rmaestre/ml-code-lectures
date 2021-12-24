import numpy as np

import ml.datasets.blobs as blobs
from ml.models.neuron import Neuron


def get_data():
    dataset = blobs()
    X, y = dataset.generate(
        n_samples=100, n_centers=2, random_state=1234, cluster_std=[3.5, 3.5]
    )
    return X, y


def test_init():
    X, y = get_data()

    neuron = Neuron(n_parameters=2, bias=True)
    neuron.fit(X, y)

    # Assert weigths and bias are correct
    assert neuron.weights.shape[0] == 3
    assert 0.0 not in neuron.weights


def test_predictions():
    X, y = get_data()

    neuron = Neuron(n_parameters=2, bias=True)
    neuron.fit(X, y)

    y_hats = neuron.predict(X)
    # All predictions are between 0 and 1
    assert ((y_hats <= 1.0) & (y_hats >= 0.0)).all()
