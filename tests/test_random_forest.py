import numpy as np

from ml.datasets.blobs import Blobs
from ml.models.random_forest import RandomForest
from ml.models.decision_tree import DecisionTree
from ml.metrics.classification import ClassificationMetrics


def get_data():
    dataset = Blobs()
    X, y = dataset.generate(
        n_samples=100, n_centers=2, random_state=1234, cluster_std=[4, 4]
    )
    return X, y


def test_init():
    X, y = get_data()

    rf = RandomForest(n_trees=3, max_depth=3, batch_size=30, is_classification=True)
    rf.fit(X, y)

    # Assert not None root and first splits
    assert len(rf.forest) is 3
    # Aseert forest are DecisionTree
    for tree in rf.forest:
        assert isinstance(tree, DecisionTree)


def test_depth_hyperparameter():
    X, y = get_data()

    rf = RandomForest(n_trees=3, max_depth=10, batch_size=30, is_classification=True)
    rf.fit(X, y)
    acc_overfit = ClassificationMetrics(y, rf.predict(X)).get_accuracy()

    rf = RandomForest(n_trees=1, max_depth=1, batch_size=30, is_classification=True)
    rf.fit(X, y)
    acc_underfit = ClassificationMetrics(y, rf.predict(X)).get_accuracy()

    assert acc_overfit > acc_underfit
