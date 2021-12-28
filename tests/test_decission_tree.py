import numpy as np

from ml.datasets.blobs import Blobs
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

    dec_tree = DecisionTree(is_classification=True)
    dec_tree.fit(X, y)

    # Assert not None root and first splits
    assert dec_tree.root is not None
    assert dec_tree.root.left is not None
    assert dec_tree.root.right is not None

    # Assert deeps
    assert dec_tree.root.deep == 0
    assert dec_tree.root.left.deep == 1

    # Assert leaf node
    cond1 = dec_tree.root.left.left.split_point["class_left"]
    cond2 = dec_tree.root.left.left.left
    assert cond1 == 0 and cond2 is None


def test_depth_hyperparameter():
    X, y = get_data()

    dec_tree = DecisionTree(is_classification=True)
    dec_tree.fit(X, y)
    acc_overfit = ClassificationMetrics(y, dec_tree.predict(X)).get_accuracy()

    dec_tree = DecisionTree(is_classification=True, max_depth=2)
    dec_tree.fit(X, y)
    acc_underfit = ClassificationMetrics(y, dec_tree.predict(X)).get_accuracy()

    assert acc_overfit > acc_underfit
