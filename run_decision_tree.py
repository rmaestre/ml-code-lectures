import matplotlib.pyplot as plt

import ml.datasets.blobs as blobs
from ml.models.decision_tree import DecisionTree
from ml.metrics.classification import ClassificationMetrics

is_classification = True
dataset = blobs()

if is_classification:
    # Generate synthetic dataset
    X, y = dataset.generate(
        n_samples=100, n_centers=2, random_state=1234, cluster_std=[4, 4]
    )
    # Plot points
    color_map = {0: "red", 1: "blue"}
    plt.scatter(X[:, 0], X[:, 1], color=[color_map[i] for i in y])
    plt.show()

    # Init classification tree model
    c_tree = DecisionTree(is_classification=True, max_depth=1)

    # Fit model
    c_tree.fit(X, y)

    # Get predictions
    y_hats = c_tree.predict(X)

    # get accuracy
    metrics = ClassificationMetrics(y, y_hats)
    print("Accuracy %f %%" % metrics.get_accuracy())

    # Assert all predictions are correct
    assert sum(y - y_hats) == 0

else:
    X, y = dataset.generate_line(
        n_samples=100, n_features=1, noise=50, random_state=1234
    )
    # Plot points
    plt.scatter(X, y)
    plt.show()

    # Init classification tree model
    c_tree = DecisionTree(is_classification=False)
    # Fit model
    c_tree.fit(X, y)
    # Get predictions
    y_hats = c_tree.predict(X)

    a = 1 + 1
