import matplotlib.pyplot as plt
import numpy as np
import pprint

from ml.datasets.blobs import Blobs
from ml.models.random_forest import RandomForest
from ml.metrics.classification import ClassificationMetrics
from ml.metrics.regression import RegressionMetrics


is_classification = True
dataset = Blobs()

if is_classification:
    # Generate synthetic dataset
    X, y = dataset.generate(
        n_samples=500, n_centers=2, n_features=2, random_state=1234, cluster_std=[1, 1]
    )
    # Add variable noise
    X[:, 0] = np.random.uniform(0, 10, X.shape[0])

    # Plot points
    color_map = {0: "red", 1: "blue"}
    plt.scatter(X[:, 0], y, color=[color_map[i] for i in y], marker="x")
    plt.title("Feature 0 and label")
    plt.show()
    plt.scatter(X[:, 1], y, color=[color_map[i] for i in y], marker="x")
    plt.title("Feature 1 and label")
    plt.show()

    # Init classification tree model
    rf = RandomForest(n_trees=10, max_depth=5, batch_size=250, is_classification=True)

    # Fit model
    rf.fit(X, y)

    # Get predictions
    y_hats = rf.predict(X)

    # get accuracy
    metrics = ClassificationMetrics(y, y_hats)
    print("Accuracy %f %%" % metrics.get_accuracy())

    # calculate feature importance
    importances = rf.feature_importance(X, y, permutation_rounds=30)
    pprint.pprint(importances)
    # Plot bars
    plt.bar(importances.keys(), importances.values())
    plt.title("Feature importance (the less the better)")
    plt.show()

else:
    X, y = dataset.generate_line(
        n_samples=500, n_features=5, noise=1, random_state=1234
    )
    y = y / np.mean(y)  # Scale
    # Add noise to one variable
    X[:, 0] += np.random.normal(X[:, 0], 10)

    # Plot points
    plt.scatter(X[:, 0], y, color="blue")
    plt.scatter(X[:, 1], y, color="red")
    plt.show()

    # Init classification tree model
    rf = RandomForest(n_trees=20, max_depth=4, batch_size=250, is_classification=False)
    # Fit model
    rf.fit(X, y)
    # Get predictions
    y_hats = rf.predict(X)

    # get accuracy
    metrics = RegressionMetrics(y, y_hats)
    print("MSE %f" % metrics.get_mse())

    plt.hist(y - y_hats)
    plt.show()

    # calculate feature importance
    importances = rf.feature_importance(X, y)
    pprint.pprint(importances)
    # Plot bars
    plt.bar(importances.keys(), importances.values())
    plt.show()
