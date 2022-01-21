import matplotlib.pyplot as plt

from ml.datasets.blobs import Blobs
from ml.models.gbm import Gbm

from ml.metrics.regression import RegressionMetrics

from sklearn import ensemble
import numpy as np

# Generate synthetic dataset
dataset = Blobs()
X, y = dataset.generate_line(n_samples=10, n_features=1, noise=1, random_state=1234)
y = y / np.mean(y)  # Scale


# Fit model
gbm = Gbm(n_trees=4, is_classification=False)
gbm.fit(X, y, learning_rate=0.3, max_depth=1)

# Get predictions
y_hats = gbm.predict(X)

# get accuracy from GBM
metrics = RegressionMetrics(y, y_hats)
print("MSE %f" % metrics.get_mse())

params = {
    "n_estimators": 4,
    "max_depth": 1,
    "learning_rate": 0.3,
    "loss": "squared_error",
}
reg = ensemble.GradientBoostingRegressor(**params)
reg.fit(X, y)
# get accuracy from Sklearn GBM
metrics = RegressionMetrics(y, reg.predict(X))
print("MSE (sklearn) %f" % metrics.get_mse())

# Plot comparative figures
plt.scatter(reg.predict(X), y, color="red")
plt.scatter(y_hats, y, color="blue")
plt.show()
