import matplotlib.pyplot as plt

import ml.datasets.blobs as blobs
from ml.models.neuron import Neuron


# Generate synthetic dataset
dataset = blobs()
X, y = dataset.generate(
    n_samples=100, n_centers=2, random_state=1234, cluster_std=[3.5, 3.5]
)

# Init neuron model
neuron = Neuron(n_parameters=2, bias=True)

# Fit model
neuron.fit(X, y)
print(neuron.weights)

# Predict labels
y_hats = neuron.predict(X)
print(y_hats)

# Plot
for px, py, lab, m in zip(X[:, 0], X[:, 1], y == 1.0, y_hats > 0.5):
    plt.scatter(px, py, marker="o" if lab else "+", color="blue" if m else "red")
plt.show()
