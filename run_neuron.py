import matplotlib.pyplot as plt

from ml.datasets.blobs import Blobs
from ml.models.neuron import Neuron

from ml.metrics.classification import ClassificationMetrics
from sklearn.metrics import confusion_matrix, classification_report


# Generate synthetic dataset
dataset = Blobs()
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
print(y_hats > 0.5)

# get accuracy
metrics = ClassificationMetrics(y, y_hats > 0.5)
print("Accuracy %f %%" % metrics.get_accuracy())
print("Confusion Matrix %f")
print(confusion_matrix(y, y_hats > 0.5))
print(classification_report(y_hats > 0.5, y))

# Plot
for px, py, lab, m in zip(X[:, 0], X[:, 1], y == 1.0, y_hats > 0.5):
    plt.scatter(px, py, marker="o" if lab else "+", color="blue" if m else "red")
plt.show()
