import matplotlib.pyplot as plt

import ml.datasets.blobs as blobs
from ml.models.kmeans import Kmeans


# Generate synthetic dataset
dataset = blobs()
X, y = dataset.generate(
    n_samples=100, n_centers=3, random_state=1234, cluster_std=[1.2, 2.4, 2.0]
)

plt.scatter(X[:, 0], X[:, 1])
plt.show()

# Init kmeans model
kmeans = Kmeans()
kmeans.fit(X, n_centers=3, n_iterations=3)

# print centroids
print(kmeans.centroids)

# Plot centroids
plt.scatter(X[:, 0], X[:, 1])
plt.scatter(kmeans.centroids[0, 0], kmeans.centroids[0, 1], marker="x", s=150)
plt.scatter(kmeans.centroids[1, 0], kmeans.centroids[1, 1], marker="x", s=150)
plt.scatter(kmeans.centroids[2, 0], kmeans.centroids[2, 1], marker="x", s=150)
plt.show()

# Get predictions
y_hat = kmeans.predict(X)
print(y)
print(y_hat)
