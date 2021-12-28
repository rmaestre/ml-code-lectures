import matplotlib.pyplot as plt
from numpy.core.arrayprint import set_printoptions
from sklearn.datasets import make_moons
import numpy as np
from nnfs.datasets import spiral_data

from ml.models.mlp import (
    Layer_Dense,
    Activation_ReLU,
    Activation_Softmax,
    Activation_Softmax_Loss_CategoricalCrossentropy,
    Optimizer_SGD,
)


num_samples = 10
num_features = 2
X, y = make_moons(n_samples=num_samples, noise=0.1)

# Plot data
"""
colors = {0: "red", 1: "blue"}
for px, py, lab in zip(X[:, 0], X[:, 1], y == 1.0):
    plt.scatter(px, py, marker="o" if lab else "+", color="blue")
plt.show()
"""

# -----------------------------------------------------------------
# Forward steps

# Create one layer dense and step a forward
layer = Layer_Dense(num_features, n_neurons=3)
layer.forward(X)
print(layer.output)

# Create one layer dense, and ReLU activation and step a forward
activation1 = Activation_ReLU()
activation1.forward(layer.output)
print(activation1.output)

softmax = Activation_Softmax()
softmax.forward([[1, 2, 3]])
print(softmax.output)


# -----------------------------------------------------------------
# Backward steps

num_samples = 100
X, y = spiral_data(samples=100, classes=3)

# Plot data
class_color = {0: "red", 1: "blue", 2: "green"}
class_marker = {0: "o", 1: "+", 2: "D"}
for px, py, lab in zip(X[:, 0], X[:, 1], y):
    plt.scatter(px, py, marker=class_marker[lab], color="blue")
plt.show()


dense1 = Layer_Dense(2, 200)
# Create ReLU activation (to be used with Dense layer):
activation1 = Activation_ReLU()
# Create second Dense layer with 64 input features (as we take output # of previous layer here) and 3 output values (output values)
dense2 = Layer_Dense(200, 3)
# Create Softmax classifier's combined loss and activation
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()  # Create optimizer
optimizer = Optimizer_SGD(learning_rate=0.1)


for epoch in range(30001):
    # Perform a forward pass of our training data through this layer
    dense1.forward(X)
    # Perform a forward pass through activation function
    # takes the output of first dense layer here
    activation1.forward(dense1.output)
    # Perform a forward pass through second Dense layer
    # takes outputs of activation function of first layer as inputs
    dense2.forward(activation1.output)
    # Perform a forward pass through the activation/loss function
    # # takes the output of second dense layer here and returns loss
    loss = loss_activation.forward(dense2.output, y)

    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)
    if not epoch % 1000:
        print(f"epoch: {epoch}, " + f"acc: {accuracy:.3f}, " + f"loss: {loss:.3f}")

    # Backward pass
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)
    # Update weights and biases
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)


predictions = np.argmax(loss_activation.output, axis=1)
# Plot data
for px, py, lab, m in zip(X[:, 0], X[:, 1], y, predictions):
    plt.scatter(px, py, marker=class_marker[lab], color=class_color[m])
plt.show()
