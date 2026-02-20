from abc import ABC, abstractmethod

import numpy as np


class Layer(ABC):
    """Promise the rest of the network that any object which is called a Layer
    will have a forward() and backward() method.
    """

    def __init__(self) -> None:
        self.input = None
        self.output = None

    @abstractmethod
    def forward(self, input_data):
        pass

    @abstractmethod
    def backward(self, input_data):
        pass


class Dense(Layer):
    def __init__(self, in_dim, out_dim, init="he"):
        scale = np.sqrt(2.0 / in_dim) if init == "he" else np.sqrt(1.0 / in_dim)
        self.W = np.random.randn(in_dim, out_dim) * scale
        self.b = np.zeros((1, out_dim))

    def forward(self, X):
        """Returns the pre-activation Z = XW + b."""
        self.X = X
        self.output_data = np.dot(self.X, self.W) + self.b
        return self.output_data

    def backward(self, output_gradient):
        """Perform the backward pass of backpropagation.

        Args:
            output_gradient: The gradient coming into this layer
            from the output direction.

        Returns:
            type: np.ndarray.
        """

        # Weight gradient:
        # Dim_{in} x Dim_{out} = (Dim_{in} x Batch) x (Batch x Dim_{out}).
        # Multiplies each feature \vec{x} with its corresponding error
        # and sums over the batch.
        # 1. Calculate and STORE gradients internally
        self.dW = np.dot(self.X.T, output_gradient)
        self.db = np.sum(output_gradient, axis=0, keepdims=True)

        # 2. Calculate the signal for the previous layer
        # Note: We use the W before the optimizer modifies it
        input_gradient = np.dot(output_gradient, self.W.T)

        return input_gradient
