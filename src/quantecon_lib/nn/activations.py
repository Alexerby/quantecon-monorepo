"""Defines the activation functions used throughout the nn subpackage."""

import numpy as np

from .layers import Layer


class ReLU(Layer):
    def forward(self, x):
        """The definition of ReLU."""
        self.input = x
        return np.maximum(0, x)

    def backward(self, output_gradient):
        """Multiply with the gradient."""
        return output_gradient * (self.input > 0)


class Sigmoid(Layer):
    def forward(self, x):
        """The definition of Sigmoid."""
        self.output = 1 / (1 + np.exp(-x))
        self.output = np.where(
            x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x))
        )
        return self.output

    def backward(self, output_gradient):
        """Multiply with the gradient."""
        return output_gradient * (self.output * (1 - self.output))
