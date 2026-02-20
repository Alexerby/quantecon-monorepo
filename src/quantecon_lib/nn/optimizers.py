from abc import ABC, abstractmethod


class Optimizer(ABC):
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    @abstractmethod
    def step(self, layers):
        pass


class SGD(Optimizer):
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def step(self, layers):
        """Iterates through layers and updates parameters using stored gradients."""
        for layer in layers:
            # Check if layer has parameters to update
            if hasattr(layer, "W"):
                layer.W -= self.learning_rate * layer.dW
                layer.b -= self.learning_rate * layer.db
