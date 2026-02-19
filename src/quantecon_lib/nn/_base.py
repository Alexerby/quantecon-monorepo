from abc import ABC, abstractmethod
import numpy as np
from quantecon_lib.nn.optimizers import Optimizer
from quantecon_lib.nn.losses import Loss
from .layers import Layer
from typing import Type

class BaseNeuralNetwork(ABC):
    def __init__(
        self,
        layers: list[Layer],
        optimizer: Optimizer,
        loss_function: Type[Loss]
    ):
        self.layers = layers
        self.optimizer = optimizer
        self._history = {"training_loss": [], "test_loss": []}
        self._loss_function = loss_function

    @abstractmethod
    def train(self, x_train, y_train, x_test, y_test, epochs: int, tol: float = 1e-6):
        pass

    def _predict(self, X, epsilon=1e-5):
        H = X
        for layer in self.layers:
            H = layer.forward(H)
        return np.where(np.abs(H) < epsilon, 0, H) if H is not None else H
