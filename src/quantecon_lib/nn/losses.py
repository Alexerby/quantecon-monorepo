from abc import ABC, abstractmethod

import numpy as np

class Loss(ABC):
    """Provides methods to calculate the forward pass (loss value)
    and the backward pass (derivative) for optimization."""
    
    @staticmethod
    @abstractmethod
    def forward(y_true, y_pred):
        pass

    @staticmethod
    @abstractmethod
    def backward(y_true, y_pred):
        pass


class MSE(Loss):
    """Mean Squared Error (MSE)"""
    @staticmethod
    def forward(y_true, y_pred):
        """The MSE cost function."""
        return np.mean(np.power(y_true - y_pred, 2))

    @staticmethod
    def backward(y_true, y_pred):
        """Derivative of MSE loss function."""
        return 2 * (y_pred - y_true) / y_true.size

class MAE(Loss):
    """Mean Absolute Error (MAE)."""
    @staticmethod
    def forward(y_true, y_pred):
        return np.absolute(y_true - y_pred) / y_true.size

    @staticmethod
    def backward(y_true, y_pred):
        return np.sign(y_true - y_pred) / y_true.size

