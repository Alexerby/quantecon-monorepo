import numpy as np
from .._base import BaseNeuralNetwork

from quantecon_lib.core.utils import timer


class FeedForwardNetwork(BaseNeuralNetwork):
    @timer
    def train(
        self,
        x_train,
        y_train,
        x_test,
        y_test,
        epochs: int,
        batch_size: int = 32,
        tol: float = 1e-6,
    ):
        prev_test_loss: float = float("inf")
        n_samples = x_train.shape[0]

        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            x_shuffled = x_train[indices]
            y_shuffled = y_train[indices]

            for i in range(0, n_samples, batch_size):
                x_batch = x_shuffled[i : i + batch_size]
                y_batch = y_shuffled[i : i + batch_size]

                H = x_batch
                for layer in self.layers:
                    H = layer.forward(H)

                loss_gradient = self._loss_function.backward(y_batch, H)
                gradient = loss_gradient

                for layer in reversed(self.layers):
                    gradient = layer.backward(gradient)

                self.optimizer.step(self.layers)

            train_preds = self._predict(x_train)
            test_preds = self._predict(x_test)

            loss_train_val = self._loss_function.forward(y_train, train_preds)
            loss_test_val = self._loss_function.forward(y_test, test_preds)

            if loss_train_val is None or loss_test_val is None:
                raise ValueError(
                    "Loss function returned None instead of a scalar value."
                )

            current_loss_train = float(loss_train_val)
            current_loss_test = float(loss_test_val)

            self._history["training_loss"].append(current_loss_train)
            self._history["test_loss"].append(current_loss_test)

            if abs(prev_test_loss - current_loss_test) <= tol:
                print(f"Converged at epoch {epoch}.")
                break

            prev_test_loss = current_loss_test

            if epoch % 10 == 0:
                print(
                    f"Epoch {epoch}. Loss (Training, Test): ({current_loss_train:.6f}, {current_loss_test:.6f})"
                )
