"""
TODO: AdaBoostRegressor
"""

from abc import ABC, abstractmethod
import numpy as np
from ._base import BaseEnsemble
from ..tree.decision_trees import DecisionTreeClassifier, DecisionTreeRegressor


class AdaBoost(BaseEnsemble, ABC):
    def __init__(self, n_estimators=100, max_depth=1) -> None:
        super().__init__(n_estimators=n_estimators, max_depth=max_depth)

        self.weights = []
        self.n_classes = None

    @abstractmethod
    def _fit(self, X, y):
        """Each subclass must implement its own boosting logic."""
        pass

    @abstractmethod
    def predict(self, X):
        """Classifiers use voting; Regressors use weighted medians."""
        pass


class AdaBoostClassifier(AdaBoost):
    def _fit(self, X, y):
        n_samples = X.shape[0]
        self.n_classes = len(np.unique(y))

        # Initialize equal weighting
        weights = np.full(n_samples, 1 / n_samples)

        # Do the boosting, BUT with the twist that we now pass
        # weights to the data points (IMPORTANT!).
        for _ in range(self.n_estimators):
            tree = DecisionTreeClassifier(max_depth=self.max_depth)
            tree.fit(X, y, weights=weights)  # Weights passed here

            # Calculate the weighted error rate
            # If it predicts a data point which the waits deem important
            # wrong, then this will yield a much larger error rate
            # than predicting a point wrong which was given
            # less importance by the weights.
            preds = tree.predict(X)
            incorrect = preds != y
            error = np.sum(weights[incorrect] / np.sum(weights))

            # If error is larger than simply guessing, then break
            # the loop
            if error >= 1 - (1 / self.n_classes):
                break
            # If the learner is perfect (error=0), then return
            if error <= 0:  # why
                self.models.append(tree)
                self.weights.append(1.0)
                break

            # How much volume this tree gets in the vote ("meritocracy")
            alpha = np.log((1.0 - error) / (error + 1e-10)) + np.log(
                self.n_classes - 1.0
            )
            weights *= np.exp(alpha * incorrect)  # Re-weighting
            weights /= np.sum(weights)  # Normalization

            self.models.append(tree)
            self.weights.append(alpha)

        return self

    def predict(self, X):
        """
        SAMME voting algorithm.
        "Stagewise additive modelling using a multi-class exponential loss function".


        Args:
            arg: X.

        Returns:
            type: np.ndarray.
        """

        X = np.asarray(X)
        n_samples = X.shape[0]

        if self.n_classes is None:
            raise ValueError("Model must be fitted before calling predict.")

        n_cls = self.n_classes

        # Rows: data points; Columns: class
        class_scores = np.zeros((n_samples, n_cls))
        unique_classes = np.arange(n_cls)

        # Translation of:
        # C(x) = \arg\max_k \sum_{m=1}^M \alpha_m\cdot 1(T_m(x) = k)
        #
        # Get predictions and place the predictions in the correct column,
        # where the column is the class.
        for alpha, model in zip(self.weights, self.models):
            preds = model.predict(X)
            for i in range(n_cls):
                class_scores[:, i] += alpha * (preds == unique_classes[i])

        return np.argmax(class_scores, axis=1)


class AdaBoostRegressor(AdaBoost):
    def _fit(self, X, y):
        self.f0 = np.mean(y)
        n_samples = X.shape[0]

        # Weight initialization
        weights = np.full(len(y), 1 / n_samples)

        for _ in range(self.n_estimators):
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, y, weights=weights)
            preds = tree.predict(X)

            errors = np.abs(y - preds)
            max_error = np.max(errors)

            if max_error == 0:
                break

            loss = errors / max_error

            avg_loss = np.sum(weights * loss)

            # If the model is worse than random guessing, stop
            if avg_loss >= 0.5:
                break

            beta = avg_loss / (1 - avg_loss)

            # Store the model and its weight (log(1/beta): trust score)
            self.models.append(tree)
            self.weights.append(beta)

            # Update and normalize weights
            weights = weights * np.power(beta, 1 - loss)
            weights /= np.sum(weights)

        return self

    def predict(self, X):
        X = np.asarray(X)
        if not self.models:
            return np.full(X.shape[0], self.f0)

        # Get predictions from all trees
        # Shape: (n_samples, n_trees)
        all_preds = np.array([tree.predict(X) for tree in self.models]).T

        # Convert betas to trust scores (alphas)
        # Trees with lower beta get higher trust
        betas = np.array(self.weights)
        alphas = np.log(1.0 / betas)

        # Calculate weighted median for each sample
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples)

        for i in range(n_samples):
            # Sort individual sample predictions
            sample_preds = all_preds[i]
            sort_idx = np.argsort(sample_preds)

            sorted_preds = sample_preds[sort_idx]
            sorted_alphas = alphas[sort_idx]

            # Find where cumulative weight exceeds 50% of total trust
            cum_weight = np.cumsum(sorted_alphas)
            cutoff = 0.5 * np.sum(sorted_alphas)
            median_idx = np.searchsorted(cum_weight, cutoff)

            # Handle edge case where searchsorted might go out of bounds
            median_idx = min(median_idx, len(sorted_preds) - 1)
            predictions[i] = sorted_preds[median_idx]

        return predictions
