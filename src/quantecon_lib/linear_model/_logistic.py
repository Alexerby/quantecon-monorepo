import numpy as np

from ._base import BaseEstimator


class LogisticRegressor(BaseEstimator):
    def __init__(self, learning_rate=0.01, tol=1e-6, n_iterations=10_000) -> None:
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.tol = tol
        self.weights = None
        self.bias = None
        self._history = {"cv_status": 0, "outcome_error": None}

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.n_iterations):
            model_linear = np.dot(X, self.weights) + self.bias
            probability = self._sigmoid(model_linear) # 
            error = probability - y

            dw = (1 / n_samples) * np.dot(X.T, error)
            db = (1 / n_samples) * np.sum(error)

            # Check convergence
            if np.linalg.norm(dw) <= self.tol:
                self._history["cv_status"] = 1
                break

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
        # Store the mean absolute error (MAE) or the final gradient norm
        self._history["outcome_error"] = np.mean(np.abs(error))
        return self

    def predict_proba(self, X):
        if self.weights is None:
            raise ValueError("Model must be fitted before predicting.")

        X = np.asarray(X)
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
