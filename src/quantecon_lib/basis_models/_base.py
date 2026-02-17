import numpy as np

class BaseBasisModel:
    def __init__(self):
        self.beta = None
        self.knots = None

    def _validate_data(self, X, y=None):
        X = np.asarray(X)
        if X.ndim == 1: # if 1D
            X = X.reshape(-1, 1) # force to 2D
        if y is not None:
            y = np.asarray(y)
        return X, y

    def fit(self, X, y):
        X, y = self._validate_data(X, y)
        H = self._transform(X)
        
        self.beta = np.linalg.pinv(H.T @ H) @ H.T @ y
        return self

    def predict(self, X):
        X, _ = self._validate_data(X)
        H = self._transform(X)
        return H @ self.beta

    def _transform(self, X):
        raise NotImplementedError("Subclasses must implement _transform")


def get_quantile_knots(X, n_knots):
    quantiles = np.linspace(0, 100, n_knots + 2)[1:-1]
    return np.percentile(X, quantiles)
