import numpy as np

class BaseEnsemble:
    def __init__(self, n_estimators=100, max_depth=None, max_features=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.models = []

    def fit(self, X, y, **kwargs):
        # Centralized validation logic
        X, y = self._validate_data(X, y)
        return self._fit(X, y, **kwargs)

    def _validate_data(self, X, y):
        return np.asarray(X), np.asarray(y)

    def _fit(self, X, y, **kwargs):
        # This acts as a safety net
        raise NotImplementedError("Subclasses must implement _fit")

class ParallelEnsemble(BaseEnsemble):
    def __init__(self, n_estimators=100, max_depth=None, max_features=None):
        super().__init__(n_estimators, max_depth, max_features)

    def _get_bootstrap_indices(self, n_samples):
        return np.random.choice(n_samples, size=n_samples, replace=True)
