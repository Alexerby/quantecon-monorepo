import numpy as np

class BaseEnsemble:
    def __init__(self, n_estimators=100, max_depth=None, max_features=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features # Now it's stored here!
        self.models = []

    def _validate_data(self, X, y):
        return np.asarray(X), np.asarray(y)

class ParallelEnsemble(BaseEnsemble):
    def __init__(self, n_estimators=100, max_depth=None, max_features=None):
        super().__init__(n_estimators, max_depth, max_features)

    def _get_bootstrap_indices(self, n_samples):
        return np.random.choice(n_samples, size=n_samples, replace=True)
