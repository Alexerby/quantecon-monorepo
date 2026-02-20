"""
TODO: BaggingClassifier
"""

import numpy as np
from ._base import ParallelEnsemble
from ..tree.decision_trees import DecisionTreeRegressor


class BaggingRegressor(ParallelEnsemble):
    def __init__(self, n_estimators=100, max_depth=None):
        super().__init__(n_estimators=n_estimators, max_depth=max_depth)
        self.r2 = None

    def fit(self, X, y):
        X, y = self._validate_data(X, y)

        n_samples = X.shape[0]
        self.models = []

        for _ in range(self.n_estimators):
            boot_indices = self._get_bootstrap_indices(n_samples)

            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X[boot_indices], y[boot_indices])
            self.models.append(tree)

        self.r2 = self._calculate_r2(y, self.predict(X))
        return self

    def predict(self, X):
        all_preds = np.array([tree.predict(np.asarray(X)) for tree in self.models])
        return np.mean(all_preds, axis=0)

    def _calculate_r2(self, y, y_hat):
        rss = np.sum((y - y_hat) ** 2)
        tss = np.sum((y - np.mean(y)) ** 2)
        return 1 - (rss / tss)
