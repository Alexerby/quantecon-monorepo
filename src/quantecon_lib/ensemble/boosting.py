import numpy as np
from ._base import BaseEnsemble
from ..tree.decision_trees import DecisionTreeRegressor

class GradientBoostingRegressor(BaseEnsemble):
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        super().__init__(n_estimators=n_estimators, max_depth=max_depth)
        self.learning_rate = learning_rate
        self.f0 = None

    def fit(self, X, y):
        X, y = self._validate_data(X, y)
        self.f0 = np.mean(y)
        current_preds = np.full(len(y), self.f0)
        self.models = []

        for _ in range(self.n_estimators):
            res = y - current_preds
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, res)
            
            tree_preds = tree.predict(X)
            current_preds += self.learning_rate * tree_preds
            self.models.append(tree)
            
        return self

    def predict(self, X):
        X = np.asarray(X)
        y_hat = np.full(len(X), self.f0)

        for tree in self.models:
            y_hat += self.learning_rate * tree.predict(X)

        return y_hat
