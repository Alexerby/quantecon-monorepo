import numpy as np

from .decision_trees import DecisionTreeRegressor

class GradientBoostingRegressor:
    def __init__(self, learning_rate, max_depth, M):
        self.M = M 
        self.learning_rate = learning_rate
        self.max_depth = max_depth 
        self.trees = []
        self.f0 = None

    def fit(self, X, y):
        self.f0 = np.mean(y)
        current_preds = np.full(len(y), self.f0)

        for _ in range(self.M):
            res = y - current_preds
            tree = DecisionTreeRegressor(self.max_depth)
            tree.fit(X, res)
            tree_preds = tree.predict(X)
            current_preds = current_preds + self.learning_rate * tree_preds
            self.trees.append(tree)
        return self

    def predict(self, X):
        y_hat = np.full(len(X), self.f0)

        for tree in self.trees:
            y_hat = y_hat + self.learning_rate * tree.predict(X)

        return y_hat


