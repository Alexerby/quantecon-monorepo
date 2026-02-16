import numpy as np 

from .decision_trees import DecisionTreeRegressor

class BaggingRegressor:
    def __init__(self, B=100, max_depth=3):
        self.B = B
        self.max_depth = max_depth
        self.models = []
        self.r2 = None

    def fit(self, X, y):
        n = len(y)
        indices = np.arange(n)
        
        for _ in range(self.B):
            boot_indices = np.random.choice(indices, size=n, replace=True)
            X_resample = X.iloc[boot_indices]
            y_resample = y[boot_indices]

            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X_resample, y_resample)
            self.models.append(tree)
        
        # Calculate R2 during fit if desired
        y_hat = self.predict(X)
        self.r2 = self._calculate_r2(y, y_hat)
        return self

    def predict(self, X):
        # Average predictions from all stored trees
        all_preds = np.array([tree.predict(X) for tree in self.models])
        return np.mean(all_preds, axis=0)

    def _calculate_r2(self, y, y_hat):
        rss = np.sum((y - y_hat) ** 2)
        ess = np.sum((y_hat - np.mean(y)) ** 2)
        return ess / (ess + rss)
