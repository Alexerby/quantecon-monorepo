import numpy as np
from ._base import BaseEnsemble
from ..tree.decision_trees import DecisionTreeRegressor

class GradientBoostingRegressor(BaseEnsemble):
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        super().__init__(n_estimators=n_estimators, max_depth=max_depth)
        self.learning_rate = learning_rate
        self.f0 = None
        self._history = None

    def _fit(self, X, y, eval_set=None, patience=10, tol=1e-4):
        self.f0 = np.mean(y)
        current_preds = np.full(len(y), self.f0)
        self.models = []
        
        self._history = {"train_loss": []}

        if eval_set:
            X_val, y_val = eval_set
            val_preds = np.full(len(y_val), self.f0)
            best_val_loss = np.inf
            no_improvement_count = 0
            
            self._history["val_loss"] = []
            self._history["val_r2"] = []
            
            y_val_mean = np.mean(y_val)
            ss_tot_val = np.sum((y_val - y_val_mean)**2)

        for _ in range(self.n_estimators):
            res = y - current_preds
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, res)

            tree_preds = tree.predict(X)
            current_preds += self.learning_rate * tree_preds
            self.models.append(tree)

            train_loss = np.mean((y - current_preds)**2)
            self._history["train_loss"].append(train_loss)

            if eval_set:
                val_preds += self.learning_rate * tree.predict(X_val)
                current_val_loss = np.mean((y_val - val_preds)**2)
                self._history["val_loss"].append(current_val_loss)

                ss_res_val = np.sum((y_val - val_preds)**2)
                current_score = 1 - (ss_res_val / ss_tot_val)
                self._history["val_r2"].append(current_score)

                if current_val_loss < best_val_loss - tol:
                    best_val_loss = current_val_loss
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                
                if no_improvement_count >= patience:
                    break
            
            return self

    def predict(self, X):
        X = np.asarray(X)
        y_hat = np.full(len(X), self.f0)

        for tree in self.models:
            y_hat += self.learning_rate * tree.predict(X)

        return y_hat

    def score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)

