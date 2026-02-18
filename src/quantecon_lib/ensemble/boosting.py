import numpy as np
from ._base import BaseEnsemble
from ..tree.decision_trees import DecisionTreeClassifier, DecisionTreeRegressor

class AdaBoostClassifier(BaseEnsemble):
    def __init__(self, n_estimators, max_depth=1) -> None:
        super().__init__(n_estimators=n_estimators, max_depth=max_depth)

        self.alphas = []
        self.models = []
        self.n_classes = None

    def fit(self, X, y):
        n_samples = X.shape[0]
        self.n_classes = len(np.unique(y))

        # Initialize equal weighting
        weights = np.full(n_samples, 1 / n_samples)

        # Do the boosting, BUT with the twist that we now pass
        # weights to the data points (IMPORTANT!).
        for _ in range(self.n_estimators):
            tree = DecisionTreeClassifier(max_depth=self.max_depth)
            tree.fit(X, y, weights=weights) # Weights passed here

            # Calculate the weighted error rate
            # If it predicts a data point which the waits deem important 
            # wrong, then this will yield a much larger error rate
            # than predicting a point wrong which was given 
            # less importance by the weights.
            preds = tree.predict(X)
            incorrect = (preds != y)
            error = np.sum(weights[incorrect] / np.sum(weights))

            # If error is larger than simply guessing, then break
            # the loop
            if error >= 1 - (1 / self.n_classes): 
                break
            # If the learner is perfect (error=0), then return 
            if error <= 0: # why
                self.models.append(tree)
                self.alphas.append(1.0)
                break

            # How much volume this tree gets in the vote ("meritocracy")
            alpha = np.log((1.0 - error) / (error + 1e-10)) + np.log(self.n_classes - 1.0)
            weights *= np.exp(alpha * incorrect) # Re-weighting
            weights /= np.sum(weights) # Normalization

            self.models.append(tree)
            self.alphas.append(alpha)

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
        for alpha, model in zip(self.alphas, self.models):
            preds = model.predict(X)
            for i in range(n_cls):
                class_scores[:, i] += alpha * (preds == unique_classes[i])

        return np.argmax(class_scores, axis=1)

class GradientBoostingRegressor(BaseEnsemble):
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        super().__init__(n_estimators=n_estimators, max_depth=max_depth)
        self.learning_rate = learning_rate
        self.f0 = None
        self.history_ = None

    def _fit(self, X, y, eval_set=None, patience=10, tol=1e-4):
        self.f0 = np.mean(y)
        current_preds = np.full(len(y), self.f0)
        self.models = []
        
        self.history_ = {'train_loss': []}

        if eval_set:
            X_val, y_val = eval_set
            val_preds = np.full(len(y_val), self.f0)
            best_val_loss = np.inf
            no_improvement_count = 0
            
            self.history_['val_loss'] = []
            self.history_['val_r2'] = []
            
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
            self.history_['train_loss'].append(train_loss)

            if eval_set:
                val_preds += self.learning_rate * tree.predict(X_val)
                current_val_loss = np.mean((y_val - val_preds)**2)
                self.history_['val_loss'].append(current_val_loss)

                ss_res_val = np.sum((y_val - val_preds)**2)
                current_score = 1 - (ss_res_val / ss_tot_val)
                self.history_["val_r2"].append(current_score)

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

