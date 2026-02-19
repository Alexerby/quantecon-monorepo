"""
TODO: AdaBoostRegressor
"""


import numpy as np
from ._base import BaseEnsemble
from ..tree.decision_trees import DecisionTreeClassifier

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

