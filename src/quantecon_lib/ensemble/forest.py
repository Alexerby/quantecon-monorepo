import numpy as np
from ._base import ParallelEnsemble
from ..tree.decision_trees import DecisionTreeClassifier

class RandomForestClassifier(ParallelEnsemble):
    def __init__(self, n_estimators=100, max_features="sqrt", max_depth=None):
        super().__init__(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features)

    def fit(self, X, y):

        # Enforce numpy and get dimensions
        X, y = self._validate_data(X, y)
        n_samples, n_features = X.shape

        # Calculate feature budget m
        if self.max_features == "sqrt":
            m = int(np.sqrt(n_features))
        elif self.max_features == "log":
            m = int(np.log2(n_features))
        elif isinstance(self.max_features, int):
            m = self.max_features
        else:
            m = n_features

        self.models = [] # Clear stored trees

        for _ in range(self.n_estimators):
            # Bootstrap sampling (Bagging)
            boot_indices = self._get_bootstrap_indices(n_samples)

            X_resample = X[boot_indices]
            y_resample = y[boot_indices]

            tree = DecisionTreeClassifier(max_depth=self.max_depth, max_features=m)
            tree.fit(X_resample, y_resample)

            self.models.append(tree)

        return self

    def predict(self, X):
        X = np.asarray(X)
        
        # Row contains predictions for individual tree
        tree_preds = np.array([tree.predict(X) for tree in self.models])

        final_preds = []

        # Find the majority vote for each sample across all trees
        for i in range(tree_preds.shape[1]): 
            sample_votes = tree_preds[:, i]
            # Fixed the parenthesis bug: bincount first, then argmax
            most_frequent = np.bincount(sample_votes.astype(int)).argmax()
            final_preds.append(most_frequent)

        return np.array(final_preds)
