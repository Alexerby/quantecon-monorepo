from typing import Optional
import numpy as np


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class BaseDecisionTree:
    def __init__(self, max_depth: Optional[int]=None, max_features=None):
        self.max_depth = max_depth
        self.max_features = max_features
        self.root = None
    
    def fit(self, X, y):

        X = np.asarray(X)
        y = np.asarray(y)

        self.root = self._grow_tree(X, y)
        return self

    def predict(self, X):
        # For each vector \vec{x} \in \mathbf{X}
        return np.array([self._traverse(x, self.root) for x in X]) 

    def _grow_tree(self, X, y, depth=0):
        """For fitting the tree."""
        n_samples, _ = X.shape
        n_labels = len(np.unique(y))

        # STOPPING CONDITION: This is what kills the recursion
        if (
            (self.max_depth is not None and depth >= self.max_depth) or 
            n_labels <= 1 or n_samples < 2
        ):
            return Node(value=self._calculate_leaf_value(y))

        split = self._find_best_split(X, y)

        # If it is mathematically impossible to split the data more
        # then return the node, as this node is terminal.
        if not split:
            return Node(value=np.mean(y))

        # RECURSION: Fit the rec
        left_child = self._grow_tree(
            X[split["left_mask"]],
            y[split["left_mask"]],
            depth + 1
        )

        right_child = self._grow_tree(
            X[split["right_mask"]],
            y[split["right_mask"]],
            depth + 1
        )

        return Node(
            feature=split["feature_idx"],
            threshold=split["threshold"],
            left=left_child,
            right=right_child
        )

    def _traverse(self, x, node):

        # Does the node have a value?
        # False: It's a split node, skip this statement
        # True: This is a terminal node, return the value of this node
        #
        # STOPPING CONDITION: This is the statement which kills the recursion
        if node.value is not None:
            return node.value

        # Because this is a split node we now want to 
        # go down the tree in either the right or left 
        # direction, depending on which side of the threshold 
        # we are at.
        # 
        # If we are at the left side of the threshold, then 
        # traverse left, otherwise traverse right
        if x[node.feature] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)


    def _find_best_split(self, X, y): 
        """Greedy search strategy for the best split point."""
        best_gain = -float("inf")
        split_info = None

        # If there are fewer than 2 unique values in X, we can't split
        if X.shape[0] < 2:
            return None

        n_features = X.shape[1]

        if self.max_features is None: # Standard bagging
            feature_indices = np.arange(n_features)

        else: #random forest (pick m features at random for this split)
            num_to_sample = min(n_features, self.max_features)
            feature_indices = np.random.choice(
                np.arange(n_features),
                size=num_to_sample,
                replace=False
            )


        
        for feature_idx in feature_indices:

            # Get unique values of the feature
            unique_vals = np.unique(X[:, feature_idx]) 

            if len(unique_vals) < 2: # Skip features with no variation
                continue

            # Suppose we have 
            # [10, 20, 30, 40], then the first slices is
            # [20, 30, 40] and the second one is [10, 20, 30].
            # Thereafter, we add the two array and divide by 2
            # [30, 50, 70] / 2 is therefore the outcome, which is 
            # [15, 25, 35]
            #
            # This is one search strategy out of many, 
            # but as we are in a continuous numerical space, 
            # there are technically an infinite number of splits
            # we could make, so this is our greedy search strategy.
            thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2 

            # Evaluate all these candidate threshold splits,
            # and return the best split
            for threshold in thresholds:
                left_mask = X[:, feature_idx] < threshold
                right_mask = ~left_mask

                # Skip splits that result in empty leaves
                if len(y[left_mask]) == 0 or len(y[right_mask]) == 0:
                    continue

                gain = self._calculate_information_gain(
                    y, 
                    y[left_mask], 
                    y[right_mask]
                )

                if gain > best_gain:
                    best_gain = gain
                    split_info = {
                            'feature_idx': feature_idx,
                            'threshold': threshold,
                            'left_mask': left_mask,
                            'right_mask': right_mask
                            }

        return split_info

    def _calculate_leaf_value(self, y):
        raise NotImplementedError("Subclasses must implement leaf calculation")

    def _calculate_information_gain(self, y, y_left, y_right):
        raise NotImplementedError("Subclasses must implement information gain.")


class DecisionTreeRegressor(BaseDecisionTree):

    def _calculate_leaf_value(self, y):
        """"""
        return np.mean(y)

    def _calculate_information_gain(self, y, y_left, y_right):
        parent_loss = np.sum((y - np.mean(y)) ** 2)
        child_loss = (
            np.sum((y_left - np.mean(y_left)) ** 2) 
            + 
            np.sum((y_right - np.mean(y_right)) ** 2)
        )
        return parent_loss - child_loss  # Positive gain = improvement


class DecisionTreeClassifier(BaseDecisionTree):

    def _calculate_leaf_value(self, y) -> float:
        labels, counts = np.unique(y, return_counts=True)
        index = np.argmax(counts)
        return labels[index]

    def _gini(self, y) -> float:
        if len(y) == 0: return 0.0
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1.0 - np.sum(probs ** 2)

    def _calculate_information_gain(self, y, y_left, y_right) -> float:
        parent_impurity = self._gini(y)

        # Weighted average of child impurities
        w_left = len(y_left) / len(y)
        w_right = len(y_right) / len(y)

        child_impurity = (w_left * self._gini(y_left)) + (w_right * self._gini(y_right))
        return parent_impurity - child_impurity # Positive gain = improvement

