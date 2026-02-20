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
    def __init__(self, max_depth: Optional[int] = None, max_features=None):
        self.max_depth = max_depth
        self.max_features = max_features
        self.root = None

    def fit(self, X, y, weights=None):

        if weights is None:
            weights = np.ones(len(y))

        X = np.asarray(X)
        y = np.asarray(y)

        self.root = self._grow_tree(X, y, weights)
        return self

    def predict(self, X):
        # For each vector \vec{x} \in \mathbf{X}
        return np.array([self._traverse(x, self.root) for x in X])

    def _grow_tree(self, X, y, weights, depth=0):
        """For fitting the tree."""
        n_samples, _ = X.shape
        n_labels = len(np.unique(y))

        # STOPPING CONDITION: This is what kills the recursion
        if (
            (self.max_depth is not None and depth >= self.max_depth)
            or n_labels <= 1
            or n_samples < 2
        ):
            return Node(value=self._calculate_leaf_value(y, weights))

        split = self._find_best_split(X, y, weights)

        # If mathematically impossible to split the data more
        # return node (terminal)
        if not split:
            return Node(value=self._calculate_leaf_value(y, weights))

        # RECURSION:
        left_child = self._grow_tree(
            X[split["left_mask"]],
            y[split["left_mask"]],
            weights[split["left_mask"]],
            depth + 1,
        )

        right_child = self._grow_tree(
            X[split["right_mask"]],
            y[split["right_mask"]],
            weights[split["right_mask"]],
            depth + 1,
        )

        return Node(
            feature=split["feature_idx"],
            threshold=split["threshold"],
            left=left_child,
            right=right_child,
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

    def _find_best_split(self, X, y, weights):
        """Greedy search strategy for the best split point."""
        best_gain = -float("inf")
        split_info = None

        # If there are fewer than 2 unique values in X, we can't split
        if X.shape[0] < 2:
            return None

        n_features = X.shape[1]

        if self.max_features is None:  # Standard bagging
            feature_indices = np.arange(n_features)

        else:  # random forest (pick m features at random for this split)
            num_to_sample = min(n_features, self.max_features)
            feature_indices = np.random.choice(
                np.arange(n_features), size=num_to_sample, replace=False
            )

        for feature_idx in feature_indices:
            # Get unique values of the feature
            unique_vals = np.unique(X[:, feature_idx])

            if len(unique_vals) < 2:  # Skip features with no variation
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
                    y[right_mask],
                    weights,
                    weights[left_mask],
                    weights[right_mask],
                )

                if gain > best_gain:
                    best_gain = gain
                    split_info = {
                        "feature_idx": feature_idx,
                        "threshold": threshold,
                        "left_mask": left_mask,
                        "right_mask": right_mask,
                    }

        return split_info

    def _calculate_leaf_value(self, y, weights):
        raise NotImplementedError("Subclasses must implement leaf calculation")

    def _calculate_information_gain(self, y, y_left, y_right, w, w_left, w_right):
        raise NotImplementedError("Subclasses must implement information gain.")


class DecisionTreeRegressor(BaseDecisionTree):
    def _calculate_leaf_value(self, y, weights):
        return np.sum(weights * y) / np.sum(weights)

    def _calculate_information_gain(self, y, y_left, y_right, w, w_left, w_right):
        """Information gain criteria for DecisionTreeRegressor.

        Args:
            y: the entire y (parent) array
            y_left: y dedicated to left child node
            y_right: y dedicated to right child node

            w: weight assigned to parent
            w_left: weight assigned to left child
            y_right: weight assigned to right child

        Returns:
            type: the information gain (positive=improvement)
        """

        parent_mean = np.sum(w * y) / np.sum(w)
        left_mean = np.sum(w_left * y_left) / np.sum(w_left)
        right_mean = np.sum(w_right * y_right) / np.sum(w_right)

        # Weighted RSS
        parent_loss = np.sum(w * (y - parent_mean) ** 2)
        child_loss = np.sum(w_left * (y_left - left_mean) ** 2) + np.sum(
            w_right * (y_right - right_mean) ** 2
        )
        return parent_loss - child_loss


class DecisionTreeClassifier(BaseDecisionTree):
    def _calculate_leaf_value(self, y, weights):
        unique_classes = np.unique(y)
        class_weights = [np.sum(weights[y == c]) for c in unique_classes]
        return unique_classes[np.argmax(class_weights)]

    def _gini(self, y, weights) -> float:
        if len(y) == 0:
            return 0.0

        total_weight = np.sum(weights)
        unique_classes = np.unique(y)

        weighted_probs_sq = 0.0
        for cls in unique_classes:
            # Sum weights of all samples belonging to this class
            cls_weight_sum = np.sum(weights[y == cls])
            p_i = cls_weight_sum / total_weight
            weighted_probs_sq += p_i**2

        return 1.0 - weighted_probs_sq

    def _calculate_information_gain(self, y, y_left, y_right, w, w_left, w_right):
        parent_impurity = self._gini(y, w)

        total_weight = np.sum(w)
        weight_ratio_l = np.sum(w_left) / total_weight
        weight_ratio_r = np.sum(w_right) / total_weight

        child_impurity = (weight_ratio_l * self._gini(y_left, w_left)) + (
            weight_ratio_r * self._gini(y_right, w_right)
        )

        return parent_impurity - child_impurity
