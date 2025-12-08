import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class DecisionTreeNode:
    feature_index: Optional[int] = None
    threshold: Optional[float] = None
    left_child: Optional["DecisionTreeNode"] = None
    right_child: Optional["DecisionTreeNode"] = None
    value: Optional[float] = None 
    n_samples: int = 0
    sum_y: float = 0.0
    sum_y_squared: float = 0.0
    subtree_rss: float = 0.0
    subtree_leaves: int = 1

    def is_leaf(self) -> bool:
        return self.left_child is None and self.right_child is None


class MyDecisionTreeRegressor:
    def __init__(
        self,
        max_depth: int = 5,
        min_samples: int = 1,
        ccp_alpha: float = 0.0,
    ):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.ccp_alpha = ccp_alpha
        self.root: Optional[DecisionTreeNode] = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        self.root = self.build_tree(X, y, depth=0)

        if self.ccp_alpha is not None and self.ccp_alpha > 0.0 and self.root is not None:
            self.prune_with_cost_complexity(self.ccp_alpha)

    def predict(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return np.array([self.predict_row(row, self.root) for row in X])

    def build_tree(self, X, y, depth: int) -> DecisionTreeNode:
        num_samples, num_features = X.shape
        n_samples = int(num_samples)
        sum_y = float(y.sum())
        sum_y_squared = float((y ** 2).sum())

        # pre pruning
        if (
            depth >= self.max_depth
            or num_samples < self.min_samples
            or np.all(y == y[0])
        ):
            leaf_value = sum_y / n_samples
            return DecisionTreeNode(
                value=leaf_value,
                n_samples=n_samples,
                sum_y=sum_y,
                sum_y_squared=sum_y_squared,
            )

        best_feature_index = None
        best_threshold = None
        best_loss = np.inf
        best_left_mask = None
        best_right_mask = None

        # search best split using RSS
        for feature_index in range(num_features):
            feature_values = X[:, feature_index]

            sorted_indices = np.argsort(feature_values)
            sorted_feature_values = feature_values[sorted_indices]

            # candidate thresholds are the  midpoints between distinct consecutive values
            for i in range(1, num_samples):
                if sorted_feature_values[i] == sorted_feature_values[i - 1]:
                    continue

                threshold = (sorted_feature_values[i] + sorted_feature_values[i - 1]) / 2.0

                left_mask = feature_values <= threshold
                right_mask = ~left_mask

                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue

                left_targets = y[left_mask]
                right_targets = y[right_mask]

                left_mse = np.mean((left_targets - left_targets.mean()) ** 2)
                right_mse = np.mean((right_targets - right_targets.mean()) ** 2)

                weighted_loss = (
                    left_targets.size * left_mse + right_targets.size * right_mse
                ) / num_samples

                if weighted_loss < best_loss:
                    best_loss = weighted_loss
                    best_feature_index = feature_index
                    best_threshold = threshold
                    best_left_mask = left_mask
                    best_right_mask = right_mask

        # if no valid split was found, make a leaf
        if best_feature_index is None:
            leaf_value = float(y.mean())
            return DecisionTreeNode(value=leaf_value)

        left_child = self.build_tree(X[best_left_mask], y[best_left_mask], depth + 1)
        right_child = self.build_tree(X[best_right_mask], y[best_right_mask], depth + 1)

        return DecisionTreeNode(
            feature_index=best_feature_index,
            threshold=best_threshold,
            left_child=left_child,
            right_child=right_child,
        )

    def predict_row(self, row, node: DecisionTreeNode) -> float:
        if node.is_leaf():
            return node.value

        if row[node.feature_index] <= node.threshold:
            return self.predict_row(row, node.left_child)
        else:
            return self.predict_row(row, node.right_child)
