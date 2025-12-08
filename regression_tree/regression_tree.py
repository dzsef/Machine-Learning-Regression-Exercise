import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple


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

                left_mean = left_targets.mean()
                right_mean = right_targets.mean()

                left_rss = np.sum((left_targets - left_mean) ** 2)
                right_rss = np.sum((right_targets - right_mean) ** 2)

                loss = left_rss + right_rss

                if loss < best_loss:
                    best_loss = loss
                    best_feature_index = feature_index
                    best_threshold = threshold
                    best_left_mask = left_mask
                    best_right_mask = right_mask

        # if no valid split was found, make a leaf
        if best_feature_index is None:
            leaf_value = sum_y / n_samples
            return DecisionTreeNode(
                value=leaf_value,
                n_samples=n_samples,
                sum_y=sum_y,
                sum_y_squared=sum_y_squared,
            )

        left_child = self.build_tree(X[best_left_mask], y[best_left_mask], depth + 1)
        right_child = self.build_tree(X[best_right_mask], y[best_right_mask], depth + 1)

        return DecisionTreeNode(
            feature_index=best_feature_index,
            threshold=best_threshold,
            left_child=left_child,
            right_child=right_child,
            value=None,
            n_samples=n_samples,
            sum_y=sum_y,
            sum_y_squared=sum_y_squared,
        )

    def predict_row(self, row, node: DecisionTreeNode) -> float:
        current = node
        while not current.is_leaf():
            if row[current.feature_index] <= current.threshold:
                current = current.left_child
            else:
                current = current.right_child
        return current.value

    # Cost complexity pruning

    def leaf_rss(self, node: DecisionTreeNode) -> float:
        if node.n_samples == 0:
            return 0.0
        mean = node.sum_y / node.n_samples
        return node.sum_y_squared - node.n_samples * (mean ** 2)

    def compute_subtree_info(self, node: DecisionTreeNode):
        if node.is_leaf():
            node.subtree_rss = self.leaf_rss(node)
            node.subtree_leaves = 1
            return
        self.compute_subtree_info(node.left_child)
        self.compute_subtree_info(node.right_child)
        node.subtree_rss = node.left_child.subtree_rss + node.right_child.subtree_rss
        node.subtree_leaves = node.left_child.subtree_leaves + node.right_child.subtree_leaves

    def collect_alpha_values(self, node: DecisionTreeNode, values: List[Tuple[DecisionTreeNode, float]]):
        if node.is_leaf():
            return
        if node.subtree_leaves > 1:
            rss_if_leaf = self.leaf_rss(node)
            alpha = (rss_if_leaf - node.subtree_rss) / (node.subtree_leaves - 1)
            values.append((node, alpha))
        if node.left_child is not None:
            self.collect_alpha_values(node.left_child, values)
        if node.right_child is not None:
            self.collect_alpha_values(node.right_child, values)

    def prune_with_cost_complexity(self, alpha: float):
        if self.root is None or self.root.is_leaf():
            return

        tolerance = 0.000000000001

        while True:
            self.compute_subtree_info(self.root)

            alpha_list: List[Tuple[DecisionTreeNode, float]] = []
            self.collect_alpha_values(self.root, alpha_list)

            if not alpha_list:
                break

            min_alpha = min(a for (_, a) in alpha_list)

            if min_alpha > alpha + tolerance:
                break

            nodes_to_prune = [node for (node, a) in alpha_list if abs(a - min_alpha) <= tolerance]

            for node in nodes_to_prune:
                if node.n_samples > 0:
                    node.value = node.sum_y / node.n_samples
                else:
                    node.value = 0.0
                node.left_child = None
                node.right_child = None

            if self.root.is_leaf():
                break
