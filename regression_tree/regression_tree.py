import numpy as np
from dataclasses import dataclass
from typing import Optional
import copy

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


@dataclass
class DecisionTreeNode:
    feature_index: Optional[int] = None
    threshold: Optional[float] = None
    left_child: Optional["DecisionTreeNode"] = None
    right_child: Optional["DecisionTreeNode"] = None
    value: Optional[float] = None  

    # stats for post pruning
    n_samples: int = 0
    sum_y: float = 0.0
    sum_y_squared: float = 0.0
    subtree_rss: float = 0.0
    subtree_leaves: int = 1

    def is_leaf(self) -> bool:
        return self.left_child is None and self.right_child is None


class MyDecisionTreeRegressor:
    def __init__(self, max_depth: int = 10, min_samples: int = 1):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.root: Optional[DecisionTreeNode] = None

        # only when post pruning
        self.best_alpha: Optional[float] = None
        self.alpha_candidates: Optional[list[float]] = None
        self.alpha_cv_errors: Optional[list[float]] = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.root = self.build_tree(X, y, depth=0)

    def predict(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return np.array([self.predict_row(row, self.root) for row in X])

    def predict_row(self, row, node: DecisionTreeNode) -> float:
        current = node
        while not current.is_leaf():
            if row[current.feature_index] <= current.threshold:
                current = current.left_child
            else:
                current = current.right_child
        return current.value

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
                subtree_rss=0.0,
                subtree_leaves=1,
            )

        best_feature_index = None
        best_threshold = None
        best_loss = np.inf
        best_left_mask = None
        best_right_mask = None

        # search best split over all features using RSS
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
                subtree_rss=0.0,
                subtree_leaves=1,
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
            subtree_rss=0.0,
            subtree_leaves=0,
        )

    # post pruning helpers

    def leaf_rss(self, node: DecisionTreeNode) -> float:
        if node.n_samples == 0:
            return 0.0
        mean = node.sum_y / node.n_samples
        # RSS = sum(y^2) - n * mean^2
        return node.sum_y_squared - node.n_samples * (mean ** 2)

    def compute_subtree_info(self, node: DecisionTreeNode):
        if node.is_leaf():
            node.subtree_rss = self.leaf_rss(node)
            node.subtree_leaves = 1
            return

        self.compute_subtree_info(node.left_child)
        self.compute_subtree_info(node.right_child)

        node.subtree_rss = (
            node.left_child.subtree_rss + node.right_child.subtree_rss
        )
        node.subtree_leaves = (
            node.left_child.subtree_leaves + node.right_child.subtree_leaves
        )

    def collect_alpha_values(
        self,
        node: DecisionTreeNode,
        values: list[tuple[DecisionTreeNode, float]],
    ):
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

    def cost_complexity_pruning_path(self) -> tuple[list[float], list[DecisionTreeNode]]:
        if self.root is None:
            raise ValueError("Man please call fit before post pruning")

        current_root = copy.deepcopy(self.root)

        ccp_alphas: list[float] = [0.0]
        subtrees: list[DecisionTreeNode] = [copy.deepcopy(current_root)]

        tolerance = 0.000000000001

        while not current_root.is_leaf():
            self.compute_subtree_info(current_root)

            alpha_list: list[tuple[DecisionTreeNode, float]] = []
            self.collect_alpha_values(current_root, alpha_list)

            if not alpha_list:
                break

            min_alpha = min(a for (_, a) in alpha_list)

            nodes_to_prune = [
                node for (node, a) in alpha_list if abs(a - min_alpha) <= tolerance
            ]

            for node in nodes_to_prune:
                if node.n_samples > 0:
                    node.value = node.sum_y / node.n_samples
                else:
                    node.value = 0.0
                node.left_child = None
                node.right_child = None

            ccp_alphas.append(float(min_alpha))
            subtrees.append(copy.deepcopy(current_root))

            if current_root.is_leaf():
                break

        return ccp_alphas, subtrees

    def predict_with_root(self, X, root: DecisionTreeNode):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return np.array([self.predict_row(row, root) for row in X])

    def select_subtree_for_alpha(
        self,
        alphas: list[float],
        subtrees: list[DecisionTreeNode],
        alpha_candidate: float,
    ) -> DecisionTreeNode:
        
        best_index = 0
        for index, alpha_value in enumerate(alphas):
            if alpha_value <= alpha_candidate:
                best_index = index
            else:
                break
        return subtrees[best_index]


    def post_prune_with_cross_validation(
        self,
        X,
        y,
        n_splits: int = 10,
        random_state: int = 42,
    ) -> float:
        if self.root is None:
            raise ValueError("Man please call fit before post pruning")

        # global pruning path from the already fitted T0
        global_alphas, global_subtrees = self.cost_complexity_pruning_path()
        candidate_alphas = list(global_alphas)

        # prepare folds and for each fold compute its own pruning path
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        fold_results: list[tuple[list[float], list[DecisionTreeNode], np.ndarray, np.ndarray]] = []

        for train_index, val_index in kfold.split(X):
            X_train_fold = X[train_index]
            y_train_fold = y[train_index]
            X_val_fold = X[val_index]
            y_val_fold = y[val_index]

            fold_tree = MyDecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples=self.min_samples,
            )
            fold_tree.fit(X_train_fold, y_train_fold)
            alphas_fold, subtrees_fold = fold_tree.cost_complexity_pruning_path()

            fold_results.append(
                (alphas_fold, subtrees_fold, X_val_fold, y_val_fold)
            )

        # for each candidate alpha compute mean cv error over folds
        mean_mse_for_alpha: list[float] = []

        for alpha_candidate in candidate_alphas:
            fold_mse_values: list[float] = []

            for alphas_fold, subtrees_fold, X_val_fold, y_val_fold in fold_results:
                subtree_root = self.select_subtree_for_alpha(
                    alphas_fold,
                    subtrees_fold,
                    alpha_candidate,
                )
                y_pred_val = self.predict_with_root(X_val_fold, subtree_root)
                mse = mean_squared_error(y_val_fold, y_pred_val)
                fold_mse_values.append(mse)

            mean_mse_for_alpha.append(float(np.mean(fold_mse_values)))

        # choose the alpha with the smallest mean cv error
        best_index = int(np.argmin(mean_mse_for_alpha))
        best_alpha = candidate_alphas[best_index]

        # select the corresponding subtree 
        final_root = self.select_subtree_for_alpha(
            global_alphas,
            global_subtrees,
            best_alpha,
        )

        self.root = final_root
        self.best_alpha = best_alpha
        self.alpha_candidates = candidate_alphas
        self.alpha_cv_errors = mean_mse_for_alpha

        return best_alpha
