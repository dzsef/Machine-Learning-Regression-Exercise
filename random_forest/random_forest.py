import numpy as np
from regression_tree.regression_tree import MyDecisionTreeRegressor


class MyRandomForestRegressor:
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth:int = 10,
        min_samples: int = 1,
        max_features=None,
        random_state: int | None = None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.max_features = max_features
        self.random_state = random_state

        self.trees: list[MyDecisionTreeRegressor] = []
        self.feature_indices_per_tree: list[np.ndarray] = []
        self.is_fitted = False

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        n_samples, n_features = X.shape
        rng = np.random.RandomState(self.random_state)

        self.trees = []
        self.feature_indices_per_tree = []

        for _ in range(self.n_estimators):
            # always bootstrap
            row_indices = rng.randint(0, n_samples, size=n_samples)
            X_sample = X[row_indices]
            y_sample = y[row_indices]

            # choose feature subset for this tree
            n_sub_features = self.num_sub_features(n_features)
            feature_indices = rng.choice(
                n_features, size=n_sub_features, replace=False
            )

            X_sample_sub = X_sample[:, feature_indices]

            tree = MyDecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples=self.min_samples,
            )
            tree.fit(X_sample_sub, y_sample)

            self.trees.append(tree)
            self.feature_indices_per_tree.append(feature_indices)

        self.is_fitted = True

    def predict(self, X):
        X = np.asarray(X)
        n_samples = X.shape[0]
        n_trees = len(self.trees)

        all_predictions = np.zeros((n_samples, n_trees), dtype=float)

        for tree_index, (tree, feature_indices) in enumerate(
            zip(self.trees, self.feature_indices_per_tree)
        ):
            X_sub = X[:, feature_indices]
            all_predictions[:, tree_index] = tree.predict(X_sub)

        # avg prediction of all trees
        return all_predictions.mean(axis=1)

    def num_sub_features(self, n_features: int) -> int:
        if self.max_features is None:
            return n_features

        if isinstance(self.max_features, str):
            if self.max_features == "sqrt":
                return max(1, int(np.sqrt(n_features)))
            if self.max_features == "log2":
                return max(1, int(np.log2(n_features)))
            return n_features

        value = self.max_features

        if 0 < value <= 1:
            return max(1, int(value * n_features))

