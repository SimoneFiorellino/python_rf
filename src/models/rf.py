import math
import random
from dataclasses import dataclass
from typing import Optional, List

import numpy as np


@dataclass  # Simple tree node structure
class _TreeNode:
    gini: float  # Gini impurity of the node
    num_samples: int  # How many samples reached this node
    num_samples_per_class: np.ndarray  # The class distribution at this node
    predicted_class: int  # The class predicted at this node
    feature_index: Optional[int] = None  # Split feature index
    threshold: Optional[float] = None  # Split threshold
    left: Optional["_TreeNode"] = None  # Left child
    right: Optional["_TreeNode"] = None  # Right child

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class DecisionTreeClassifierNumpy:
    """
    Simple decision tree for classification using numpy arrays.
    - Supports only numeric features
    - Uses Gini impurity

    :param max_depth: The maximum depth of each decision tree in the forest. If None, nodes are expanded until all leaves are pure.
    :param min_samples_split: The minimum number of samples required to split an internal node.
    :param max_features: The number of features to consider when looking for the best split:
        - If "sqrt", then max_features=sqrt(n_features).
        - If "log2", then max_features=log2(n_features).
        - If None, then max_features=n_features. Use all features (like a standard decision tree)
        - If int, then consider max_features features at each split.
        - If float, then max_features is a fraction and int(max_features * n_features) features are considered at each split.
    :param random_state: Seed for the random number generator. (e.g., for feature subsampling and bootstrap)
    """

    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        max_features: Optional[str | int | float] = "sqrt",
        random_state: Optional[int] = None,
    ):
        # User parameters
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state

        # Public attributes
        self.n_classes_: Optional[int] = None
        self.n_features_: Optional[int] = None
        self.tree_: Optional[_TreeNode] = None

        # Internal attributes
        self._rng = random.Random(random_state)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        X: (n_samples, n_features) float array
        y: (n_samples,) array of class indices (0..n_classes-1)
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)

        _, n_features = X.shape  # (n_samples, n_features)
        self.n_features_ = n_features
        self.n_classes_ = int(np.max(y) + 1)

        self.tree_ = self._build_tree(X, y, depth=0)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        preds = [self._predict_single(x) for x in X]
        return np.array(preds, dtype=int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        all_proba = [self._predict_single_proba(x) for x in X]
        return np.stack(all_proba, axis=0)

    # --------- tree building --------- #

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> _TreeNode:
        num_samples = y.shape[0]
        num_samples_per_class = np.bincount(y, minlength=self.n_classes_).astype(float)
        gini = self._gini(y)
        predicted_class = int(np.argmax(num_samples_per_class))

        node = _TreeNode(
            gini=gini,
            num_samples=num_samples,
            num_samples_per_class=num_samples_per_class,
            predicted_class=predicted_class,
        )

        # Check stopping criteria
        if (
            (self.max_depth is not None and depth >= self.max_depth)
            or num_samples < self.min_samples_split
            or gini == 0.0
        ):
            return node

        # Find best split
        feature_idx, threshold = self._best_split(X, y)
        if feature_idx is None:
            return node

        # Split data
        left_mask = X[:, feature_idx] <= threshold
        right_mask = X[:, feature_idx] > threshold

        if not left_mask.any() or not right_mask.any():
            return node

        node.feature_index = feature_idx
        node.threshold = float(threshold)
        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        return node

    def _gini(self, y: np.ndarray) -> float:
        m = y.shape[0]
        if m == 0:
            return 0.0
        counts = np.bincount(y, minlength=self.n_classes_).astype(float)
        p = counts / m
        return float(1.0 - np.sum(p**2))

    def _num_features_to_consider(self) -> int:
        if self.max_features is None:
            return self.n_features_

        if isinstance(self.max_features, str):
            if self.max_features == "sqrt":
                # square root of n_features
                return max(1, int(math.sqrt(self.n_features_)))
            if self.max_features == "log2":
                # base-2 logarithm of n_features
                return max(1, int(math.log2(self.n_features_)))
            raise ValueError(f"Unsupported max_features string: {self.max_features}")

        if isinstance(self.max_features, (int, float)):
            if isinstance(self.max_features, float) and 0 < self.max_features <= 1:
                # fraction of n_features
                return max(1, int(self.max_features * self.n_features_))
            return max(1, int(self.max_features))

        raise ValueError(f"Unsupported max_features type: {type(self.max_features)}")

    def _best_split(self, X: np.ndarray, y: np.ndarray):
        m, n = X.shape
        if m < self.min_samples_split:
            return None, None

        n_features_to_try = self._num_features_to_consider()
        feature_indices = list(range(n))
        self._rng.shuffle(feature_indices)
        feature_indices = feature_indices[:n_features_to_try]

        best_gini = 1.0
        best_idx = None
        best_thr = None

        for feature in feature_indices:
            Xf = X[:, feature]
            # Sort by feature values
            sorted_idx = np.argsort(Xf)
            Xf_sorted = Xf[sorted_idx]
            y_sorted = y[sorted_idx]

            # Counts for right side (initially all samples)
            num_right = np.bincount(y_sorted, minlength=self.n_classes_).astype(float)
            num_left = np.zeros(self.n_classes_, dtype=float)

            for i in range(m - 1):
                c = int(y_sorted[i])
                num_left[c] += 1.0
                num_right[c] -= 1.0

                if Xf_sorted[i] == Xf_sorted[i + 1]:
                    continue  # skip equal values, no real split between them

                left_count = i + 1
                right_count = m - left_count

                gini_left = 1.0 - np.sum((num_left / left_count) ** 2)
                gini_right = 1.0 - np.sum((num_right / right_count) ** 2)
                gini = (left_count * gini_left + right_count * gini_right) / m

                gini_val = float(gini)
                if gini_val < best_gini:
                    best_gini = gini_val
                    best_idx = feature
                    best_thr = 0.5 * (Xf_sorted[i] + Xf_sorted[i + 1])

        return best_idx, best_thr

    # --------- prediction helpers --------- #

    def _predict_single(self, x: np.ndarray) -> int:
        node = self.tree_
        while not node.is_leaf:
            if x[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predicted_class

    def _predict_single_proba(self, x: np.ndarray) -> np.ndarray:
        node = self.tree_
        while not node.is_leaf:
            if x[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        counts = node.num_samples_per_class
        return counts / counts.sum()


class RandomForestClassifierNumpy:
    """
    Simple Random Forest classifier implemented with NumPy arrays.
    - Ensemble of DecisionTreeClassifierNumpy
    - Bagging (bootstrap samples)
    - Feature subsampling at each split via tree.max_features
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        max_features: Optional[str | int | float] = "sqrt",
        bootstrap: bool = True,
        random_state: Optional[int] = None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.trees: List[DecisionTreeClassifierNumpy] = []
        # Public attributes
        self.n_classes_: Optional[int] = None
        # Internal random generator
        self._rng = np.random.default_rng(random_state)

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)

        n_samples, _ = X.shape
        self.n_classes_ = int(np.max(y) + 1)

        self.trees = []
        for i in range(self.n_estimators):
            # bootstrap sampling
            if self.bootstrap:
                indices = self._rng.integers(0, n_samples, size=n_samples)
                sample_X = X[indices]
                sample_y = y[indices]
            else:
                sample_X, sample_y = X, y

            tree = DecisionTreeClassifierNumpy(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                random_state=None
                if self.random_state is None
                else self.random_state + i,
            )
            tree.fit(sample_X, sample_y)
            self.trees.append(tree)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)

        # (n_trees, n_samples, n_classes)
        all_proba = np.stack([tree.predict_proba(X) for tree in self.trees], axis=0)
        # Average across trees -> (n_samples, n_classes)
        return all_proba.mean(axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)


if (
    __name__ == "__main__"
):  # Simple test comparing with sklearn's RandomForestClassifier
    from sklearn.datasets import load_breast_cancer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    from src.utils.custom_context_menager import Timer

    # ------------------------
    # 1. Load a harder real dataset
    # ------------------------
    data = load_breast_cancer()
    X_np = data.data  # shape (n_samples, n_features)
    y_np = data.target  # binary classification (0/1)

    # train/test split (harder than training on all data)
    X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
        X_np,
        y_np,
        test_size=0.3,  # 30% test
        random_state=42,
        stratify=y_np,
    )

    # ------------------------
    # 2. NumPy RandomForest
    # ------------------------
    model_np = RandomForestClassifierNumpy(
        n_estimators=10,
        max_depth=1,
        max_features="sqrt",
        bootstrap=True,
        random_state=42,
    )
    with Timer():
        model_np.fit(X_train_np, y_train_np)

    preds_np_test = model_np.predict(X_test_np)
    acc_np_test = (preds_np_test == y_test_np).mean()
    print(f"NumPy RF test accuracy: {acc_np_test:.3f}")

    # ------------------------
    # 3. sklearn RandomForest
    # ------------------------
    model_skl = RandomForestClassifier(
        n_estimators=10,
        max_depth=1,
        max_features="sqrt",
        bootstrap=True,
        random_state=42,
    )
    with Timer():
        model_skl.fit(X_train_np, y_train_np)

    preds_skl_test = model_skl.predict(X_test_np)
    acc_skl_test = (preds_skl_test == y_test_np).mean()
    print(f"sklearn RF test accuracy: {acc_skl_test:.3f}")

    # ------------------------
    # 4. Agreement between models on test set
    # ------------------------
    agreement_test = (preds_np_test == preds_skl_test).mean()
    print(f"Prediction agreement on test set: {agreement_test:.3f}")
