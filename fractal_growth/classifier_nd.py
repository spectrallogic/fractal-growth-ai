import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple


@dataclass
class FractalNodeND:
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    depth: int
    idxs: np.ndarray
    split_dim: Optional[int] = None
    split_value: Optional[float] = None
    children: Optional[List["FractalNodeND"]] = None
    pred_class: Optional[int] = None
    class_counts: Optional[np.ndarray] = None
    impurity: float = 0.0
    gain: float = 0.0

    def is_leaf(self) -> bool:
        return self.children is None


class FractalGrowthClassifierND:
    """
    Experimental non-neural classifier for higher-dimensional data.

    Version 4:
    - information-gain splitting
    - smarter candidate thresholds near class-boundary changes
    - global feature preselection
    - optional feature subset size
    - no destructive pruning by default

    Goal:
    Get faster without sacrificing as much accuracy.
    """

    def __init__(
        self,
        max_depth=14,
        min_points=6,
        impurity_threshold=0.005,
        n_candidate_splits=16,
        max_features="auto",
        min_gain=1e-5,
        random_state=42,
    ):
        self.max_depth = max_depth
        self.min_points = min_points
        self.impurity_threshold = impurity_threshold
        self.n_candidate_splits = n_candidate_splits
        self.max_features = max_features
        self.min_gain = min_gain
        self.random_state = random_state

        self.root = None
        self.X = None
        self.y = None
        self.n_classes = None
        self.history = []
        self.feature_order_ = None
        self.rng_ = np.random.RandomState(random_state)

    def _gini(self, y_subset):
        if len(y_subset) == 0:
            return 0.0

        counts = np.bincount(y_subset, minlength=self.n_classes)
        probs = counts / counts.sum()
        return 1.0 - np.sum(probs ** 2)

    def _make_node(self, bounds_min, bounds_max, depth, idxs):
        y_subset = self.y[idxs]

        if len(y_subset) > 0:
            counts = np.bincount(y_subset, minlength=self.n_classes)
            pred_class = int(np.argmax(counts))
            impurity = self._gini(y_subset)
        else:
            counts = np.zeros(self.n_classes, dtype=int)
            pred_class = 0
            impurity = 0.0

        return FractalNodeND(
            bounds_min=bounds_min.copy(),
            bounds_max=bounds_max.copy(),
            depth=depth,
            idxs=idxs,
            pred_class=pred_class,
            class_counts=counts,
            impurity=impurity,
        )

    def _leaf_nodes(self, node):
        if node.is_leaf():
            return [node]

        leaves = []
        for child in node.children:
            leaves.extend(self._leaf_nodes(child))
        return leaves

    def _information_gain(self, parent_y, left_y, right_y):
        if len(left_y) == 0 or len(right_y) == 0:
            return 0.0

        parent_impurity = self._gini(parent_y)
        left_impurity = self._gini(left_y)
        right_impurity = self._gini(right_y)

        n = len(parent_y)

        weighted_child_impurity = (
            (len(left_y) / n) * left_impurity
            + (len(right_y) / n) * right_impurity
        )

        return parent_impurity - weighted_child_impurity

    def _compute_global_feature_order(self):
        """
        Cheap ANOVA-style feature ranking.

        This gives the fractal model a better search order so it does not
        waste time testing useless dimensions first.
        """
        X = self.X
        y = self.y
        n_features = X.shape[1]

        scores = np.zeros(n_features, dtype=float)

        global_mean = X.mean(axis=0)

        for c in range(self.n_classes):
            mask = y == c
            if not np.any(mask):
                continue

            Xc = X[mask]
            class_mean = Xc.mean(axis=0)
            scores += len(Xc) * (class_mean - global_mean) ** 2

        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

        return np.argsort(scores)[::-1]

    def _get_feature_subset(self, n_features):
        if self.max_features is None or self.max_features == "all":
            k = n_features
        elif self.max_features == "sqrt":
            k = max(1, int(np.sqrt(n_features)))
        elif self.max_features == "log2":
            k = max(1, int(np.log2(n_features)))
        elif self.max_features == "auto":
            # More generous than sqrt.
            # This is the compromise between v2 accuracy and v3 speed.
            k = min(n_features, max(8, int(2.5 * np.sqrt(n_features))))
        elif isinstance(self.max_features, int):
            k = min(n_features, max(1, self.max_features))
        elif isinstance(self.max_features, float):
            k = min(n_features, max(1, int(n_features * self.max_features)))
        else:
            k = n_features

        return self.feature_order_[:k]

    def _candidate_split_values(self, values, labels):
        """
        Smart candidate thresholds.

        Instead of testing random percentiles, test midpoints where sorted
        labels change. Those are the places where a split can actually
        separate classes.

        This is usually both faster and more accurate.
        """
        order = np.argsort(values)
        sorted_values = values[order]
        sorted_labels = labels[order]

        different_label = sorted_labels[:-1] != sorted_labels[1:]
        different_value = sorted_values[:-1] != sorted_values[1:]

        boundary_positions = np.where(different_label & different_value)[0]

        if len(boundary_positions) == 0:
            return []

        candidates = (
            sorted_values[boundary_positions]
            + sorted_values[boundary_positions + 1]
        ) / 2.0

        candidates = np.unique(candidates)

        if len(candidates) <= self.n_candidate_splits:
            return candidates

        # Evenly sample from useful class-boundary candidates.
        pick = np.linspace(
            0,
            len(candidates) - 1,
            self.n_candidate_splits,
        ).astype(int)

        return candidates[pick]

    def _best_split(self, idxs) -> Tuple[Optional[int], Optional[float], float]:
        Xsub = self.X[idxs]
        ysub = self.y[idxs]

        best_dim = None
        best_value = None
        best_gain = 0.0

        n_features = Xsub.shape[1]
        feature_subset = self._get_feature_subset(n_features)

        for dim in feature_subset:
            values = Xsub[:, dim]
            candidates = self._candidate_split_values(values, ysub)

            for split_value in candidates:
                left_mask = values <= split_value
                right_mask = ~left_mask

                left_y = ysub[left_mask]
                right_y = ysub[right_mask]

                gain = self._information_gain(
                    parent_y=ysub,
                    left_y=left_y,
                    right_y=right_y,
                )

                if gain > best_gain:
                    best_gain = gain
                    best_dim = dim
                    best_value = float(split_value)

        return best_dim, best_value, best_gain

    def _split_node(self, node):
        dim, split_value, gain = self._best_split(node.idxs)

        if dim is None:
            return False

        if gain < self.min_gain:
            return False

        Xsub = self.X[node.idxs]
        mask_left = Xsub[:, dim] <= split_value

        left_idxs = node.idxs[mask_left]
        right_idxs = node.idxs[~mask_left]

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return False

        left_min = node.bounds_min.copy()
        left_max = node.bounds_max.copy()
        left_max[dim] = split_value

        right_min = node.bounds_min.copy()
        right_max = node.bounds_max.copy()
        right_min[dim] = split_value

        left_child = self._make_node(
            bounds_min=left_min,
            bounds_max=left_max,
            depth=node.depth + 1,
            idxs=left_idxs,
        )

        right_child = self._make_node(
            bounds_min=right_min,
            bounds_max=right_max,
            depth=node.depth + 1,
            idxs=right_idxs,
        )

        node.split_dim = dim
        node.split_value = split_value
        node.children = [left_child, right_child]
        node.gain = gain

        return True

    def fit(self, X, y):
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y).astype(int)

        self.rng_ = np.random.RandomState(self.random_state)
        self.n_classes = len(np.unique(self.y))
        self.history = []

        self.feature_order_ = self._compute_global_feature_order()

        bounds_min = self.X.min(axis=0) - 1e-9
        bounds_max = self.X.max(axis=0) + 1e-9

        self.root = self._make_node(
            bounds_min=bounds_min,
            bounds_max=bounds_max,
            depth=0,
            idxs=np.arange(len(self.X)),
        )

        failed_nodes = set()

        for step in range(1, 100_000):
            leaves = self._leaf_nodes(self.root)

            eligible = [
                leaf
                for leaf in leaves
                if id(leaf) not in failed_nodes
                and leaf.depth < self.max_depth
                and len(leaf.idxs) >= self.min_points
                and leaf.impurity > self.impurity_threshold
            ]

            if not eligible:
                break

            target = max(
                eligible,
                key=lambda leaf: (leaf.impurity, len(leaf.idxs), -leaf.depth),
            )

            did_split = self._split_node(target)

            if not did_split:
                failed_nodes.add(id(target))
                continue

            current_leaves = self._leaf_nodes(self.root)

            self.history.append(
                {
                    "step": step,
                    "num_leaves": len(current_leaves),
                    "avg_impurity": float(
                        np.mean([leaf.impurity for leaf in current_leaves])
                    ),
                    "max_impurity": float(
                        np.max([leaf.impurity for leaf in current_leaves])
                    ),
                }
            )

        return self

    def _predict_one_from_node(self, x, node):
        if node.is_leaf():
            return node.pred_class

        if x[node.split_dim] <= node.split_value:
            return self._predict_one_from_node(x, node.children[0])

        return self._predict_one_from_node(x, node.children[1])

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_one_from_node(x, self.root) for x in X])

    def get_num_regions(self):
        if self.root is None:
            return 0

        return len(self._leaf_nodes(self.root))