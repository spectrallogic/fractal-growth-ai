import numpy as np
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class FractalNode2D:
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    depth: int
    idxs: np.ndarray
    children: Optional[List["FractalNode2D"]] = None
    pred_class: Optional[int] = None
    class_counts: Optional[np.ndarray] = None
    impurity: float = 0.0

    def is_leaf(self) -> bool:
        return self.children is None


class FractalGrowthClassifier2D:
    """
    Experimental non-neural classifier.

    The model learns by growing a fractal-like decision structure.
    It recursively splits uncertain regions of 2D space instead of using
    neural-network weights or backpropagation.
    """

    def __init__(self, max_depth=7, min_points=10, impurity_threshold=0.03):
        self.max_depth = max_depth
        self.min_points = min_points
        self.impurity_threshold = impurity_threshold

        self.root = None
        self.X = None
        self.y = None
        self.n_classes = None
        self.history = []

    def _gini(self, y_subset):
        if len(y_subset) == 0:
            return 0.0

        counts = np.bincount(y_subset, minlength=self.n_classes)
        probs = counts / counts.sum()
        return 1.0 - np.sum(probs ** 2)

    def _make_node(self, xmin, xmax, ymin, ymax, depth, idxs):
        y_subset = self.y[idxs]

        if len(y_subset) > 0:
            counts = np.bincount(y_subset, minlength=self.n_classes)
            pred_class = int(np.argmax(counts))
            impurity = self._gini(y_subset)
        else:
            counts = np.zeros(self.n_classes, dtype=int)
            pred_class = 0
            impurity = 0.0

        return FractalNode2D(
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
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

    def _split_node(self, node):
        xm = (node.xmin + node.xmax) / 2
        ym = (node.ymin + node.ymax) / 2

        regions = [
            (node.xmin, xm, node.ymin, ym),
            (xm, node.xmax, node.ymin, ym),
            (node.xmin, xm, ym, node.ymax),
            (xm, node.xmax, ym, node.ymax),
        ]

        Xsub = self.X[node.idxs]
        children = []

        for xmin, xmax, ymin, ymax in regions:
            mask = (
                (Xsub[:, 0] >= xmin)
                & (Xsub[:, 0] <= xmax)
                & (Xsub[:, 1] >= ymin)
                & (Xsub[:, 1] <= ymax)
            )

            child_idxs = node.idxs[mask]
            child = self._make_node(
                xmin=xmin,
                xmax=xmax,
                ymin=ymin,
                ymax=ymax,
                depth=node.depth + 1,
                idxs=child_idxs,
            )
            children.append(child)

        node.children = children

    def fit(self, X, y):
        self.X = np.asarray(X)
        self.y = np.asarray(y).astype(int)
        self.n_classes = len(np.unique(self.y))
        self.history = []

        pad = 0.15
        xmin = self.X[:, 0].min() - pad
        xmax = self.X[:, 0].max() + pad
        ymin = self.X[:, 1].min() - pad
        ymax = self.X[:, 1].max() + pad

        self.root = self._make_node(
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
            depth=0,
            idxs=np.arange(len(self.X)),
        )

        for step in range(1, 10_000):
            leaves = self._leaf_nodes(self.root)

            eligible = [
                leaf
                for leaf in leaves
                if leaf.depth < self.max_depth
                and len(leaf.idxs) >= self.min_points
                and leaf.impurity > self.impurity_threshold
            ]

            if not eligible:
                break

            target = max(
                eligible,
                key=lambda leaf: (leaf.impurity, len(leaf.idxs), -leaf.depth),
            )

            self._split_node(target)

            self.history.append(
                {
                    "step": step,
                    "num_leaves": len(self._leaf_nodes(self.root)),
                    "max_impurity": max(leaf.impurity for leaf in self._leaf_nodes(self.root)),
                }
            )

        return self

    def _predict_one(self, x, node):
        if node.is_leaf():
            return node.pred_class

        for child in node.children:
            if (
                child.xmin <= x[0] <= child.xmax
                and child.ymin <= x[1] <= child.ymax
            ):
                return self._predict_one(x, child)

        return node.pred_class

    def predict(self, X):
        X = np.asarray(X)
        return np.array([self._predict_one(x, self.root) for x in X])