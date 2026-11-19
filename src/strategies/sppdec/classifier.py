# src/strategies/sppdec/classifier.py
"""
A small decision tree, written out so that no machine-learning package is needed at run time.

Used by the lightweight-classifier arbiters of SPPDEC. The trees are trained offline
(analysis/revision/train_classifiers.py) and stored as JSON; a decision costs a handful of comparisons.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

MODELS_PATH = Path(__file__).with_name("classifier_models.json")


class DecisionTree:
    """Binary classification tree with axis-aligned splits and weighted Gini impurity."""

    def __init__(self, max_depth: int = 3, min_leaf_weight: float = 0.0):
        self.max_depth = max_depth
        self.min_leaf_weight = min_leaf_weight
        self.root: Optional[Dict] = None
        self.features: List[str] = []

    # -- training -------------------------------------------------------------------------
    @staticmethod
    def _gini(w0: float, w1: float) -> float:
        t = w0 + w1
        if t <= 0:
            return 0.0
        p = w1 / t
        return 2.0 * p * (1.0 - p)

    def fit(self, X: Sequence[Sequence[float]], y: Sequence[int], features: Sequence[str],
            weights: Optional[Sequence[float]] = None) -> "DecisionTree":
        self.features = list(features)
        w = list(weights) if weights is not None else [1.0] * len(y)
        self.root = self._grow(list(range(len(y))), X, y, w, 0)
        return self

    def _leaf(self, idx, y, w) -> Dict:
        w1 = sum(w[i] for i in idx if y[i] == 1)
        w0 = sum(w[i] for i in idx if y[i] == 0)
        return {"leaf": 1 if w1 > w0 else 0, "w": [w0, w1]}

    def _grow(self, idx, X, y, w, depth) -> Dict:
        w1 = sum(w[i] for i in idx if y[i] == 1)
        w0 = sum(w[i] for i in idx if y[i] == 0)
        if depth >= self.max_depth or w0 == 0 or w1 == 0 or len(idx) < 2:
            return self._leaf(idx, y, w)
        best = None  # (impurity, feature, threshold, left, right)
        base = (w0 + w1) * self._gini(w0, w1)
        for f in range(len(self.features)):
            vals = sorted({X[i][f] for i in idx})
            for a, b in zip(vals, vals[1:]):
                thr = (a + b) / 2.0
                left = [i for i in idx if X[i][f] <= thr]
                right = [i for i in idx if X[i][f] > thr]
                l1 = sum(w[i] for i in left if y[i] == 1)
                l0 = sum(w[i] for i in left if y[i] == 0)
                r1 = sum(w[i] for i in right if y[i] == 1)
                r0 = sum(w[i] for i in right if y[i] == 0)
                if min(l0 + l1, r0 + r1) < self.min_leaf_weight:
                    continue
                imp = (l0 + l1) * self._gini(l0, l1) + (r0 + r1) * self._gini(r0, r1)
                if best is None or imp < best[0] - 1e-12:
                    best = (imp, f, thr, left, right)
        if best is None or best[0] >= base - 1e-12:
            return self._leaf(idx, y, w)
        _, f, thr, left, right = best
        return {"f": self.features[f], "t": thr,
                "l": self._grow(left, X, y, w, depth + 1), "r": self._grow(right, X, y, w, depth + 1)}

    # -- use ------------------------------------------------------------------------------
    def predict(self, row: Dict[str, float]) -> int:
        node = self.root
        while "leaf" not in node:
            node = node["l"] if row[node["f"]] <= node["t"] else node["r"]
        return node["leaf"]

    def to_json(self) -> Dict:
        return {"features": self.features, "max_depth": self.max_depth, "root": self.root}

    @classmethod
    def from_json(cls, d: Dict) -> "DecisionTree":
        t = cls(max_depth=d.get("max_depth", 3))
        t.features = d["features"]
        t.root = d["root"]
        return t

    def describe(self, node: Optional[Dict] = None, indent: int = 0) -> str:
        node = node or self.root
        pad = "  " * indent
        if "leaf" in node:
            return f"{pad}-> {'coordinate' if node['leaf'] else 'greedy'}  (weights greedy {node['w'][0]:.2f}, coordinate {node['w'][1]:.2f})\n"
        return (f"{pad}if {node['f']} <= {node['t']:.4g}:\n" + self.describe(node["l"], indent + 1)
                + f"{pad}else:\n" + self.describe(node["r"], indent + 1))


def load_models(path: Optional[Path] = None) -> Dict:
    """Trained trees by name: {'regime': {'A12_P36': tree json, ..., 'all': tree json}, 'distilled': tree json}."""
    p = Path(path) if path else MODELS_PATH
    return json.loads(p.read_text(encoding="utf-8"))
