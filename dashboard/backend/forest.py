"""Numpy-only Random Forest scoring and exact TreeSHAP.

Reads the arrays written by export_model.py. Matches sklearn's
RandomForestClassifier.predict_proba and shap's path-dependent TreeExplainer,
so the API needs neither library at runtime.
"""

from math import factorial

import numpy as np

# Leaves are explained in blocks to bound memory (block x features^2 floats).
SHAP_BLOCK = 2048


class Forest:
    def __init__(self, path):
        d = np.load(path)
        self.left = d["left"]
        self.right = d["right"]
        self.feature = d["feature"]
        self.threshold = d["threshold"]
        self.value = d["value"]
        self.roots = d["roots"]
        self.scaler_mean = d["scaler_mean"]
        self.scaler_scale = d["scaler_scale"]
        self.n_features = int(d["n_features"])
        self.n_trees = len(self.roots)

        cover = d["cover"]
        edge_child = d["edge_child"]
        self._edge_leaf = d["edge_leaf"]
        self._edge_parent = d["edge_parent"]
        self._edge_feat = self.feature[self._edge_parent]
        self._edge_left = edge_child == self.left[self._edge_parent]

        # Edges were written leaf by leaf; each leaf's last edge ends at it.
        last = np.r_[np.flatnonzero(np.diff(self._edge_leaf)), len(self._edge_leaf) - 1]
        leaf_node = edge_child[last]

        # z[leaf, j]: share of training cover that follows this path's splits on j.
        self._z = np.ones((len(leaf_node), self.n_features))
        np.multiply.at(
            self._z,
            (self._edge_leaf, self._edge_feat),
            cover[edge_child] / cover[self._edge_parent],
        )

        # Shapley weights w[k] = k!(n-k-1)!/n!, laid out so a^T W b sums
        # w[deg] over every coefficient pair of two polynomials a and b.
        n = self.n_features
        w = np.array([factorial(k) * factorial(n - k - 1) / factorial(n) for k in range(n)])
        idx = np.add.outer(np.arange(n), np.arange(n))
        self._w = np.where(idx < n, w[np.minimum(idx, n - 1)], 0.0)

        self._leaf_value = self.value[leaf_node] / self.n_trees
        self.expected_value = float((self._leaf_value * self._z.prod(axis=1)).sum())

    def scale(self, x):
        return (x - self.scaler_mean) / self.scaler_scale

    def predict_proba(self, x):
        """Fraud probability for each row of x (already scaled), shape (n,)."""
        # sklearn compares float32 inputs against float64 thresholds.
        x = np.asarray(x, dtype=np.float32).astype(np.float64)
        rows = np.arange(len(x))[:, None]
        node = np.broadcast_to(self.roots, (len(x), self.n_trees)).copy()
        while True:
            internal = self.left[node] >= 0
            if not internal.any():
                break
            go_left = x[rows, self.feature[node]] <= self.threshold[node]
            nxt = np.where(go_left, self.left[node], self.right[node])
            node = np.where(internal, nxt, node)
        return self.value[node].mean(axis=1)

    def shap_values(self, x):
        """SHAP contributions toward the fraud probability for one scaled row."""
        x = np.asarray(x, dtype=np.float32).astype(np.float64)
        n = self.n_features

        # o[leaf, j]: 1 if x follows every split on j along the path, else 0.
        follows = (x[self._edge_feat] <= self.threshold[self._edge_parent]) == self._edge_left
        o = np.ones_like(self._z)
        o[self._edge_leaf[~follows], self._edge_feat[~follows]] = 0.0

        phi = np.zeros(n)
        for start in range(0, len(o), SHAP_BLOCK):
            z_b = self._z[start : start + SHAP_BLOCK]
            o_b = o[start : start + SHAP_BLOCK]
            m = len(z_b)
            # Polynomials in t of prod_{j<i} (z_j + o_j t) and prod_{j>i} (z_j + o_j t).
            pre = np.zeros((m, n, n))
            suf = np.zeros((m, n, n))
            pre[:, 0, 0] = 1.0
            suf[:, n - 1, 0] = 1.0
            for i in range(1, n):
                pre[:, i] = z_b[:, i - 1, None] * pre[:, i - 1]
                pre[:, i, 1:] += o_b[:, i - 1, None] * pre[:, i - 1, :-1]
                j = n - 1 - i
                suf[:, j] = z_b[:, j + 1, None] * suf[:, j + 1]
                suf[:, j, 1:] += o_b[:, j + 1, None] * suf[:, j + 1, :-1]
            weighted = ((pre @ self._w) * suf).sum(axis=-1)
            leaf_value = self._leaf_value[start : start + SHAP_BLOCK, None]
            phi += (leaf_value * (o_b - z_b) * weighted).sum(axis=0)
        return phi
