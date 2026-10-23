"""Export the trained Random Forest and scaler to plain numpy arrays.

The API scores and explains with dashboard/backend/forest.py, which reads this
file instead of the pickles, so it runs without scikit-learn or shap (needed to
fit inside a Vercel Python function). Re-run after retraining:

    python -m dashboard.backend.export_model
"""

import argparse
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def export(model_path: Path, scaler_path: Path, out_path: Path) -> None:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    fraud = list(model.classes_).index(1)

    left, right, feature, threshold, value, cover, roots = [], [], [], [], [], [], []
    edge_leaf, edge_parent, edge_child = [], [], []
    n_leaves = 0
    offset = 0

    for est in model.estimators_:
        t = est.tree_
        roots.append(offset)
        is_leaf = t.children_left == -1
        left.append(np.where(is_leaf, -1, t.children_left + offset))
        right.append(np.where(is_leaf, -1, t.children_right + offset))
        feature.append(np.where(is_leaf, 0, t.feature))
        threshold.append(t.threshold)
        v = t.value[:, 0, :]
        value.append(v[:, fraud] / v.sum(axis=1))
        cover.append(t.weighted_n_node_samples)

        # Every root-to-leaf path as (leaf, parent, child) edges, for TreeSHAP.
        stack = [(0, [])]
        while stack:
            node, path = stack.pop()
            if is_leaf[node]:
                for parent, child in path:
                    edge_leaf.append(n_leaves)
                    edge_parent.append(parent + offset)
                    edge_child.append(child + offset)
                n_leaves += 1
                continue
            for child in (t.children_left[node], t.children_right[node]):
                stack.append((child, path + [(node, child)]))

        offset += t.node_count

    np.savez_compressed(
        out_path,
        left=np.concatenate(left).astype(np.int32),
        right=np.concatenate(right).astype(np.int32),
        feature=np.concatenate(feature).astype(np.int32),
        threshold=np.concatenate(threshold),
        value=np.concatenate(value),
        cover=np.concatenate(cover),
        roots=np.asarray(roots, dtype=np.int32),
        edge_leaf=np.asarray(edge_leaf, dtype=np.int32),
        edge_parent=np.asarray(edge_parent, dtype=np.int32),
        edge_child=np.asarray(edge_child, dtype=np.int32),
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
        n_features=np.int32(model.n_features_in_),
    )
    print(f"Wrote {out_path} ({len(roots)} trees, {offset} nodes, {n_leaves} leaves)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=ROOT / "models/random_forest.pkl")
    parser.add_argument("--scaler", type=Path, default=ROOT / "data/processed/scaler.pkl")
    parser.add_argument("--out", type=Path, default=ROOT / "models/random_forest_trees.npz")
    args = parser.parse_args()
    export(args.model, args.scaler, args.out)
