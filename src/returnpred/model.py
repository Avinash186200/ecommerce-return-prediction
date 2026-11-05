"""Fit and evaluate return classifiers on a time-ordered split."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import build_features, feature_columns, time_split


def make_model(name: str, seed: int = 0) -> Pipeline:
    """Unfitted scaler + classifier pipeline (``logreg`` or ``rf``)."""
    if name == "logreg":
        clf = LogisticRegression(max_iter=1000)
    elif name == "rf":
        clf = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=20,
                                     random_state=seed, n_jobs=-1)
    else:
        raise ValueError(f"unknown model {name!r}")
    return Pipeline([("scale", StandardScaler()), ("clf", clf)])


def evaluate(orders: pd.DataFrame, models: tuple[str, ...] = ("logreg", "rf"),
             label_lag_days: int = 14, threshold: float = 0.5, seed: int = 0) -> pd.DataFrame:
    """Time split, point-in-time features, fixed 0.5 threshold. Returns one row per model."""
    train, test = time_split(orders, gap_days=label_lag_days)
    feats = build_features(orders, label_lag_days, global_rate=float(train["return"].mean()))
    cols = feature_columns(feats)
    tr = train[["order_item_id", "return"]].merge(feats, on="order_item_id")
    te = test[["order_item_id", "return"]].merge(feats, on="order_item_id")

    majority = int(tr["return"].mean() >= 0.5)
    rows = [{"model": "majority_class", "auc": 0.5,
             "accuracy": accuracy_score(te["return"], np.full(len(te), majority))}]
    for name in models:
        m = make_model(name, seed).fit(tr[cols], tr["return"])
        p = m.predict_proba(te[cols])[:, 1]
        rows.append({"model": name, "auc": roc_auc_score(te["return"], p),
                     "accuracy": accuracy_score(te["return"], (p >= threshold).astype(int))})
    out = pd.DataFrame(rows)
    out["train_rows"], out["test_rows"] = len(tr), len(te)
    return out.round(3)
