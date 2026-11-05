import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from returnpred.data import make_synthetic_orders  # noqa: E402
from returnpred.features import PRIOR, build_features, time_split  # noqa: E402
from returnpred.model import evaluate  # noqa: E402


def test_columns_match_contract():
    df = make_synthetic_orders(500, seed=1)
    assert list(df.columns) == ["order_item_id", "order_date", "delivery_date", "item_id",
                                "item_size", "item_color", "brand_id", "item_price", "user_id",
                                "user_title", "user_dob", "user_state", "user_reg_date", "return"]
    assert df["order_item_id"].is_unique


def test_user_history_uses_only_known_labels():
    """For every row, the prior count equals a brute-force count of orders known by then."""
    df = make_synthetic_orders(1500, n_users=100, seed=2)
    lag = 10
    feats = build_features(df, lag).set_index("order_item_id")
    for _, r in df.sample(60, random_state=0).iterrows():
        known = df[(df["user_id"] == r.user_id)
                   & (df["order_date"] <= r.order_date - pd.Timedelta(days=lag + 1))]
        assert feats.loc[r.order_item_id, "user_id_n_prior"] == len(known)


def test_own_label_never_used():
    """Flipping one order's label must not change that order's own features."""
    df = make_synthetic_orders(1500, n_users=100, seed=3)
    a = build_features(df, 14, global_rate=0.5).set_index("order_item_id")
    df2 = df.copy()
    df2.loc[700, "return"] = 1 - df2.loc[700, "return"]
    b = build_features(df2, 14, global_rate=0.5).set_index("order_item_id")
    oid = df.loc[700, "order_item_id"]
    pd.testing.assert_series_equal(a.loc[oid], b.loc[oid])


def test_ages_measured_at_order_date():
    df = make_synthetic_orders(300, seed=4)
    f = build_features(df).set_index("order_item_id").loc[df["order_item_id"]]
    expected = (df["order_date"] - df["user_dob"]).dt.days.to_numpy() / 365.25
    assert np.allclose(f["user_age_years"].to_numpy(), expected)


def test_time_split_gap():
    df = make_synthetic_orders(3000, seed=5)
    tr, te = time_split(df, gap_days=14)
    assert te["order_date"].min() - tr["order_date"].max() >= pd.Timedelta(days=14)


def test_shrinkage_with_no_history():
    df = make_synthetic_orders(300, seed=6)
    f = build_features(df, 14, global_rate=0.4)
    first_day = df["order_date"] == df["order_date"].min()
    ids = df.loc[first_day, "order_item_id"]
    rates = f.set_index("order_item_id").loc[ids, "user_id_ret_rate"]
    assert np.allclose(rates, PRIOR * 0.4 / PRIOR)


def test_noise_input_shows_no_edge():
    res = evaluate(make_synthetic_orders(12000, signal_strength=0.0, seed=8)).set_index("model")
    assert abs(res.loc["logreg", "auc"] - 0.5) < 0.03
    assert abs(res.loc["rf", "auc"] - 0.5) < 0.03


def test_signal_is_recovered():
    res = evaluate(make_synthetic_orders(12000, signal_strength=1.0, seed=8)).set_index("model")
    assert res.loc["logreg", "auc"] > 0.6
