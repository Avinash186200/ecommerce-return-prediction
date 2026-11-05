"""Point-in-time features for return prediction, computed at order time.

A return is only known some days after the order (delivery plus the customer's
return window). ``label_lag_days`` models that: when scoring an order placed on
day t, only orders placed on or before ``t - label_lag_days`` have a known label.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HISTORY_KEYS = ["user_id", "item_id", "brand_id", "item_size", "item_color", "user_state"]
PRIOR = 20.0  # pseudo-orders for shrinking small-sample return rates toward the global mean


def _prior_rate(df: pd.DataFrame, key: str, lag: int, global_rate: float) -> pd.DataFrame:
    """Smoothed return rate and order count for ``key`` using only labels known by order time."""
    daily = df.groupby([key, "order_date"])["return"].agg(["sum", "count"]).reset_index()
    daily = daily.sort_values([key, "order_date"])
    daily[["cum_ret", "cum_n"]] = daily.groupby(key)[["sum", "count"]].cumsum()
    daily["known_from"] = daily["order_date"] + pd.Timedelta(days=lag + 1)

    lookup = df[["order_item_id", key, "order_date"]].sort_values("order_date")
    hist = daily[[key, "known_from", "cum_ret", "cum_n"]].sort_values("known_from")
    m = pd.merge_asof(lookup, hist, left_on="order_date", right_on="known_from", by=key,
                      direction="backward")
    m[["cum_ret", "cum_n"]] = m[["cum_ret", "cum_n"]].fillna(0.0)
    rate = (m["cum_ret"] + PRIOR * global_rate) / (m["cum_n"] + PRIOR)
    return pd.DataFrame({"order_item_id": m["order_item_id"].to_numpy(),
                         f"{key}_ret_rate": rate.to_numpy(),
                         f"{key}_n_prior": m["cum_n"].to_numpy()})


def build_features(df: pd.DataFrame, label_lag_days: int = 14,
                   global_rate: float | None = None) -> pd.DataFrame:
    """Return one feature row per order line, keyed by ``order_item_id``.

    Ages are measured at the order date, not at the date the code runs.
    ``delivery_date`` is not used: it is not known when the order is placed.
    ``global_rate`` should come from training data; it only sets the shrinkage target.
    """
    if global_rate is None:
        global_rate = float(df["return"].mean())
    out = pd.DataFrame({"order_item_id": df["order_item_id"].to_numpy()})
    out["item_price"] = df["item_price"].to_numpy()
    out["user_age_years"] = ((df["order_date"] - df["user_dob"]).dt.days / 365.25).to_numpy()
    out["account_age_days"] = (df["order_date"] - df["user_reg_date"]).dt.days.to_numpy()
    out["is_company"] = (df["user_title"] == "Company").astype(int).to_numpy()
    for key in HISTORY_KEYS:
        out = out.merge(_prior_rate(df, key, label_lag_days, global_rate), on="order_item_id")
    return out


def feature_columns(feats: pd.DataFrame) -> list[str]:
    """Model input columns: everything except the row key."""
    return [c for c in feats.columns if c != "order_item_id"]


def time_split(df: pd.DataFrame, test_frac: float = 0.25, gap_days: int = 14
               ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on the earliest days, test on the last ``test_frac`` of days, with a gap between."""
    dates = np.sort(df["order_date"].unique())
    cut = pd.Timestamp(dates[int(len(dates) * (1 - test_frac))])
    train = df[df["order_date"] < cut - pd.Timedelta(days=gap_days)]
    test = df[df["order_date"] >= cut]
    return train, test
