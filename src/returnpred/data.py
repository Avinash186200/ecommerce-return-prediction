"""Synthetic order-line data with the same columns as the original case-study data."""

from __future__ import annotations

import numpy as np
import pandas as pd

SIZES = ["s", "m", "l", "xl", "xxl", "38", "40", "42"]
COLORS = ["black", "navy", "grey", "white", "brown", "green", "red", "blue"]


def make_synthetic_orders(n_orders: int = 20000, n_users: int = 3000, n_items: int = 300,
                          start: str = "2020-01-06", days: int = 90,
                          signal_strength: float = 1.0, seed: int = 7) -> pd.DataFrame:
    """Generate synthetic order lines.

    Return probability depends on a hidden per-user propensity, a hidden per-item
    "fit" problem, and price, all scaled by ``signal_strength``. With
    ``signal_strength=0`` every line is returned with the same probability, so no
    model should beat chance.

    Returns:
        DataFrame with columns ``order_item_id, order_date, delivery_date, item_id,
        item_size, item_color, brand_id, item_price, user_id, user_title, user_dob,
        user_state, user_reg_date, return``.
    """
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp(start)

    users = pd.DataFrame({
        "user_id": np.arange(n_users),
        "user_title": rng.choice(["Mrs", "Mr", "Company", "not reported"], n_users, p=[.88, .08, .02, .02]),
        "user_dob": t0 - pd.to_timedelta(rng.integers(18 * 365, 80 * 365, n_users), unit="D"),
        "user_state": rng.integers(1, 13, n_users),
        "user_reg_date": t0 - pd.to_timedelta(rng.integers(0, 6 * 365, n_users), unit="D"),
        "u_prop": rng.normal(0, 1, n_users),
    })
    items = pd.DataFrame({
        "item_id": np.arange(n_items),
        "brand_id": rng.integers(1, 60, n_items),
        "base_price": rng.choice([19.9, 29.9, 39.9, 49.9, 69.9, 89.9, 129.9], n_items),
        "i_fit": rng.normal(0, 1, n_items),
    })

    u = rng.integers(0, n_users, n_orders)
    i = rng.integers(0, n_items, n_orders)
    order_date = t0 + pd.to_timedelta(np.sort(rng.integers(0, days, n_orders)), unit="D")
    ship_days = rng.integers(1, 10, n_orders).astype(float)
    delivery = order_date + pd.to_timedelta(ship_days, unit="D")
    delivery = pd.Series(delivery).where(rng.random(n_orders) > 0.09)  # some delivery dates missing

    df = pd.DataFrame({
        "order_item_id": np.arange(1, n_orders + 1),
        "order_date": order_date,
        "delivery_date": delivery.to_numpy(),
        "item_id": i,
        "item_size": rng.choice(SIZES, n_orders),
        "item_color": rng.choice(COLORS, n_orders),
        "user_id": u,
    })
    df = df.merge(items, on="item_id").merge(users, on="user_id")
    df["item_price"] = df["base_price"]
    z = signal_strength * (0.8 * df["u_prop"] + 0.6 * df["i_fit"]
                           + 0.4 * (df["item_price"] - 60) / 40)
    p = 1 / (1 + np.exp(-(z - 0.1)))
    df["return"] = (rng.random(len(df)) < p).astype(int)

    cols = ["order_item_id", "order_date", "delivery_date", "item_id", "item_size", "item_color",
            "brand_id", "item_price", "user_id", "user_title", "user_dob", "user_state",
            "user_reg_date", "return"]
    return df[cols].sort_values(["order_date", "order_item_id"]).reset_index(drop=True)
