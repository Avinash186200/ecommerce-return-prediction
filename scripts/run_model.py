"""Train and evaluate return classifiers on synthetic data (default) or your own order CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from returnpred.data import make_synthetic_orders  # noqa: E402
from returnpred.model import evaluate  # noqa: E402

DATE_COLS = ["order_date", "delivery_date", "user_dob", "user_reg_date"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--orders", help="CSV with the synthetic-data columns; omit for synthetic")
    ap.add_argument("--n-orders", type=int, default=20000)
    ap.add_argument("--signal", type=float, default=1.0, help="synthetic signal strength (0 = noise)")
    ap.add_argument("--label-lag-days", type=int, default=14)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", help="directory for the results CSV")
    a = ap.parse_args()

    if a.orders:
        orders = pd.read_csv(a.orders, parse_dates=DATE_COLS)
    else:
        orders = make_synthetic_orders(a.n_orders, signal_strength=a.signal, seed=a.seed)
    res = evaluate(orders, label_lag_days=a.label_lag_days, seed=a.seed)
    print(res.to_string(index=False))
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        res.to_csv(Path(a.out) / "results.csv", index=False)


if __name__ == "__main__":
    main()
