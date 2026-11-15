# E-commerce Return Prediction

[![CI](https://github.com/Avinash186200/ecommerce-return-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/Avinash186200/ecommerce-return-prediction/actions/workflows/ci.yml) ![Python](https://img.shields.io/badge/python-3.11-blue) [![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

I built this to predict, at the moment an order is placed, whether an online fashion order line will be sent back. It started as a 2021 take-home case study: a few months of order lines (item, size, colour, brand, price, customer details) with a returned / kept label.

The original data is not mine to share and is not in this repo. Everything here runs on a **synthetic order file with the same columns**, so you can clone it and reproduce every number below. This is a small repo: one dataset, one question, two models.

## At a glance

| | |
|---|---|
| **Input** | One row per order line: `order_item_id, order_date, delivery_date, item_id, item_size, item_color, brand_id, item_price, user_id, user_title, user_dob, user_state, user_reg_date, return` |
| **Method** | Point-in-time features (customer, item, brand, size, colour and region return rates from labels already known, ages at order date), time-ordered split with a gap, logistic regression and random forest |
| **Output** | P(return) per order line; AUC and accuracy against a majority-class baseline |
| **Leakage controls** | Return labels only count once a label lag has passed, no `delivery_date` at order time, scaler fitted on training rows, fixed 0.5 threshold, noise test in CI |
| **Status** | Small case study, rebuilt leak-free. Synthetic results only |

## How it works

```
order lines ──► point-in-time features ──► time split + gap ──► scaler + model ──► P(return)
 (CSV)          features.build_features    features.time_split   model.evaluate
```

1. **History features.** For each key (customer, item, brand, size, colour, region), the return rate of earlier orders whose outcome was already known: orders placed at least `label_lag_days + 1` days before (default 14). Rates are shrunk toward the training-set mean when history is thin.
2. **Customer features.** Age and account age, both measured at the order date.
3. **Split by time.** Train on the earliest 75% of days, skip a gap equal to the label lag, test on the last 25%.
4. **Models.** Logistic regression and a shallow random forest, each in a pipeline with a scaler. Threshold fixed at 0.5.

## Results on synthetic data

`python scripts/run_model.py` (seed 7, 20,000 order lines over 90 days; 11,786 training rows, 5,037 test rows, the 14-day gap between them dropped):

| Model | Test AUC | Test accuracy | Test AUC with `--signal 0` |
|---|---|---|---|
| Majority class | 0.500 | 50.8% | 0.500 |
| Logistic regression | 0.679 | 63.4% | 0.501 |
| Random forest | 0.659 | 62.1% | 0.509 |

With `--signal 0` (returns independent of every feature), both models score an AUC of about 0.50 (0.501 and 0.509); a noise test in CI enforces this. With about 5,000 test rows the standard error of an AUC is roughly 0.008, so 0.509 on noise is within sampling error, while the 0.66-0.68 with signal is far outside it. The gap between logistic regression and the random forest (0.02) is small enough that I would not pick a model on this one split alone. The generator plants a hidden per-customer and per-item return tendency, so the history features carry real signal here by design.

> **About the original data.** I don't quote the original case-study accuracy figures. They came from a random split, with history features that ignored the return lag and a scaler fitted on all rows, so they are optimistic by construction. I have not re-run the corrected method on that data, so I make no real-data performance claim.

## Leakage and bias audit

| Issue | Effect | Fix |
|---|---|---|
| Customer and item return history counted returns from the **previous order day** | A return is not known until days after delivery, so the features used outcomes that were not yet known | Labels count only after `label_lag_days` (default 14) |
| `delivery_date` / delivery time used as a feature | Not known when the order is placed | Dropped |
| Ages computed against **the date the notebook ran** | Not reproducible; drifts every day | Measured at the order date |
| Random 70/30 split on time-dependent features | Future orders train the model that scores past ones | Time-ordered split with a gap |
| `StandardScaler` fitted on all rows; for the scoring file, fitted on training + scoring rows combined | Test statistics leak into training | Scaler inside the pipeline, fitted on training rows only |
| Cut-offs of 0.51, 0.65 and 0.48 picked by looking at the outcome | Threshold fitted on evaluation data | Fixed 0.5, and AUC reported, which needs no threshold |
| Missing delivery times filled with the full-sample mean | Small leak of test statistics | Feature dropped |

## Repo layout

```
src/returnpred/  data.py (synthetic orders) · features.py (point-in-time features, split) · model.py
scripts/         run_model.py
tests/           brute-force label-lag check, own-label check, ages, split gap, noise test
docs/            methodology.md
```

## Quick start

```bash
pip install -r requirements.txt
python -m pytest -q
python scripts/run_model.py                 # synthetic
python scripts/run_model.py --signal 0      # pure noise: AUC about 0.5
python scripts/run_model.py --orders your_orders.csv --label-lag-days 21 --out results
```

## Original research code

[`original/`](original/README.md) holds the code as I first wrote it for this work, sanitised and nothing else:
client, company and person names replaced, credentials and local paths removed, data files and notebook outputs
dropped. It is kept so the real work can be read alongside the maintained version. Some of it contains the issues
described above; comparing the two shows each fix.

## License

MIT

See [CONTRIBUTING.md](CONTRIBUTING.md) and the [CHANGELOG](CHANGELOG.md).
