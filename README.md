# E-commerce Return Prediction

[![CI](https://github.com/Avinash186200/ecommerce-return-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/Avinash186200/ecommerce-return-prediction/actions/workflows/ci.yml) ![Python](https://img.shields.io/badge/python-3.11-blue) [![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

I built this to predict, at the moment an order is placed, whether an online fashion order line will be sent back. It started as a 2021 take-home case study: a few months of order lines (item, size, colour, brand, price, customer details) with a returned / kept label.

The original data is not mine to share and is not in this repo. Everything here runs on a **synthetic order file with the same columns**, so you can clone it and reproduce every number below. This is a small repo: one dataset, one question, two models.

## What problem this solves

Online fashion shops get a large share of orders sent back, and every return costs shipping and handling. If you can tell at checkout which order lines are likely to come back, you can act early, for example by showing size advice. The hard part is not the model. It is making sure the model only uses information that was really known when the order was placed.

## What you will learn

- How to build "history" features (past return rates) without letting future outcomes leak in.
- Why a label is not known the moment an event happens, and how a label lag fixes that.
- Why a random train/test split is wrong for time-ordered data, and how a time split with a gap works.
- Why the scaler belongs inside the pipeline, fitted on training rows only.
- How a noise test (a run with no signal in the data) proves the pipeline is not leaking.
- How to read AUC and accuracy against a majority-class baseline.

## Key concepts

- **Prediction point:** the moment the model is asked for an answer. Here, when the order is placed. Only facts known then may be features.
- **Label lag:** a return is only known some days after the order. An order's label counts in history from `order_date + label_lag_days + 1`.
- **Point-in-time merge:** `pandas.merge_asof` joins each order to the latest history row whose "known from" date is on or before the order date.
- **Smoothed (shrunk) rate:** `(returns + 20 * p0) / (orders + 20)`. Thin history is pulled toward the training mean `p0`, so one return out of one order does not mean 100%.
- **Time split with a gap:** train on early days, skip a gap as long as the label lag, test on later days. No test-period outcome can reach training.
- **Train-only scaling:** the `StandardScaler` sits inside the scikit-learn `Pipeline`, so it only ever sees training rows.
- **Majority-class baseline:** always predict the most common answer. A model must beat it to be worth anything.
- **AUC:** the chance that a random returned line gets a higher score than a random kept line. 0.5 is a coin flip. It needs no threshold.
- **Noise test:** set `--signal 0` so returns are random. A clean pipeline then scores an AUC near 0.5.

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

### Reading order

There are no notebooks here. Read the code in this order; each file is short.

1. [`src/returnpred/data.py`](src/returnpred/data.py): `make_synthetic_orders` builds the order file. It teaches what the input looks like, and how a hidden per-customer and per-item return tendency is planted so there is something to find.
2. [`src/returnpred/features.py`](src/returnpred/features.py): `_prior_rate` and `build_features` build the point-in-time features; `time_split` makes the split with a gap. This is the core of the repo: how to stop future labels leaking into features.
3. [`src/returnpred/model.py`](src/returnpred/model.py): `make_model` and `evaluate`. It teaches the pipeline (scaler + model), the baseline, and the metrics.
4. [`scripts/run_model.py`](scripts/run_model.py): the command line that ties it together. Run it, then run it again with `--signal 0`.
5. [`tests/test_returnpred.py`](tests/test_returnpred.py): each test checks one leakage rule. Reading the tests is the fastest way to see what "leak-free" means in code.
6. [`docs/methodology.md`](docs/methodology.md): the full method and the audit of the original notebook.

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
src/returnpred/
  data.py          synthetic order lines with a planted return tendency
  features.py      point-in-time history features, ages, time split with a gap
  model.py         pipelines (scaler + logistic regression / random forest) and evaluation
scripts/
  run_model.py     command line: build features, split, train, print the results table
tests/
  test_returnpred.py   brute-force label-lag check, own-label check, ages, split gap, noise test
docs/
  README.md        index of the docs
  methodology.md   method, validation design and leakage audit
original/          the original 2021 notebooks (read-only record)
.github/workflows/ci.yml   runs the tests and the script on every push
requirements.txt · CHANGELOG.md · CONTRIBUTING.md · LICENSE
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

[`original/`](original/README.md) holds the code as I first wrote it for this project. It is kept so you can
read the first version alongside the maintained one; some of it contains the issues in the audit table above,
and comparing the two shows each fix.

## License

MIT

See [CONTRIBUTING.md](CONTRIBUTING.md) and the [CHANGELOG](CHANGELOG.md).
