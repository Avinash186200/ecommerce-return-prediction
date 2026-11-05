# Methodology and leakage audit

## Problem

Predict at order time whether an order line will be returned (`return = 1`). A useful answer lets a shop flag risky orders, for example to show size advice before checkout.

## Prediction point and label lag

The model scores an order when it is placed. At that moment:

- `delivery_date` is unknown, so it is not a feature.
- The outcomes of recent orders are unknown too. A return arrives some days after delivery. `label_lag_days` (default 14) sets how long that takes: an order placed on day d has a known label from day d + lag + 1.

All history features are built with `merge_asof` on that "known from" date, per key. A test checks this against a brute-force count, and another checks that flipping an order's own label does not change its own features.

## Features

| Feature | Definition |
|---|---|
| `item_price` | Price of the line |
| `user_age_years` | (order_date - user_dob) / 365.25 |
| `account_age_days` | order_date - user_reg_date |
| `is_company` | Customer title is "Company" |
| `<key>_ret_rate` | Smoothed return rate of earlier, already-labelled orders for user, item, brand, size, colour and region, (returns + 20 * p0) / (orders + 20), with p0 the training return rate |
| `<key>_n_prior` | Number of those earlier labelled orders |

The original one-hot encoded item, size, colour, brand and region, about 2,250 sparse columns. The smoothed rates carry the same information in 12 columns and do not break on items that appear only in the scoring file.

## Validation design

- Time split: earliest 75% of days to train, last 25% to test, with a gap equal to the label lag so that every training label is known before the test period starts.
- Scaler fitted inside the pipeline.
- Fixed 0.5 threshold; AUC as the main metric.
- Baseline: always predict the majority class.

## What the original notebook did, and what changed

1. History features used a one-day shift, so returns were treated as known the day after the order. Fixed with the label lag.
2. Delivery time was a feature. Dropped.
3. Ages were measured against the run date. Now measured at the order date.
4. A random split was used. Now time-ordered with a gap.
5. The scaler was fitted on all rows, and for the scoring file on train + scoring rows together. Now fitted on training rows only.
6. Cut-offs were picked by eye from the outcome. Now fixed.

## Checks in CI

- Column contract of the synthetic data.
- Label lag: prior counts match a brute-force count.
- An order's own label never affects its own features.
- Ages are measured at the order date.
- Split gap is at least the label lag.
- New keys fall back to the training mean.
- **Noise test**: with no signal, AUC stays within 0.03 of 0.5.

## Dropped from the original

- The Keras network (on the original split it did worse than the random forest, and its predictions collapsed to a few values), permutation-importance exports, Excel and CSV dumps, and the presentation deck.
- Unused imports (database, email, MQTT and technical-analysis libraries left over from other work).

## Known limitations

- Synthetic data only, with a planted signal.
- A single split over about three months; there is no seasonality to test.
- The label lag is one fixed number. In practice it varies by delivery time.
