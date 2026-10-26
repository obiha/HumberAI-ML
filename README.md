# Late Deliveries

Predicts, at the moment a QuickBite order is placed, whether it will arrive
more than ten minutes late, so the company can offer a $4 voucher instead of
paying $15 in refunds and lost repeat custom when a late order is missed.

Team: Joshua Obiha, Gerardo Ahuatzin Galicia

## Setup
```
conda env create -f environment.yml
conda activate lab05-late-deliveries
pip install -r requirements.txt
pip install -e .
```

## Run
```
python -m latedeliveries.train
```

## Check it
```
python -m ruff format --check .
python -m ruff check .
python -m pytest
```

## Decisions

### Columns used and columns left out
Only columns known the instant an order is placed are used as features:

- **Numeric:** `distance_km`, `items`, `hour`, `courier_months`
- **Categorical:** `weather`, `day_type`, `vehicle`

Left out, with the reason for each:

- `order_id` — an identifier, not a predictive signal.
- `prep_minutes` — the kitchen has not started preparing the order yet when
  it is placed, so the real prep time is not known at prediction time. It
  is also mechanically tied to lateness in the scenario, so including it
  would leak the outcome.
- `customer_rating` — only given after the order has already arrived (it is
  missing for half of all orders because many customers never rate), so it
  cannot be known when the prediction is made.

### How the model was chosen
A `DummyClassifier` baseline (predicts the majority class, i.e. "never
late") and three real candidates — plain logistic regression, logistic
regression with `class_weight="balanced"`, and a random forest — were
compared with five-fold stratified cross-validation on the training data
only (see `reports/cv_results.csv`):

| model | accuracy | precision | recall | f1 | ROC-AUC |
|---|---|---|---|---|---|
| baseline (most frequent) | 0.844 ± 0.001 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.500 ± 0.000 |
| logistic regression | 0.862 ± 0.008 | 0.661 ± 0.068 | 0.239 ± 0.028 | 0.351 ± 0.039 | 0.789 ± 0.036 |
| logistic regression (balanced) | 0.739 ± 0.016 | 0.333 ± 0.027 | 0.672 ± 0.065 | 0.445 ± 0.037 | 0.788 ± 0.037 |
| random forest | 0.854 ± 0.006 | 0.602 ± 0.079 | 0.208 ± 0.016 | 0.308 ± 0.022 | 0.756 ± 0.036 |

**`logistic_regression_balanced` was chosen.** Its ROC-AUC is effectively
tied with plain logistic regression (both well ahead of the random forest)
but its recall is roughly three times higher (0.672 vs 0.239). Under the
scenario's cost rule, missing a late order costs $15 while flagging one
costs only $4, so a model that misses far fewer late orders is worth the
lower accuracy and precision — accuracy alone would have pointed to the
wrong model.

### What tuning changed
`GridSearchCV` searched `model__C` (0.01, 0.1, 1, 10) and `model__solver`
(`lbfgs`, `liblinear`) — 8 combinations — scored by ROC-AUC, because the
final decision uses a cost-based threshold rather than the default 0.5
cut-off, so the search should reward a model that ranks orders well across
all thresholds. The best settings were `C=0.01`, `solver=liblinear`, giving
a CV ROC-AUC of 0.7889. The untuned model scored 0.7881 ± 0.0368. The
difference (0.0008) is far smaller than the fold-to-fold spread, so
**tuning did not meaningfully beat the untuned model** on this data.

### How the threshold was chosen
Out-of-fold predicted probabilities were obtained with `cross_val_predict`
on the training data only, and every threshold from 0.05 to 0.95 (in steps
of 0.05) was scored with the scenario's cost rule: $4 per flagged order,
plus $15 per late order left unflagged. The threshold with the lowest total
cost on the training folds, **0.6**, was chosen — the test set played no
part in this choice. A fixed 0.5 cut-off would not reflect the fact that a
false negative costs almost four times as much as a false positive.

## Result

Test-set confusion matrix at threshold 0.6 (rows = actual, columns =
predicted; `[[on-time, late], ...]`):

```
[[564  69]
 [ 67  50]]
```

| metric | value |
|---|---|
| precision | 0.420 |
| recall | 0.427 |
| F1 | 0.424 |
| ROC-AUC | 0.752 |

Total cost on the test set, by policy:

| policy | cost |
|---|---|
| flag nobody (today's policy) | $1,755 |
| flag everybody | $3,000 |
| this model | **$1,481** |

The model cuts the cost of today's "flag nobody" policy by about 16%. The
test set was scored exactly once, so this number can be trusted as an
honest estimate; the CV standard deviations above (around 0.03-0.04 ROC-AUC)
suggest a replication on a different seed could plausibly land a few
percent either side of $1,481.

### Leakage, seen for real
Re-running the five-model comparison with `prep_minutes` and
`customer_rating` added back as features lifts the logistic-regression CV
ROC-AUC from **0.788 to 0.916** — a jump far larger than the fold-to-fold
spread (±0.03-0.04). A score that improves that sharply from two columns
that should not have been knowable at prediction time is a warning sign,
not good news, and confirms they were correctly excluded.

## What the tests cover

The 12 tests in `tests/` check: the feature list excludes the three
leaky/identifier columns; the stratified split preserves the late rate and
is repeatable from the same seed; predicted probabilities are valid
(shape, in `[0, 1]`, rows sum to 1); the pipeline predicts for a row with a
missing numeric value and a weather category it has never seen; the cost
function on a hand-worked example and a parametrized set of hand-worked
examples; that the cheapest threshold can be either extreme (flag nobody
when nothing is late, flag everybody when everything is late); and that a
saved-then-reloaded pipeline (via `tmp_path`) predicts identically to the
original.

Not tested: the exact numeric value of the cross-validation scores or the
chosen hyperparameters. Those depend on library versions and are checked by
reading `reports/cv_results.csv` and `reports/metrics.json` instead, since
pinning them in a test would make the suite fail on an unrelated library
upgrade rather than on a real bug.

## Known limitations

Recall at the chosen threshold is only 0.427, so more than half of all
late orders are still not flagged. The model also only sees information
available at order placement; a feature such as the kitchen's current
backlog (not simulated here) would likely improve on this. Finally, the
whole dataset is invented by `data.py`, so these numbers describe how well
the method works on this simulation, not on QuickBite's real orders.
