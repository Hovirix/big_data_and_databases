# Preparation and majority baseline

Reproduce: `nix develop --command uv run python prepare.py`
Check: `nix develop --command uv run python -m unittest test_prepare.py`

- Shared stratified 80/20 split, seed 42: 5,634 training / 1,409 test customers.
- `split_ids.csv` preserves partition membership and row order within each partition.
- All 19 candidate predictors retained, including tenure; customerID never enters preprocessing.
- TotalCharges blanks become missing; numerical imputation uses training medians.
- One-hot encoding treats SeniorCitizen as categorical and retains service-absence categories.
- Logistic Regression preprocessing scales numerical features. Use
  `make_preprocessor(scale_numeric=False)` for Random Forest.
- Put preprocessing inside each model pipeline: do not reuse the full-training
  fitted transformer during validation.
- `settings.json` records data hash, versions, split/CV settings and fitted training
  preprocessing parameters; `encoded_features.csv` records transformed feature order.

## Baseline: training-only five-fold cross-validation

Pooled out-of-fold ROC-AUC: **0.500**.
Precision: **0.000**; recall: **0.000**;
F1: **0.000**.

Confusion matrix (actual rows / predicted columns):

| | Predicted No | Predicted Yes |
|---|---:|---:|
| Actual No | 4139 | 0 |
| Actual Yes | 1495 | 0 |

The majority baseline predicts No for every validation customer. Its positive-class
probability is zero; precision is undefined without positive predictions and reported
as zero (`zero_division=0`). It detects no churners.

The test set has not been transformed or evaluated. Next: compare Logistic Regression
and Random Forest using the same training folds; reserve test evaluation until model
settings are fixed. The earlier full-dataset descriptive exploration remains a limitation.
