# Training-only model comparison

Reproduce after `prepare.py`: `nix develop --command uv run python train.py`.
Checks: `nix develop --command uv run python -m unittest test_train.py`.
Source: [IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d).

All models use the saved 5,634 training customers and the baseline's exact
five stratified validation folds. Imputation, encoding and scaling are fitted
separately inside each fold. Logistic Regression scales numerical variables;
Random Forest does not. All 19 candidate predictors are retained.

## Cross-validation results

| Model | Mean ROC-AUC | ROC-AUC SD | Mean precision | Mean recall | Mean F1 |
|---|---:|---:|---:|---:|---:|
| majority_baseline | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| logistic_regression | 0.8463 | 0.0127 | 0.6552 | 0.5485 | 0.5965 |
| random_forest | 0.8465 | 0.0089 | 0.6782 | 0.5084 | 0.5808 |

These are means of fold scores, not pooled out-of-fold scores. SD describes
variation across folds, not a confidence interval. Precision, recall and F1
use the estimator's default binary decision rule (threshold 0.5; exact ties
predict No). Undefined baseline precision is reported as zero.

## Selected settings

Selection criterion: highest mean validation ROC-AUC.
- logistic_regression: `{'model__C': 10}`.
- random_forest: `{'model__max_depth': 10, 'model__min_samples_leaf': 5}`.

Full grid results, including per-fold metrics, are in `*_cv_results.csv`.
`selected_settings.json` records selected/grid/default parameters, preprocessing
choices, versions, input hashes and the fixed threshold.

## Limits and next step

The same folds were used for tuning and comparison, so selected CV scores can
be optimistic; they are not independent final performance estimates.
Earlier descriptive exploration also used the full dataset.

No final full-training refit, test transformation, test prediction or test
evaluation has been performed. Next, keep settings fixed, fit both selected
pipelines and the baseline on training data, then evaluate once on the saved
test customers. Report probability-based ROC-AUC, precision, recall, F1 and
confusion matrices. Interpret predictors as associations, not causal effects.
