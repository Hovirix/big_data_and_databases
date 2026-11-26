# Dataset overview

Source: [IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d).
One row represents one customer. These are descriptive results, not model evaluations.

## Reproduce

From the repository root, using the pandas dependency in `pyproject.toml`:

```sh
nix develop --command uv run python overview.py
nix develop --command uv run python -m unittest test_overview.py
```

The source CSV is read only. Outputs in `outputs/` are regenerated.

## Dataset and quality

- 7,043 customers, 21 columns.
- Churn = Yes (customer left, positive class): 1,869 (26.54%).
- Churn = No (customer did not leave): 5,174 (73.46%).
- Duplicate rows: 0; duplicate customer IDs: 0.
- TotalCharges blanks: 11; of these,
  11 have tenure = 0.
- Blanks become missing numeric values; no rows are removed and no values imputed.
- TotalCharges is loaded as text because of blank entries and converted to numeric.
- SeniorCitizen is categorical despite numeric storage.
- "No internet service" and "No phone service" are valid categories, not missing values.

## Predictors

The 19 candidate predictors fall into three groups:

| Group | Variables |
|---|---|
| Demographics | `gender`, `SeniorCitizen`, `Partner`, `Dependents` |
| Subscription characteristics (including billing) | `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges` |
| Subscribed services | `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` |

`customerID` is an identifier, not a predictor; `Churn` is the target.

## Example association: contract and churn

| Contract | Customers | Churn rate |
|---|---:|---:|
| Month-to-month | 3875 | 42.71% |
| One year | 1473 | 11.27% |
| Two year | 1695 | 2.83% |

These are unadjusted associations, not evidence that changing contracts causes retention.

## Output tables

- `data_dictionary.csv`: every field's role, meaning, type and observed categories/range.
- `quality.json`: missingness, duplicates and basic validity checks.
- `numeric_summary.csv`: counts, mean, standard deviation, quartiles and range.
- `numeric_by_churn.csv`: the same numerical summaries within each churn class.
- `churn_distribution.csv`: target counts and percentages.
- `categorical_churn.csv`: category counts, dataset shares, churn counts and rates,
  covering demographics, subscriptions, services and billing.
- [churn_patterns.md](churn_patterns.md): ordered group comparisons and descriptive findings.
- `group_churn.csv`: combined group rates, including tenure and charge bins, with
  percentage-point differences from overall churn.
- `group_settings.json`: reproducible descriptive bin boundaries.

## Answering the research question

The data describes subscribed services, not measured usage: no call minutes,
data consumption or viewing activity. Use "subscribed services" in the question
or explicitly acknowledge this mismatch.

The tables describe associations; predictive performance is not established yet.
Next compare Logistic Regression and Random Forest against a majority baseline
using a shared fixed-seed stratified 80/20 split. Retain customerID for tracing,
never as a predictor, and retain tenure as a candidate predictor.
Fit preprocessing and tune models only within training data and validation folds.
Reserve test data for final ROC-AUC (probabilities), precision, recall, F1 and
confusion matrices. Accuracy alone is inadequate: always predicting No gives
73.46% accuracy here but zero churn recall.

Use Logistic Regression coefficients/odds ratios and Random Forest permutation
importance for model interpretation; related predictors such as tenure and
TotalCharges can complicate interpretation.

## Limitations

- Observational associations do not establish causal effects or intervention benefits.
- Small categories can have unstable churn rates; inspect their counts.
- No dated customer history or explicit prediction horizon is available in this CSV;
  a random split cannot establish future-time performance.
- Measurement timing relative to churn is not established here; verify it before deployment.
- Descriptive results use the full dataset, so later test results are not a fully
  untouched confirmatory analysis. Do not tune models using these comparisons.
- Generalization beyond this dataset requires external validation.
