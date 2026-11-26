# Telco customer churn — Python analysis

Predict churn using pandas and scikit-learn: majority baseline, Logistic Regression
and Random Forest. Source: [IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d).
The original CSV is preserved. One row represents one customer.

## Environment

Requires Nix with flakes enabled. From the repository root:

```sh
nix develop
uv sync --locked
```

Alternatively, use `direnv allow` with the existing `.envrc`.

## Inspect the completed analysis

For a quick offline visual check of the original data:

```sh
uv run --locked python -m src.view_data
```

Open `outputs/data_view.html` in your browser (no server needed). It includes all
customer records with search, churn/blank-value filters, group churn charts and
the CSV fingerprint compared with modelling. It does not rerun models.

```sh
uv run --locked python -m src.evaluate --check-only
uv run --locked python -m unittest discover
```

The audit recomputes metrics from saved predictions and checks source/input hashes,
test IDs and targets, probability ranges, threshold decisions and confusion matrices.
It does not fit models, rewrite results or prove that the original models were correct.
Tests use temporary directories and reduced tuning grids; they do not overwrite
the completed analysis.

Entry points and outputs:

| Script | Purpose | Saved outputs |
|---|---|---|
| `src/overview.py` | Dictionary, quality checks, descriptive comparisons | `outputs/overview.md`, `outputs/churn_patterns.md`, supporting CSV/JSON |
| `src/prepare.py` | Shared split, preprocessing, training-only baseline | `outputs/modelling/preparation.md`, IDs, settings, baseline predictions |
| `src/train.py` | Training-only five-fold model tuning | `outputs/modelling/tuning/comparison.md`, grid results and selected settings |
| `src/evaluate.py` | Frozen-setting final evaluation and interpretation | `outputs/modelling/final/evaluation.md`, predictions, metrics, SVG charts |

Code lives in `src/`; tests live in `tests/`. Run modules from the repository
root with `uv run --locked python -m src.<module>`. Tests are discovered with
`uv run --locked python -m unittest discover`.

Completed final evaluations refuse to overwrite existing metrics. Do not delete
results merely to try different settings after seeing test performance.

## Reproduce separately

The existing default outputs record the original completed analysis. To reproduce
the modelling steps without overwriting them, use a **new, unused directory**:

```sh
uv run --locked python - <<'PY'
from pathlib import Path
from src.prepare import prepare
from src.train import tune
from src.evaluate import evaluate, validate_results

source = Path("data/Telco-Customer-Churn.csv")
output = Path("outputs/reproduction")
if output.exists():
    raise FileExistsError("Choose a new output directory")
prepare(source, output)
tune(source, output)
evaluate(source, output)
validate_results(source, output)
PY
```

This repeats the same fixed analysis for reproducibility, not a new tuning round.
For descriptive outputs alone, run `uv run --locked python -m src.overview`.
All scripts expect the repository root as working directory.

## Method and interpretation

- Churn = Yes is positive. All 19 candidate predictors are retained; `customerID`
  is only for tracing and split reproduction.
- Fixed-seed (`42`), stratified 80/20 split; saved IDs and their order are shared.
- Blank `TotalCharges` values become missing. Numerical median imputation and
  categorical one-hot encoding are learned inside each training fold.
  `SeniorCitizen` is categorical. Numerical scaling is used only for Logistic Regression.
- Model selection uses mean validation ROC-AUC, never test metrics. Selected settings
  and a probability threshold of `> 0.5` (ties → No) are frozen before final fitting.
- ROC-AUC uses probabilities. Precision, recall, F1 and confusion matrices use
  thresholded predictions; undefined precision is reported as zero.
- Final RF permutation importance is post-hoc interpretation, not feature selection.
  Logistic odds ratios use category coefficient differences or one numeric
  training standard deviation.

Correlated charges, tenure and service subscriptions can make conditional coefficients
differ from unadjusted descriptive patterns. In particular, the fitted negative
`MonthlyCharges` coefficient does **not** mean raising prices prevents churn.
Structurally dependent service categories also limit isolated category comparisons.

The dataset records subscribed services, not usage intensity. Associations are not
causal effects. Earlier descriptive exploration included the eventual test customers;
the test set is not a fully untouched confirmatory holdout. A random split does not
establish future-time performance, fairness or intervention benefit.

Frozen settings record library versions and source/code hashes at evaluation time.
Later audit/documentation changes do not replace that historical record.
No KNIME workflow or course PDF is produced by this Python package.
