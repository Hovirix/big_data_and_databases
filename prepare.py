"""Run: nix develop --command uv run python prepare.py

Source: https://github.com/IBM/telco-customer-churn-on-icp4d
Split, preprocessing and training-only baseline; no final test evaluation.
"""

import hashlib
import json
from pathlib import Path

import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from overview import FIELDS, NUMERIC


SEED = 42
CATEGORICAL = [
    name for name, _, kind, _ in FIELDS if kind == "Categorical" and name != "Churn"
]


def make_preprocessor(scale_numeric=True):
    """Return unfitted preprocessing; put inside each model pipeline before CV."""
    numeric = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("numeric", Pipeline(numeric), NUMERIC),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ], remainder="drop")


def prepare(source=Path("data/Telco-Customer-Churn.csv"), output=Path("outputs/modelling")):
    df = pd.read_csv(source)
    if set(df.columns) != {field[0] for field in FIELDS}:
        raise ValueError("Unexpected dataset columns")
    if df["customerID"].isna().any() or df["customerID"].duplicated().any():
        raise ValueError("Customer IDs must be present and unique")
    if df["Churn"].isna().any() or not df["Churn"].isin(["Yes", "No"]).all():
        raise ValueError("Expected Churn = Yes/No")
    charges = df["TotalCharges"].astype("string")
    df["TotalCharges"] = pd.to_numeric(charges.str.strip().replace("", pd.NA), errors="raise")
    X = df[NUMERIC + CATEGORICAL]
    if X[CATEGORICAL].isna().any().any():
        raise ValueError("Unexpected missing categorical values")
    y = df["Churn"].eq("Yes").astype(int)
    train, test = train_test_split(
        df.index, test_size=0.2, stratify=y, random_state=SEED,
    )
    X_train, y_train = X.loc[train], y.loc[train]
    output.mkdir(parents=True, exist_ok=True)
    pd.concat([
        df.loc[train, ["customerID", "Churn"]].assign(partition="train"),
        df.loc[test, ["customerID", "Churn"]].assign(partition="test"),
    ]).to_csv(output / "split_ids.csv", index=False)

    # Fit only on training data for inspectable KNIME settings, not for reuse in CV.
    preprocessor = make_preprocessor().fit(X_train)
    numeric = preprocessor.named_transformers_["numeric"]
    encoder = preprocessor.named_transformers_["categorical"]
    settings = {
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "versions": {"pandas": pd.__version__, "scikit_learn": sklearn.__version__},
        "seed": SEED, "test_size": 0.2, "stratified": True,
        "positive_class": "Yes", "target_mapping": {"No": 0, "Yes": 1},
        "train_customers": len(train), "test_customers": len(test),
        "numeric_predictors": NUMERIC, "categorical_predictors": CATEGORICAL,
        "total_charges_blanks": "Missing; training-median imputation",
        "categorical_encoding": "One-hot, no dropped category, ignore unknown categories",
        "scaling": "StandardScaler for Logistic Regression; scale_numeric=False for Random Forest",
        "cv": {"folds": 5, "shuffle": True, "seed": SEED, "scope": "training only"},
        "baseline": "DummyClassifier(strategy='most_frequent')",
        "training_preprocessing": {
            "medians": dict(zip(NUMERIC, numeric["impute"].statistics_.tolist())),
            "means": dict(zip(NUMERIC, numeric["scale"].mean_.tolist())),
            "scales": dict(zip(NUMERIC, numeric["scale"].scale_.tolist())),
            "categories": {
                name: values.tolist() for name, values in zip(CATEGORICAL, encoder.categories_)
            },
        },
    }
    (output / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    pd.Series(preprocessor.get_feature_names_out(), name="feature").to_csv(
        output / "encoded_features.csv", index=False,
    )

    # Each fold learns its own preprocessing; test data is never transformed or scored.
    baseline = Pipeline([
        ("preprocess", make_preprocessor()),
        ("model", DummyClassifier(strategy="most_frequent")),
    ])
    predictions = pd.DataFrame({
        "customerID": df.loc[train, "customerID"].to_numpy(),
        "actual": y_train.to_numpy(),
    })
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (fit, validation) in enumerate(cv.split(X_train, y_train), start=1):
        model = clone(baseline).fit(X_train.iloc[fit], y_train.iloc[fit])
        predictions.loc[validation, "fold"] = fold
        predictions.loc[validation, "prediction"] = model.predict(X_train.iloc[validation])
        predictions.loc[validation, "churn_probability"] = model.predict_proba(
            X_train.iloc[validation]
        )[:, 1]
    predictions[["fold", "prediction"]] = predictions[["fold", "prediction"]].astype(int)
    predictions.to_csv(output / "baseline_cv_predictions.csv", index=False)
    actual, predicted = predictions["actual"], predictions["prediction"]
    metrics = {
        "scope": "Pooled out-of-fold predictions on training data only",
        "roc_auc": roc_auc_score(actual, predictions["churn_probability"]),
        "precision": precision_score(actual, predicted, zero_division=0),
        "recall": recall_score(actual, predicted, zero_division=0),
        "f1": f1_score(actual, predicted, zero_division=0),
        "confusion_matrix": confusion_matrix(actual, predicted, labels=[0, 1]).tolist(),
        "confusion_matrix_order": "Rows = actual, columns = predicted; labels = [No, Yes]",
    }
    (output / "baseline_cv_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    report = f"""# Preparation and majority baseline

Reproduce: `nix develop --command uv run python prepare.py`
Check: `nix develop --command uv run python -m unittest test_prepare.py`

- Shared stratified 80/20 split, seed {SEED}: {len(train):,} training / {len(test):,} test customers.
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

Pooled out-of-fold ROC-AUC: **{metrics['roc_auc']:.3f}**.
Precision: **{metrics['precision']:.3f}**; recall: **{metrics['recall']:.3f}**;
F1: **{metrics['f1']:.3f}**.

Confusion matrix (actual rows / predicted columns):

| | Predicted No | Predicted Yes |
|---|---:|---:|
| Actual No | {metrics['confusion_matrix'][0][0]} | {metrics['confusion_matrix'][0][1]} |
| Actual Yes | {metrics['confusion_matrix'][1][0]} | {metrics['confusion_matrix'][1][1]} |

The majority baseline predicts No for every validation customer. Its positive-class
probability is zero; precision is undefined without positive predictions and reported
as zero (`zero_division=0`). It detects no churners.

The test set has not been transformed or evaluated. Next: compare Logistic Regression
and Random Forest using the same training folds; reserve test evaluation until model
settings are fixed. The earlier full-dataset descriptive exploration remains a limitation.
"""
    (output / "preparation.md").write_text(report)
    return settings, metrics


if __name__ == "__main__":
    prepare()
    print("Preparation + baseline → outputs/modelling/preparation.md")
