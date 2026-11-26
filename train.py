"""Training-only tuning: nix develop --command uv run python train.py"""

import hashlib
import json
from pathlib import Path

import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from prepare import CATEGORICAL, NUMERIC, make_preprocessor


def load_training(source, preparation):
    settings = json.loads((preparation / "settings.json").read_text())
    if hashlib.sha256(source.read_bytes()).hexdigest() != settings["source_sha256"]:
        raise ValueError("Dataset changed; regenerate preparation before tuning")
    df = pd.read_csv(source).set_index("customerID")
    split = pd.read_csv(preparation / "split_ids.csv")
    if (
        split["customerID"].duplicated().any()
        or set(split["customerID"]) != set(df.index)
        or not split["partition"].isin(["train", "test"]).all()
        or split["partition"].value_counts().to_dict()
        != {"train": settings["train_customers"], "test": settings["test_customers"]}
    ):
        raise ValueError("Invalid saved split")
    if split["Churn"].tolist() != df.loc[split["customerID"], "Churn"].tolist():
        raise ValueError("Saved split targets do not match source data")
    # Select training rows before converting predictors or constructing targets.
    train = df.loc[split.loc[split["partition"].eq("train"), "customerID"]].copy()
    train["TotalCharges"] = pd.to_numeric(
        train["TotalCharges"].astype("string").str.strip().replace("", pd.NA),
        errors="raise",
    )
    y = train["Churn"].eq("Yes").astype(int)
    cv_settings = settings["cv"]
    cv = StratifiedKFold(
        n_splits=cv_settings["folds"],
        shuffle=cv_settings["shuffle"],
        random_state=cv_settings["seed"],
    )
    folds = list(cv.split(train, y))
    baseline = pd.read_csv(preparation / "baseline_cv_predictions.csv")
    if baseline["customerID"].tolist() != train.index.tolist():
        raise ValueError("Baseline training IDs/order do not match saved split")
    if baseline["actual"].tolist() != y.tolist():
        raise ValueError("Baseline targets do not match training data")
    for fold, (_, validation) in enumerate(folds, start=1):
        if not baseline.iloc[validation]["fold"].eq(fold).all():
            raise ValueError("Baseline validation folds do not match settings")
    return train[NUMERIC + CATEGORICAL], y, folds, baseline, settings


def model_specs(seed):
    return {
        "logistic_regression": (
            Pipeline([
                ("preprocess", make_preprocessor()),
                ("model", LogisticRegression(max_iter=2000, random_state=seed)),
            ]),
            {"model__C": [0.1, 1, 10]},
        ),
        "random_forest": (
            Pipeline([
                ("preprocess", make_preprocessor(scale_numeric=False)),
                ("model", RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=1)),
            ]),
            {"model__max_depth": [None, 10], "model__min_samples_leaf": [1, 5]},
        ),
    }


def tune(source=Path("data/Telco-Customer-Churn.csv"), preparation=Path("outputs/modelling")):
    X, y, folds, baseline, preparation_settings = load_training(source, preparation)
    output = preparation / "tuning"
    output.mkdir(parents=True, exist_ok=True)
    scoring = {
        "roc_auc": "roc_auc",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": "recall",
        "f1": "f1",
    }
    baseline_scores = []
    for _, validation in folds:
        rows = baseline.iloc[validation]
        baseline_scores.append({
            "roc_auc": roc_auc_score(rows["actual"], rows["churn_probability"]),
            "precision": precision_score(rows["actual"], rows["prediction"], zero_division=0),
            "recall": recall_score(rows["actual"], rows["prediction"]),
            "f1": f1_score(rows["actual"], rows["prediction"]),
        })
    baseline_scores = pd.DataFrame(baseline_scores)
    comparison = [{
        "model": "majority_baseline",
        **{f"mean_cv_{metric}": float(baseline_scores[metric].mean()) for metric in scoring},
        "std_cv_roc_auc": float(baseline_scores["roc_auc"].std(ddof=0)),
    }]
    selected = {}
    for name, (pipeline, grid) in model_specs(preparation_settings["seed"]).items():
        search = GridSearchCV(
            pipeline, grid, scoring=scoring, refit=False, cv=folds,
            n_jobs=2, error_score="raise", return_train_score=False,
        ).fit(X, y)
        results = pd.DataFrame(search.cv_results_)
        results.to_csv(output / f"{name}_cv_results.csv", index=False)
        # First maximum breaks ties in the declared grid order.
        best = results["mean_test_roc_auc"].idxmax()
        params = search.cv_results_["params"][best]
        selected[name] = {
            "best_parameters": params,
            "parameter_grid": grid,
            "estimator_defaults": pipeline["model"].get_params(),
            "scale_numeric": name == "logistic_regression",
        }
        comparison.append({
            "model": name,
            **{
                f"mean_cv_{metric}": float(results.loc[best, f"mean_test_{metric}"])
                for metric in scoring
            },
            "std_cv_roc_auc": float(results.loc[best, "std_test_roc_auc"]),
        })
    comparison = pd.DataFrame(comparison)
    comparison.to_csv(output / "cv_comparison.csv", index=False)
    settings = {
        "scope": "Training-only selection; no test evaluation or final full-training refit",
        "selection_metric": "mean validation ROC-AUC",
        "classification_threshold": 0.5,
        "cv": preparation_settings["cv"],
        "seed": preparation_settings["seed"],
        "versions": {"pandas": pd.__version__, "scikit_learn": sklearn.__version__},
        "input_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [source, preparation / "split_ids.csv", preparation / "settings.json"]
        },
        "models": selected,
    }
    (output / "selected_settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    rows = "\n".join(
        f"| {row.model} | {row.mean_cv_roc_auc:.4f} | {row.std_cv_roc_auc:.4f} | "
        f"{row.mean_cv_precision:.4f} | {row.mean_cv_recall:.4f} | {row.mean_cv_f1:.4f} |"
        for row in comparison.itertuples()
    )
    parameters = "\n".join(
        f"- {name}: `{values['best_parameters']}`." for name, values in selected.items()
    )
    report = f"""# Training-only model comparison

Reproduce after `prepare.py`: `nix develop --command uv run python train.py`.
Checks: `nix develop --command uv run python -m unittest test_train.py`.
Source: [IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d).

All models use the saved {len(X):,} training customers and the baseline's exact
five stratified validation folds. Imputation, encoding and scaling are fitted
separately inside each fold. Logistic Regression scales numerical variables;
Random Forest does not. All 19 candidate predictors are retained.

## Cross-validation results

| Model | Mean ROC-AUC | ROC-AUC SD | Mean precision | Mean recall | Mean F1 |
|---|---:|---:|---:|---:|---:|
{rows}

These are means of fold scores, not pooled out-of-fold scores. SD describes
variation across folds, not a confidence interval. Precision, recall and F1
use the estimator's default binary decision rule (threshold 0.5; exact ties
predict No). Undefined baseline precision is reported as zero.

## Selected settings

Selection criterion: highest mean validation ROC-AUC.
{parameters}

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
"""
    (output / "comparison.md").write_text(report)
    return comparison


if __name__ == "__main__":
    tune()
    print("Training CV comparison → outputs/modelling/tuning/comparison.md")
