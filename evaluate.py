"""Final evaluation: nix develop --command uv run python evaluate.py

Uses frozen tuning settings; refuses to overwrite completed final results.
Source: https://github.com/IBM/telco-customer-churn-on-icp4d
"""

import argparse
import hashlib
from html import escape
import json
import math
from pathlib import Path

import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from prepare import CATEGORICAL, NUMERIC
from train import load_training, model_specs


def validate_results(source=Path("data/Telco-Customer-Churn.csv"), preparation=Path("outputs/modelling")):
    """Audit saved final results without fitting models or changing files."""
    output = preparation / "final"
    frozen = json.loads((output / "frozen_settings.json").read_text())
    selected = frozen["selected_settings"]
    selected_path = preparation / "tuning/selected_settings.json"
    if hashlib.sha256(selected_path.read_bytes()).hexdigest() != frozen["selected_settings_sha256"]:
        raise ValueError("Selected settings changed after final evaluation")
    for filename, expected in selected["input_sha256"].items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Evaluation input changed: {filename}")
    raw = pd.read_csv(source).set_index("customerID")
    if hashlib.sha256(source.read_bytes()).hexdigest() != json.loads(
        (preparation / "settings.json").read_text()
    )["source_sha256"]:
        raise ValueError("Source dataset changed")
    split = pd.read_csv(preparation / "split_ids.csv")
    if split["customerID"].duplicated().any() or set(split["customerID"]) != set(raw.index):
        raise ValueError("Invalid split IDs")
    if split["Churn"].tolist() != raw.loc[split["customerID"], "Churn"].tolist():
        raise ValueError("Saved split targets do not match source")
    test_ids = split.loc[split["partition"].eq("test"), "customerID"].tolist()
    predictions = pd.read_csv(output / "test_predictions.csv")
    scores = pd.read_csv(output / "test_metrics.csv").set_index("model")
    expected_models = {"majority_baseline", *selected["models"]}
    if set(scores.index) != expected_models or not scores.index.is_unique:
        raise ValueError("Unexpected or duplicate metric models")
    if set(predictions["model"]) != expected_models:
        raise ValueError("Unexpected prediction models")
    matrices = json.loads((output / "confusion_matrices.json").read_text())["models"]
    for name, rows in predictions.groupby("model"):
        if rows["customerID"].tolist() != test_ids:
            raise ValueError(f"Test IDs/order mismatch: {name}")
        actual = raw.loc[test_ids, "Churn"].eq("Yes").astype(int).tolist()
        if rows["actual"].tolist() != actual:
            raise ValueError(f"Test targets mismatch: {name}")
        probabilities = rows["churn_probability"]
        if not probabilities.between(0, 1).all():
            raise ValueError(f"Invalid probabilities: {name}")
        predicted = probabilities.gt(selected["classification_threshold"]).astype(int)
        if rows["prediction"].tolist() != predicted.tolist():
            raise ValueError(f"Threshold predictions mismatch: {name}")
        computed = {
            "roc_auc": roc_auc_score(actual, probabilities),
            "precision": precision_score(actual, predicted, zero_division=0),
            "recall": recall_score(actual, predicted), "f1": f1_score(actual, predicted),
        }
        for metric, value in computed.items():
            if not math.isclose(scores.loc[name, metric], value, rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError(f"Metric mismatch: {name}/{metric}")
        if matrices[name] != confusion_matrix(actual, predicted, labels=[0, 1]).tolist():
            raise ValueError(f"Confusion matrix mismatch: {name}")
    return scores.reset_index()


def bar_chart(table, label, value, title, destination):
    """Small dependency-free SVG; includes negative permutation importances."""
    values = table[value].tolist()
    low, high = min(0, min(values)), max(0.01, max(values))
    width, height = 900, 80 + 30 * len(table)
    position = lambda x: 300 + (x - low) / (high - low) * 470
    zero = position(0)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">',
        f'<title>{escape(title)}</title>',
        '<rect width="100%" height="100%" fill="white"/>',
        '<g font-family="sans-serif" font-size="14" fill="#111">',
        f'<text x="20" y="28">{escape(title)}</text>',
        f'<line x1="{zero}" y1="40" x2="{zero}" y2="{height - 20}" stroke="#aaa"/>',
    ]
    for i, (_, row) in enumerate(table.iterrows()):
        y, endpoint = 55 + i * 30, position(row[value])
        parts.extend([
            f'<text x="20" y="{y + 15}">{escape(str(row[label]))}</text>',
            f'<rect x="{min(zero, endpoint)}" y="{y}" '
            f'width="{abs(endpoint - zero)}" height="20" fill="#2878a0"/>',
            f'<text x="790" y="{y + 15}">{row[value]:.4f}</text>',
        ])
    destination.write_text("\n".join(parts) + "</g></svg>\n")


def coefficient_contrasts(pipeline):
    prep = pipeline["preprocess"]
    coefficients = pipeline["model"].coef_[0]
    rows = [
        {"predictor": name, "comparison": "Increase of one training standard deviation",
         "log_odds_contrast": float(coefficient), "odds_ratio": math.exp(coefficient)}
        for name, coefficient in zip(NUMERIC, coefficients[:len(NUMERIC)])
    ]
    offset = len(NUMERIC)
    for name, categories in zip(CATEGORICAL, prep.named_transformers_["categorical"].categories_):
        reference = coefficients[offset]
        for i, category in enumerate(categories[1:], start=1):
            contrast = float(coefficients[offset + i] - reference)
            rows.append({
                "predictor": name, "comparison": f"{category} versus {categories[0]}",
                "log_odds_contrast": contrast, "odds_ratio": math.exp(contrast),
            })
        offset += len(categories)
    return pd.DataFrame(rows)


def evaluate(source=Path("data/Telco-Customer-Churn.csv"), preparation=Path("outputs/modelling")):
    output = preparation / "final"
    if (output / "test_metrics.csv").exists():
        raise FileExistsError("Final evaluation already exists; inspect saved results, do not retune.")
    selected_path = preparation / "tuning/selected_settings.json"
    selected = json.loads(selected_path.read_text())
    for filename, expected in selected["input_sha256"].items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Tuning input changed: {filename}")
    current_versions = {"pandas": pd.__version__, "scikit_learn": sklearn.__version__}
    if selected["versions"] != current_versions:
        raise ValueError("Library versions differ from tuning; use the locked environment")
    if selected["classification_threshold"] != 0.5:
        raise ValueError("Expected the preselected threshold 0.5")
    X_train, y_train, _, _, settings = load_training(source, preparation)
    specs = model_specs(settings["seed"])
    models = {"majority_baseline": DummyClassifier(strategy="most_frequent")}
    for name, (pipeline, _) in specs.items():
        saved = selected["models"][name]
        pipeline["model"].set_params(**saved["estimator_defaults"])
        pipeline.set_params(**saved["best_parameters"])
        models[name] = pipeline

    # Freeze before fitting or inspecting test performance.
    output.mkdir(parents=True, exist_ok=True)
    frozen = {
        "selected_settings": selected,
        "selected_settings_sha256": hashlib.sha256(selected_path.read_bytes()).hexdigest(),
        "threshold_rule": "Churn probability > 0.5; exact ties predict No",
        "permutation_importance": {"scoring": "roc_auc", "n_repeats": 10, "seed": settings["seed"]},
        "code_sha256": {
            name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
            for name in ["evaluate.py", "train.py", "prepare.py", "overview.py"]
        },
    }
    frozen_path = output / "frozen_settings.json"
    if frozen_path.exists() and json.loads(frozen_path.read_text()) != frozen:
        raise ValueError("An earlier evaluation started with different settings")
    frozen_path.write_text(json.dumps(frozen, indent=2) + "\n")
    for model in models.values():
        model.fit(X_train, y_train)

    split = pd.read_csv(preparation / "split_ids.csv")
    df = pd.read_csv(source).set_index("customerID")
    test = df.loc[split.loc[split["partition"].eq("test"), "customerID"]].copy()
    test["TotalCharges"] = pd.to_numeric(
        test["TotalCharges"].astype("string").str.strip().replace("", pd.NA), errors="raise",
    )
    X_test, y_test = test[NUMERIC + CATEGORICAL], test["Churn"].eq("Yes").astype(int)
    scores, predictions, matrices = [], [], {}
    for name, model in models.items():
        probability = model.predict_proba(X_test)[:, list(model.classes_).index(1)]
        predicted = (probability > selected["classification_threshold"]).astype(int)
        scores.append({
            "model": name, "roc_auc": roc_auc_score(y_test, probability),
            "precision": precision_score(y_test, predicted, zero_division=0),
            "recall": recall_score(y_test, predicted), "f1": f1_score(y_test, predicted),
        })
        matrices[name] = confusion_matrix(y_test, predicted, labels=[0, 1]).tolist()
        predictions.append(pd.DataFrame({
            "model": name, "customerID": test.index, "actual": y_test.to_numpy(),
            "prediction": predicted, "churn_probability": probability,
        }))
    scores = pd.DataFrame(scores)
    pd.concat(predictions).to_csv(output / "test_predictions.csv", index=False)
    (output / "confusion_matrices.json").write_text(json.dumps({
        "order": "Actual rows / predicted columns; labels [No, Yes]", "models": matrices,
    }, indent=2) + "\n")

    contrasts = coefficient_contrasts(models["logistic_regression"])
    contrasts.to_csv(output / "logistic_contrasts.csv", index=False)
    rf_importance = permutation_importance(
        models["random_forest"], X_test, y_test, scoring="roc_auc",
        n_repeats=10, random_state=settings["seed"], n_jobs=1,
    )
    importance = pd.DataFrame({
        "predictor": X_test.columns, "mean_auc_drop": rf_importance.importances_mean,
        "std_auc_drop": rf_importance.importances_std,
    }).sort_values("mean_auc_drop", ascending=False)
    importance.to_csv(output / "random_forest_permutation_importance.csv", index=False)
    pd.DataFrame(rf_importance.importances, index=X_test.columns).to_csv(
        output / "permutation_repeats.csv", index_label="predictor",
    )
    for metric in ["roc_auc", "precision", "recall", "f1"]:
        bar_chart(scores, "model", metric, f"Final test {metric} (0–1)", output / f"{metric}.svg")
    bar_chart(importance, "predictor", "mean_auc_drop", "RF permutation importance: mean test ROC-AUC drop",
              output / "importance.svg")

    score_rows = "\n".join(
        f"| {row.model} | {row.roc_auc:.4f} | {row.precision:.4f} | {row.recall:.4f} | {row.f1:.4f} |"
        for row in scores.itertuples()
    )
    confusion_tables = "\n\n".join(
        f"### {name}\n\n| Actual / predicted | No | Yes |\n|---|---:|---:|\n"
        f"| No | {matrix[0][0]} | {matrix[0][1]} |\n"
        f"| Yes | {matrix[1][0]} | {matrix[1][1]} |"
        for name, matrix in matrices.items()
    )
    top = "\n".join(
        f"- `{row.predictor}`: mean ROC-AUC drop {row.mean_auc_drop:.4f}, repeat SD {row.std_auc_drop:.4f}."
        for row in importance.head(5).itertuples()
    )
    contrast_rows = "\n".join(
        f"| {row.predictor} | {row.comparison} | {row.odds_ratio:.3f} |"
        for row in contrasts.itertuples()
    )
    report = f"""# Final evaluation and retention hypotheses

Source: [IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d).
Execution order: `overview.py` → `prepare.py` → `train.py` → `evaluate.py`.
Run each from the repository root with `nix develop --command uv run python <script>`.
Tests: `nix develop --command uv run python -m unittest test_train.py test_evaluate.py`.

Settings and threshold were frozen before evaluation. All models used the same
{len(X_train):,} training and {len(X_test):,} test customers. Churn = Yes is positive.
No retuning or threshold selection used test performance. Completed evaluations
are not overwritten; reproduction should use a separate output directory via
`evaluate(source, preparation)` after recreating preparation/tuning there.

## Test performance

| Model | ROC-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
{score_rows}

Threshold: probability > 0.5; exact ties predict No. Undefined precision is zero.
ROC-AUC uses probabilities, not predicted classes. Single-split scores are not
proof of a statistically significant difference or future deployment performance.

![ROC-AUC comparison](roc_auc.svg)
![Precision comparison](precision.svg)
![Recall comparison](recall.svg)
![F1 comparison](f1.svg)

## Confusion matrices

{confusion_tables}

## Which predictors contribute?

Random Forest permutation importance measures the change in final test ROC-AUC
when one original predictor is shuffled (10 repeats). Top five:

{top}

![Permutation importance](importance.svg)

Repeat SD describes shuffle variability, not a confidence interval.
Correlated or redundant predictors can share or mask importance. Negative values
do not establish a protective association. These are post-hoc interpretations,
not a new feature-selection step.

### Logistic Regression conditional associations

| Predictor | Comparison | Odds ratio |
|---|---|---:|
{contrast_rows}

All categorical levels were one-hot encoded. Accordingly, category comparisons
use coefficient differences against the first training category, not the
individual dummy coefficient. Numerical odds ratios correspond to one training
standard deviation after imputation/scaling, not a one-unit raw change.
These are regularized, model-dependent conditional associations. Related and
structurally dependent service categories limit "holding everything else fixed"
interpretations. No causal effects or statistical significance are established.

## Retention hypotheses to test

- Earlier descriptive results suggest prioritizing an onboarding-support pilot
  for newer customers, rather than assuming longer tenure itself prevents churn.
- Investigate month-to-month customers' flexibility and value needs; test voluntary
  offers rather than assuming longer contracts cause retention.
- Test service/support outreach where useful, not blanket add-on sales.
- Use predicted risk to prioritize a controlled retention experiment, not to
  assume a high-risk customer will benefit from an intervention. Choose operating
  thresholds later using training validation, contact capacity and intervention
  costs—not by optimizing on this test set.
- Do not target or disadvantage customers solely on demographics.

## Limitations

- Earlier descriptive exploration included the full dataset, including eventual
  test customers: this is not a fully untouched confirmatory holdout.
- One observational dataset and a random split cannot establish causal effects,
  future-time performance or generalization to another company.
- Services record subscriptions, not usage intensity. Measurement timing relative
  to churn and the prediction horizon need verification.
- All 19 candidates were retained; correlated predictors complicate interpretation.
- No confidence intervals, fairness audit or intervention-benefit validation
  is provided. Recommendations are hypotheses, not proven retention actions.

Settings, source/code hashes, predictions, metrics, contrasts and importance
repeats are saved here for review and later KNIME comparison.
"""
    (output / "evaluation.md").write_text(report)
    # Completion marker last: a failed chart/interpretation step must remain retryable.
    scores.to_csv(output / "test_metrics.csv", index=False)
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="Audit saved results; no fitting or writes")
    parser.add_argument("--preparation", type=Path, default=Path("outputs/modelling"))
    args = parser.parse_args()
    if args.check_only:
        validate_results(preparation=args.preparation)
        print("Saved final results ✓")
    else:
        evaluate(preparation=args.preparation)
        print(f"Final evaluation → {args.preparation / 'final/evaluation.md'}")
