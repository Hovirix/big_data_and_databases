"""Dataset overview only (no model fitting).

Run from the repository root:
    nix develop --command uv run python -m src.overview
"""

import json
from pathlib import Path

import pandas as pd


# Dataset: https://github.com/IBM/telco-customer-churn-on-icp4d
FIELDS = [
    ("customerID", "Identifier", "Identifier", "Customer identifier; exclude from predictors"),
    ("gender", "Demographics", "Categorical", "Recorded gender"),
    ("SeniorCitizen", "Demographics", "Categorical", "Senior citizen indicator: 0 = No, 1 = Yes"),
    ("Partner", "Demographics", "Categorical", "Whether the customer has a partner"),
    ("Dependents", "Demographics", "Categorical", "Whether the customer has dependents"),
    ("tenure", "Subscription", "Numeric", "Months with the company"),
    ("PhoneService", "Services", "Categorical", "Phone service subscription"),
    ("MultipleLines", "Services", "Categorical", "Multiple phone lines"),
    ("InternetService", "Services", "Categorical", "Internet service type"),
    ("OnlineSecurity", "Services", "Categorical", "Online security subscription"),
    ("OnlineBackup", "Services", "Categorical", "Online backup subscription"),
    ("DeviceProtection", "Services", "Categorical", "Device protection subscription"),
    ("TechSupport", "Services", "Categorical", "Technical support subscription"),
    ("StreamingTV", "Services", "Categorical", "Streaming TV subscription"),
    ("StreamingMovies", "Services", "Categorical", "Streaming movies subscription"),
    ("Contract", "Subscription", "Categorical", "Contract duration"),
    ("PaperlessBilling", "Subscription", "Categorical", "Paperless billing indicator"),
    ("PaymentMethod", "Subscription", "Categorical", "Payment method"),
    ("MonthlyCharges", "Subscription", "Numeric", "Monthly charge, dataset currency units"),
    ("TotalCharges", "Subscription", "Numeric", "Cumulative charges, dataset currency units"),
    ("Churn", "Target", "Categorical", "Whether the customer left: Yes = positive class"),
]
NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]


def churn_summary(df, name):
    grouped = (
        df.assign(churned=df["Churn"].eq("Yes"))
        .groupby(name, dropna=False, observed=True)["churned"]
        .agg(customers="size", churned="sum")
        .reset_index()
        .rename(columns={name: "category"})
    )
    grouped.insert(0, "column", name)
    grouped["customer_percentage"] = grouped["customers"] / len(df) * 100
    grouped["churn_percentage"] = grouped["churned"] / grouped["customers"] * 100
    grouped["difference_pp"] = grouped["churn_percentage"] - df["Churn"].eq("Yes").mean() * 100
    return grouped


def create_overview(source=Path("data/Telco-Customer-Churn.csv"), output=Path("outputs")):
    raw = pd.read_csv(source)
    if set(raw.columns) != {field[0] for field in FIELDS}:
        raise ValueError("Unexpected dataset columns")
    if raw.empty or raw["Churn"].isna().any() or not raw["Churn"].isin(["Yes", "No"]).all():
        raise ValueError("Expected nonempty data with Churn = Yes/No")

    df = raw.copy()
    charges = raw["TotalCharges"].astype("string")
    blanks = charges.str.strip().eq("").fillna(False)
    df["TotalCharges"] = pd.to_numeric(charges, errors="coerce")
    invalid = charges.notna() & ~blanks & df["TotalCharges"].isna()
    if invalid.any():
        raise ValueError("Unexpected non-numeric TotalCharges values")

    quality = {
        "rows": len(df),
        "columns": len(df.columns),
        "duplicate_rows": int(raw.duplicated().sum()),
        "duplicate_customer_ids": int(raw["customerID"].duplicated().sum()),
        "missing_before_conversion": raw.isna().sum().to_dict(),
        "missing_after_conversion": df.isna().sum().to_dict(),
        "blank_total_charges": int(blanks.sum()),
        "blank_total_charges_with_zero_tenure": int((blanks & df["tenure"].eq(0)).sum()),
        "negative_numeric_values": df[NUMERIC].lt(0).sum().to_dict(),
        "invalid_senior_citizen_values": int((~df["SeniorCitizen"].isin([0, 1])).sum()),
        "total_charges_policy": "Convert blanks to missing; no imputation or row removal during overview.",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "quality.json").write_text(json.dumps(quality, indent=2) + "\n")

    dictionary = pd.DataFrame(FIELDS, columns=["column", "group", "type", "description"])
    dictionary["observed_dtype"] = [str(df[name].dtype) for name in dictionary["column"]]
    dictionary["values_or_range"] = [
        f"{df[name].min()} to {df[name].max()}" if kind == "Numeric"
        else "Unique customer identifiers" if kind == "Identifier"
        else " | ".join(sorted(df[name].dropna().astype(str).unique()))
        for name, _, kind, _ in FIELDS
    ]
    dictionary.to_csv(output / "data_dictionary.csv", index=False)
    df[NUMERIC].describe().T.to_csv(output / "numeric_summary.csv", index_label="column")
    df.groupby("Churn")[NUMERIC].describe().to_csv(output / "numeric_by_churn.csv")
    churn = df["Churn"].value_counts().reindex(["No", "Yes"], fill_value=0).rename("customers").to_frame()
    churn["percentage"] = churn["customers"] / len(df) * 100
    churn.to_csv(output / "churn_distribution.csv", index_label="Churn")

    comparisons = []
    for name, _, kind, _ in FIELDS:
        if kind != "Categorical" or name == "Churn":
            continue
        comparisons.append(churn_summary(df, name))
    categorical = pd.concat(comparisons, ignore_index=True)
    categorical.to_csv(output / "categorical_churn.csv", index=False)

    # Descriptive bins only; do not reuse full-dataset quartiles in model preprocessing.
    tenure_edges = [-1, 12, 24, 48, 72]
    tenure_labels = ["0–12", "13–24", "25–48", "49–72"]
    binned = df.copy()
    binned["Tenure band (months)"] = pd.cut(
        df["tenure"], bins=tenure_edges, labels=tenure_labels
    )
    binned["Monthly charge quartile"], charge_edges = pd.qcut(
        df["MonthlyCharges"], 4, labels=["Q1", "Q2", "Q3", "Q4"], retbins=True
    )
    if binned["Tenure band (months)"].isna().any():
        raise ValueError("Tenure must be present and within 0–72 months")
    groups = pd.concat(
        [categorical, churn_summary(binned, "Tenure band (months)"),
         churn_summary(binned, "Monthly charge quartile")],
        ignore_index=True,
    )
    groups.to_csv(output / "group_churn.csv", index=False)
    settings = {
        "tenure": {"edges": tenure_edges, "labels": tenure_labels, "right_closed": True},
        "monthly_charges": {
            "edges": charge_edges.tolist(), "labels": ["Q1", "Q2", "Q3", "Q4"],
            "right_closed": True, "include_lowest": True,
            "scope": "Full-dataset descriptive quartiles, not model preprocessing",
        },
    }
    (output / "group_settings.json").write_text(json.dumps(settings, indent=2) + "\n")

    sections = []
    ordered = ["Contract", "Tenure band (months)", "Monthly charge quartile"] + [
        name for name, group, _, _ in FIELDS if group in ["Demographics", "Services"]
    ]
    for name in ordered:
        table = groups[groups["column"].eq(name)]
        rows = "\n".join(
            f"| {row.category} | {row.customers:,} | {row.churned:,} | "
            f"{row.churn_percentage:.2f}% | {row.difference_pp:+.2f} |"
            for row in table.itertuples()
        )
        highest = table.loc[table["churn_percentage"].idxmax()]
        lowest = table.loc[table["churn_percentage"].idxmin()]
        sections.append(
            f"## {name}\n\n"
            "| Group | Customers | Churners | Churn rate | Difference (pp) |\n"
            "|---|---:|---:|---:|---:|\n" + rows
            + f"\n\nHighest observed rate: {highest['category']} "
            f"({highest['churn_percentage']:.2f}%); lowest: {lowest['category']} "
            f"({lowest['churn_percentage']:.2f}%).\n"
        )
    charge_ranges = "\n".join(
        f"- Q{i + 1}: {'≥' if i == 0 else '>'}{charge_edges[i]:.2f} "
        f"and ≤{charge_edges[i + 1]:.2f}."
        for i in range(4)
    )
    patterns = f"""# Churn across customer groups

Source: [IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d).
Reproduce from the repository root: `nix develop --command uv run python -m src.overview`.

Overall churn: **{churn.loc['Yes', 'percentage']:.2f}%**.
Churn rate = churners / customers within each group × 100.
Difference (pp) = group churn rate minus overall churn, in percentage points.
Counts show the size behind each percentage; rates are not shares of all churners.

Tenure bands are inclusive integer months. SeniorCitizen: 0 = No, 1 = Yes.
Monthly charges use approximately equal-sized quartile groups:

{charge_ranges}

Exact bin boundaries and closure rules are saved in `group_settings.json`.
Full results, including other billing categories, are in `group_churn.csv`.

""" + "\n".join(sections) + """
## Interpretation and limitations

- Inspect how contract and tenure rates differ before comparing demographics and services.
- Charge quartiles allow nonlinear patterns; do not assume churn rises with every quartile.
- All comparisons are unadjusted associations, not causal effects or model importance rankings.
- Contracts, tenure, charges and services overlap; models will assess their joint predictive value.
- "No internet service" and "No phone service" remain distinct from not buying an add-on.
  The repeated no-internet group is the same set of customers, not independent evidence.
- Subscriptions are recorded, not actual usage intensity.
- Small groups can have unstable rates; no confidence intervals or significance tests are computed.
- These full-dataset descriptive bins are not learned preprocessing for later models.
  Do not tune models using test-set patterns. No models have been fitted here.
"""
    (output / "churn_patterns.md").write_text(patterns)

    contracts = categorical[categorical["column"].eq("Contract")]
    contract_table = "\n".join(
        f"| {row.category} | {row.customers} | {row.churn_percentage:.2f}% |"
        for row in contracts.itertuples()
    )
    predictor_table = "\n".join(
        f"| {label} | "
        + ", ".join(f"`{name}`" for name in dictionary.loc[dictionary["group"].eq(group), "column"])
        + " |"
        for group, label in [
            ("Demographics", "Demographics"),
            ("Subscription", "Subscription characteristics (including billing)"),
            ("Services", "Subscribed services"),
        ]
    )
    report = f"""# Dataset overview

Source: [IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d).
One row represents one customer. These are descriptive results, not model evaluations.

## Reproduce

From the repository root, using the pandas dependency in `pyproject.toml`:

```sh
nix develop --command uv run python -m src.overview
nix develop --command uv run python -m unittest tests.test_overview
```

The source CSV is read only. Outputs in `outputs/` are regenerated.

## Dataset and quality

- {len(df):,} customers, {len(df.columns)} columns.
- Churn = Yes (customer left, positive class): {churn.loc['Yes', 'customers']:,} ({churn.loc['Yes', 'percentage']:.2f}%).
- Churn = No (customer did not leave): {churn.loc['No', 'customers']:,} ({churn.loc['No', 'percentage']:.2f}%).
- Duplicate rows: {quality['duplicate_rows']}; duplicate customer IDs: {quality['duplicate_customer_ids']}.
- TotalCharges blanks: {quality['blank_total_charges']}; of these,
  {quality['blank_total_charges_with_zero_tenure']} have tenure = 0.
- Blanks become missing numeric values; no rows are removed and no values imputed.
- TotalCharges is loaded as text because of blank entries and converted to numeric.
- SeniorCitizen is categorical despite numeric storage.
- "No internet service" and "No phone service" are valid categories, not missing values.

## Predictors

The {len(df.columns) - 2} candidate predictors fall into three groups:

| Group | Variables |
|---|---|
{predictor_table}

`customerID` is an identifier, not a predictor; `Churn` is the target.

## Example association: contract and churn

| Contract | Customers | Churn rate |
|---|---:|---:|
{contract_table}

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
{churn.loc['No', 'percentage']:.2f}% accuracy here but zero churn recall.

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
"""
    (output / "overview.md").write_text(report)
    return quality


if __name__ == "__main__":
    create_overview()
    print("Dataset overview → outputs/overview.md")
