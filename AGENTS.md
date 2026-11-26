# Predicting Customer Churn in Telecommunications

##  context
This is project using IBM's Telco Customer Churn dataset. The objective is to predict customer churn, compare Logistic Regression and Random Forest, and translate findings into practical retention recommendations.

We are developing the complete analysis in Python first. It will subsequently be ported to KNIME, which is required for the final course implementation. Keep the Python solution straightforward and easy to reproduce in KNIME.

## Research Question
To what extent can customer demographics, subscription characteristics, and service usage predict customer churn, and which factors are most strongly associated with churn?

## Existing reference
https://github.com/IBM/telco-customer-churn-on-icp4d

The repository provides:
- `data/Telco-Customer-Churn.csv`
- A Python/Spark example covering preparation, exploration and Random Forest.
- An example notebook with saved outputs.

Reuse the dataset and relevant ideas with attribution. Do not reproduce IBM Cloud, Watson deployment or the Flask application. The repository is archived; its old environment is not our target stack.

## Technical expectations
- Use pandas and scikit-learn.
- Preserve the original CSV; use relative paths.
- Retain `customerID` for tracing records and reproducing splits, never as a predictor.
- Include `tenure` among candidate predictors.
- Convert `TotalCharges` to numeric and explicitly handle blank values.
- Treat `Churn = Yes` as the positive class.
- Use a fixed-seed, stratified 80/20 train/test split shared by all models.
- Fit learned preprocessing only on training data, including within validation folds.
- Use appropriate categorical encoding and scaling for Logistic Regression.
- Tune on training data only; reserve the test set for final evaluation.
- Compare against a majority-class baseline.
- Report ROC-AUC using probabilities, precision, recall, F1 and confusion matrices.
- Save split IDs, settings and outputs to support the later KNIME comparison.
- Interpret predictors as associations, not causal effects.

## Team and deliverables
Work is divided into Code, Report and Review. Produce clear outputs that the report writer and reviewer can use independently.

The final package includes the dataset, PDF report and supporting files. A portable KNIME workflow export is recommended supporting material.

The report covers the objective, dataset and descriptive statistics, preparation and feature selection, both algorithms and evaluation, limitations, and managerial implications.

## Working conventions
Inspect existing files before editing. Prefer a small, reproducible implementation over unnecessary infrastructure. Document dependencies and execution commands. Generate findings and charts from actual runs; never fabricate results or claim unperformed checks passed.

Cite adapted material and preserve applicable license notices. Follow course-specific AI and reuse rules; do not assume permission where unclear. Do not submit anything to Blackboard.
