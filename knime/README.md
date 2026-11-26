# Native KNIME reproduction specification

## Status

**Specification only; not an imported or executed KNIME workflow.**
No KNIME executable was found on `PATH`, in the system/profile executable
locations, `/opt`, or the searched user installation/download/project locations
on 2026-11-26. No `.knwf` is supplied: an untested hand-written archive would not
establish a working workflow.

Inspected `src/prepare.py`, `src/train.py`, `src/evaluate.py` and the saved
preparation, tuning and final artifacts. The read-only command
`.venv/bin/python -m src.evaluate --check-only` passed. Training-only inspection
also confirmed the saved CV folds and identical category sets in every fit fold.
These are **Python checks**, not KNIME validation.

The mapping below preserves the inputs, partitions, fold membership, feature
representation, positive class and metric definitions. **Native learners are
cross-platform replications, not bit-identical scikit-learn implementations.**
In particular, KNIME's logistic learner cannot select LBFGS or directly consume
scikit-learn's `C`. Its Gaussian-prior conversion is an unresolved equivalence
check, not a claim that the original regularization has been reproduced.

## 1. Dependencies and files

Use KNIME Analytics Platform 5.x. Record the actual version before building;
none was installed here. Node names below refer to the documented native nodes,
not Python, R, H2O or Weka substitutes. Install from the update site matching
the installed platform:

| Extension | Feature ID | Used for |
|---|---|---|
| KNIME Base nodes | `org.knime.features.base.feature.group` | Readers/writers, filters, joins, rules, math, statistics, loops, logistic regression, scoring |
| KNIME Ensemble Learning Wrappers | `org.knime.features.ensembles.feature.group` | Random Forest Learner / Predictor |
| KNIME Views | `org.knime.features.base.views.feature.group` | Modern ROC Curve node |
| KNIME Timeseries nodes (only for AUC fallback) | `org.knime.features.timeseries.feature.group` | Moving Aggregator, cumulative sums |

No Python integration, scripting extension or Parameter Optimization extension
is required. Use **Table Row to Variable Loop Start** for explicit candidate
tables rather than an optimizer that might choose different candidates.
If a named node is hidden as deprecated, enable deprecated nodes in the node
repository; do not silently substitute an implementation with different behavior.

Set one workflow string variable `project_root` to the extracted repository
directory. Resolve the following paths relative to it, not relative to a user's
home. Keep the workflow under `knime/`; keep output under a **new**
`outputs/knime/<run-name>/` directory. CSV Readers use UTF-8, comma delimiter,
double-quote quoting, first row as headers, no row limit.

| Input | Purpose |
|---|---|
| `data/Telco-Customer-Churn.csv` | Unchanged IBM source |
| `outputs/modelling/split_ids.csv` | Exact train/test membership **and order** |
| `outputs/modelling/baseline_cv_predictions.csv` | Exact five-fold assignment by customer ID |
| `outputs/modelling/settings.json` | Training preprocessing reference |
| `outputs/modelling/encoded_features.csv` | Exact 46-feature order |
| `outputs/modelling/tuning/selected_settings.json` | Original grid/defaults/selected settings |
| `outputs/modelling/final/frozen_settings.json` | Historical evaluation provenance |
| `outputs/modelling/final/test_predictions.csv` | Python comparison only, never model input |
| `outputs/modelling/final/test_metrics.csv` | Python comparison only |
| `outputs/modelling/final/confusion_matrices.json` | Python comparison only |

Dataset SHA-256:
`16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`.
Verify before building, e.g. `sha256sum data/Telco-Customer-Churn.csv`.
The original split/settings hashes are in `selected_settings.json`.
The Python versions recorded there are pandas **3.1.1**, scikit-learn **1.9.1**.
Do not overwrite the frozen Python outputs or call `prepare()` again.

## 2. Input and split nodes

IDs below identify nodes/connections in this specification. `table` and `model`
denote port types; connect by type, not an assumed port number. RF's model is its
third output, whereas LR's model is its first output.

| ID | Native node | Input → configuration → output |
|---|---|---|
| D01 | CSV Reader | Source CSV. Explicitly read `TotalCharges` as String; `tenure` as numeric, `MonthlyCharges` as Double, `SeniorCitizen` as String (`"0"`, `"1"`), all remaining fields as String. No automatic numeric conversion of `TotalCharges`. |
| D02 | String Manipulation | D01 → replace `TotalCharges` with `strip($TotalCharges$)`. Do not trim or recode other categories. |
| D03 | Rule Engine | D02 → append `TotalCharges_clean` with the single rule `NOT $TotalCharges$ = "" => $TotalCharges$`. No fallback rule: unmatched empty strings become genuine missing cells, not the literal string `"?"`. |
| D04 | String to Number | D03 → convert only `TotalCharges_clean` to Double, decimal point `.`, fail on non-numeric nonmissing strings. Missing remains missing. |
| D05 | Column Filter + Column Rename | D04 → remove original `TotalCharges`; rename `TotalCharges_clean` to `TotalCharges`. |
| D06 | CSV Reader | `split_ids.csv`; String columns `customerID,Churn,partition`. |
| D07 | Counter Generation | D06 → append integer `split_order`, start 0, increment 1. Generate **before** filtering/joining. |
| D08 | Joiner | Left D07, right D05; inner equality join on `customerID`; retain left `customerID,Churn,partition,split_order`, right 19 predictors and right `Churn` renamed `source_Churn`. No duplicate matching rows permitted. |
| D09 | Rule-based Row Splitter | D08 → `$Churn$ = $source_Churn$ => TRUE`, `TRUE => FALSE`. Mismatch output must have 0 rows; stop if not. |
| D10 | Column Filter | D09 matching output → remove `source_Churn`. |
| D11 | Row Splitter | D10 → include `partition = train`; matching output = train, other output = test (after validating partition domain). |
| D12 | Sorter | D11 train → ascending numeric `split_order`. |
| D13 | Sorter | D11 test → ascending numeric `split_order`. |
| D14 | CSV Reader | `baseline_cv_predictions.csv`; read `customerID` String, `actual,fold` Integer. Exclude `prediction,churn_probability` from join output. |
| D15 | Joiner | Left D12, right D14; inner on `customerID`; append `actual,fold`. |
| D16 | Sorter | D15 → ascending `split_order`. This is the only source table for CV. |

Before passing D08/D16 downstream, inspect **GroupBy** counts and duplicate-ID
counts (group by `customerID`, count; **Row Filter** count > 1 must be empty).
Check unmatched Joiner outputs as well; an inner join alone does not prove coverage.
Use **Rule Engine** to check missing/invalid targets, categories and partitions;
stop on unexpected values, not a default guess.

Required invariants:

* 7,043 unique, nonmissing IDs; 5,634 train + 1,409 test; no overlap.
* Train: No=4,139, Yes=1,495. Test: No=1,035, Yes=374.
* Exactly 11 blank `TotalCharges` in source: 8 train, 3 test.
* `actual = 1` iff `Churn = "Yes"`; no missing categorical predictors.
* Every train ID matches exactly one saved fold and the saved actual label.
* D12/D13 ID sequences equal the corresponding sequences in D06.
* No Partitioning/Table Partitioner or X-Partitioner node: seed 42 in KNIME
  would **not** recreate NumPy's split or folds.

Target remains nominal `"No"` / `"Yes"` for native learners. This is equivalent
to Python's 0/1 target only when probability/metric columns explicitly select Yes.
Keep ID/order/fold/partition as metadata, never as predictors.

## 3. Preprocessing component: FIT + APPLY

Build the following native-node component once conceptually and instantiate it
inside each CV iteration and for final fitting. Inputs are `FIT` and `APPLY`;
outputs are `FIT_RF, APPLY_RF, FIT_LR, APPLY_LR`. FIT means only that iteration's
fit customers; APPLY means validation, or the final test set.

### 3.1 Median imputation

| ID | Native node | Connections and settings |
|---|---|---|
| P01 | Missing Value | FIT → column-specific Median for `tenure,MonthlyCharges,TotalCharges`; no categorical replacement, no row deletion. Output imputed FIT plus learned model. |
| P02 | Missing Value (Apply) | P01 model + APPLY table → imputed APPLY. Never fit a second Missing Value node on APPLY. |

KNIME's median must average the two middle nonmissing values for even counts;
verify against the numbers below rather than accepting a lower/upper median.

| Fit scope | n | tenure median | MonthlyCharges median | TotalCharges median |
|---|---:|---:|---:|---:|
| Full train | 5634 | 29 | 70.5 | 1398.125 |
| Fold 1 fit | 4507 | 29 | 70.5 | 1389.525 |
| Fold 2 fit | 4507 | 29 | 70.35 | 1396.9 |
| Fold 3 fit | 4507 | 29 | 70.45 | 1411.35 |
| Fold 4 fit | 4507 | 29 | 70.4 | 1396.25 |
| Fold 5 fit | 4508 | 29 | 70.7 | 1405.3 |

### 3.2 One-hot encoding, explicit native-rule implementation

Use the following data-specific implementation to avoid dependence on a PMML
apply node's unknown-category handling. Training-only inspection confirmed that
**every fit fold contains every listed category**, so these frozen training-derived
rules equal fitting `OneHotEncoder(handle_unknown="ignore", drop=None)` in each
fold on these saved data. This fact must be checked again in KNIME.

P03: **Domain Calculator**, FIT only, recompute (not retain inherited) domains
for the 16 categorical columns below. Inspect that they equal the listed sets
for each fold. Do not infer domains from concatenated FIT/APPLY data.
If any fit domain differs, **stop**: this fixed mapping is no longer valid;
rebuild its rules from that fit domain. This specification does not promise to
generalize to a changed dataset/split.

P04-F / P04-A: on P03 FIT and P02 APPLY respectively, instantiate **one Rule
Engine node for each category in the table**, chaining the output tables.
There are 43 nodes per branch, no scripts. For predictor `p`, category `v`,
append numeric column `categorical__p_v` with exactly:

```text
$p$ = "v" => 1
TRUE => 0
```

Replace `p,v` with the literal column/category below, including spaces and
parentheses. `SeniorCitizen` was read as String, so its values are `"0","1"`.
Thus an unrecognized value produces zero in **all** columns for that predictor,
not a new category, missing value, or dropped row. Reject missing categories
upstream as Python does. Keep all levels; do not use reference/dummy coding.

| Predictor, in order | Categories, in order |
|---|---|
| gender | Female; Male |
| SeniorCitizen | 0; 1 |
| Partner | No; Yes |
| Dependents | No; Yes |
| PhoneService | No; Yes |
| MultipleLines | No; No phone service; Yes |
| InternetService | DSL; Fiber optic; No |
| OnlineSecurity | No; No internet service; Yes |
| OnlineBackup | No; No internet service; Yes |
| DeviceProtection | No; No internet service; Yes |
| TechSupport | No; No internet service; Yes |
| StreamingTV | No; No internet service; Yes |
| StreamingMovies | No; No internet service; Yes |
| Contract | Month-to-month; One year; Two year |
| PaperlessBilling | No; Yes |
| PaymentMethod | Bank transfer (automatic); Credit card (automatic); Electronic check; Mailed check |

Example column: `categorical__MultipleLines_No phone service`.
P05-F/A: **Column Filter** removes the 16 original categorical columns,
retains metadata/target, three numeric predictors and 43 indicators.
P06-F/A: **Column Rename** renames numeric predictors to
`numeric__tenure,numeric__MonthlyCharges,numeric__TotalCharges`.
P07-F/A: **Column Resorter**, numeric columns first in that order, then the 43
indicator columns in table order. The feature sequence must equal
`encoded_features.csv`. Metadata may follow, outside the learner include list.
P07 outputs are FIT_RF and APPLY_RF: **no scaling**.

An alternative shorter implementation is One to Many (PMML) plus PMML
Transformation Apply, but it must first pass schema, unknown-category and
feature-order tests. It is not silently assumed equivalent here.

### 3.3 Population standardization: LR only

Avoid an implicit sample-SD versus population-SD difference in Normalizer.
Use explicit learned statistics:

| ID | Native node | Connections and settings |
|---|---|---|
| P08 | GroupBy | P07-F; no grouping columns. For each of three numeric columns calculate mean, count (nonmissing), standard deviation. Rename results `mean_tenure,n_tenure,s_tenure`, etc. Standard deviation here means sample SD, denominator n−1. |
| P09 | Math Formula ×3 | P08 → append `sigma_p = $s_p$ * sqrt(($n_p$ - 1.0) / $n_p$)` for each numeric p. This converts sample SD to population SD. If the installed aggregation already uses population SD, do not apply this correction; verify against reference values. |
| P10-F/A | Cross Joiner | P07-F/A × one-row P09 statistics table; creates one matched statistics row per input row, not n² rows. |
| P11-F/A | Math Formula ×3 | Replace only `numeric__p` with `($numeric__p$ - $mean_p$) / $sigma_p$`; Double output. For a zero-variance feature use denominator 1, as StandardScaler does. |
| P12-F/A | Column Filter + Column Resorter | Remove all nine statistic columns plus three sigma columns; restore the P07 feature order. Outputs FIT_LR, APPLY_LR. |

All FIT stats are computed **after median imputation**. All counts here exceed
one and all sigmas are nonzero on these data.

Full-training reference:

| p | mean after imputation | population SD |
|---|---:|---:|
| tenure | 32.48509052183174 | 24.56656301538815 |
| MonthlyCharges | 64.92996095136671 | 30.135430576006627 |
| TotalCharges | 2301.3199503017395 | 2277.606686938009 |

Do not paste these constants inside CV; its fit statistics differ.
Save each scope's medians, means, sigmas and feature list with CSV Writer.

## 4. Native learners and frozen settings

For every learner use `Churn` as target and **only the 46 named encoded
features** as attributes. No target reweighting, sampling to balance classes,
feature selection, calibration or threshold optimization.

### Logistic Regression Learner → Logistic Regression Predictor

Connect FIT_LR table → learner; learner PMML model + APPLY_LR table → predictor.
Configure predictor to append probabilities for **both** classes; rename its
Yes probability column to `churn_probability`. Do not use the reference-class
probability or the predictor's hard classification for evaluation.

| Setting | Python frozen value | Native configuration |
|---|---|---|
| Binary target / reference | Yes positive / No negative | `Churn`, reference category `No` |
| Encoding | All 43 dummy columns | Numeric encoded features only; no internal nominal dummy coding |
| Intercept | True, unpenalized | Include native intercept; verify its prior treatment before claiming equivalent penalty |
| Solver | LBFGS | **SAG**, because IRLS lacks regularization |
| L2 strength | selected `C=10`; grid `0.1,1,10` | Gaussian prior, variance **10** for the proposed native run; CV variances `0.1,1,10` |
| Iteration cap | `max_iter=2000` | Maximal epochs **2000** |
| Stopping tolerance | `tol=0.0001` | Epsilon **0.0001** |
| Random seed | 42 | Use seed, **42** |
| SAG-only settings | Not applicable | Fixed step size **0.1**, hold data in memory enabled, lazy calculations disabled |
| Inference statistics | Not produced | Calculate statistics for coefficients disabled |

**Penalty equivalence is not verified.** KNIME documents lambda =
`1/prior_variance`; variance 10 is therefore a nominal lambda=0.1 mapping to
`1/C`. Objective averaging, intercept penalization and convergence criteria must
be checked in the installed implementation before treating this as the *same*
regularized fit. Equal numeric epsilon/iteration settings do not mean equal
stopping conditions. Record this as `native_lr_gaussian_variance_10`, with
`python_C=10` and `penalty_equivalence=unverified`, not as an exact C=10 port.
Do not switch to unregularized IRLS or drop dummy columns to make fitting easier.

If strict identical estimator settings are mandatory, native KNIME LR is a
blocker; the specification cannot remove that platform limitation.

### Random Forest Learner → Random Forest Predictor

Connect FIT_RF table → learner; its **model output** + APPLY_RF table → predictor.

| Setting | Native value |
|---|---|
| Target | Churn |
| Attributes | The same 46 numeric features |
| Split criterion | Gini |
| Number of models | **300** |
| Limit number of levels | Enabled, **10**; root-only split is depth 1 |
| Minimum child node size | **5** |
| Static seed | Enabled, **42** |
| Save target distribution in tree nodes | **Enabled** |
| Predictor: use soft voting | **Enabled** |
| Predictor: append individual class probabilities | **Enabled**, select Yes for evaluation |
| Predictor: out-of-bag filtering | Disabled for validation/test |
| Highlighting | Disabled |

Native Random Forest fixes bootstrap sample size to the input size and samples
sqrt(m) features at each split, matching the intended bootstrap=True,
max_samples=None, max_features=sqrt choices. Here m=46, intended candidate
count floor(sqrt(46))=6; verify rounding in the installed node.
Python's other defaults are `min_samples_split=2`, no class weights, no pruning
(`ccp_alpha=0`), no impurity threshold, no leaf cap, `n_jobs=1`, no OOB scoring.
The native wrapper does not expose all of these as knobs; do not invent them.
Its child-size 5 effectively requires enough records for two children, but
bootstrap multiplicity/counting and split implementations can differ from
scikit-learn. RNGs also differ: seed 42 does not imply identical trees.
If a version exposes a minimum split-node size, use **10** with child size 5
(and **2** with child size 1); document its semantics.
Automatic native OOB output is unused, never a replacement for the saved folds.

Using original nominal columns instead of the 43 numeric indicators changes
the split search and sqrt(m), and is **not** this mapping. Hard-vote probability
(fraction of winning trees) is **not** Python's mean leaf probability.

### Majority baseline

FIT → **GroupBy** by `Churn`, Count → **Sorter** descending count →
**Row Filter** first row → **Table Row to Variable**. The majority must be No
in every fit scope. APPLY → **Constant Value Column** `churn_probability=0.0`
(Double), `model="majority_baseline"` → evaluation below.
Do not set probability to the training churn prevalence: Python's
`DummyClassifier(strategy="most_frequent")` emits zero for Yes.

## 5. Saved-fold CV and original candidate grid

This section reproduces the **design**, not previously observed Python scores.
The default final comparison uses the **frozen selected settings above**;
do not replace them after seeing native CV or test scores.

For each family, create the candidate table with **Table Creator**:

| family | candidate_order | C / proposed prior_variance | depth_limited | depth | min_child |
|---|---:|---:|---|---:|---:|
| logistic_regression | 0 | 0.1 | — | — | — |
| logistic_regression | 1 | 1 | — | — | — |
| logistic_regression | 2 | 10 | — | — | — |
| random_forest | 0 | — | false | 10 (ignored) | 1 |
| random_forest | 1 | — | false | 10 (ignored) | 5 |
| random_forest | 2 | — | true | 10 | 1 |
| random_forest | 3 | — | true | 10 | 5 |

These RF rows are the Python `ParameterGrid` order (depth outer, leaf inner).
The LR rows are a **native prior-variance comparison**, not a verified C search.
Use separate LR/RF branches; this avoids a learner-type switch inside a loop.

Connections:

1. C01 **Table Row to Variable Loop Start** over candidate table, emitting
   numeric/boolean variables to the appropriate learner configuration.
   Bind values through the learner's Flow Variables tab; use the installed
   dialog's keys, not guessed serialized settings keys.
2. C02 **Table Creator**, integer `validation_fold` rows `1,2,3,4,5`;
   C01 flow-variable connection → C02 → C03 **Table Row to Variable Loop Start**.
3. D16 + C03 variables → C04 **Rule-based Row Splitter**:
   `$fold$ = $${Ivalidation_fold}$$ => TRUE`, `TRUE => FALSE`.
   Matching table is APPLY/validation; other table is FIT. Sort both by
   `split_order` before preprocessing/learning.
4. C04 FIT/APPLY → a fresh P01–P12 instance → family learner/predictor →
   evaluation component E below.
5. Append `family,candidate_order,validation_fold` via **Variable to Table
   Column** to the one-row metric output → C05 **Loop End** (concatenate).
   This closes the inner fold loop; no model is reused across folds.
6. C05 → **GroupBy**, group on family/candidate_order; mean of each metric,
   population SD of ROC-AUC (sample SD × sqrt(4/5) if necessary).
   Preserve per-fold rows to a CSV Writer branch.
7. Aggregate rows → C06 outer **Loop End** (concatenate candidates).
8. C06 → **Sorter**, mean ROC-AUC descending then candidate_order ascending →
   **Row Filter**, first row. This implements Python's first-maximum tie rule.

Append candidate labels before closing their loops so that rows remain traceable.
For OOF predictions use **Loop End (2 ports)** at C05 to concatenate
`customerID,split_order,actual,prediction,churn_probability,validation_fold`
inside the fold loop. Sort collected rows back by split_order for export.

Validation sizes: folds 1–4 each 1,127 = 828 No + 299 Yes;
fold 5 has 1,126 = 827 No + 299 Yes.
35 learned fits total (3×5 LR + 4×5 RF); baseline can use one five-fold branch.
Selection metrics are **arithmetic means of five fold metrics**, not metrics
on pooled OOF predictions. The baseline preparation artifact also reports
pooled OOF metrics separately; retain this distinction.

The native optimum may differ. Save it as a difference; the final reproduction
still uses frozen Python-selected depth=10/leaf=5 and nominal LR variance=10.
A separately labelled native retuning experiment must not replace that comparison.

## 6. Common evaluation component E

Input: scored rows with nominal `Churn`, `customerID`, `split_order`, positive
`churn_probability`, family and scope. Assert no missing/out-of-range
probabilities before scoring.

| ID | Native node | Configuration / connections |
|---|---|---|
| E01 | Rule Engine | Append nominal `predicted_Churn`: `$churn_probability$ > 0.5 => "Yes"`, `TRUE => "No"`. Ignore predictor's built-in class column. |
| E02 | Rule Engine ×2 | Append integer `actual`: `$Churn$ = "Yes" => 1`, `TRUE => 0`; append integer `prediction` similarly from `predicted_Churn`. |
| E03 | Scorer | Actual `Churn`, predicted `predicted_Churn`; retain confusion matrix and class statistics, **Yes** row for precision/recall/F1, not macro/weighted averages. |
| E04 | ROC Curve | Actual `Churn`, positive class **Yes**, one probability column **churn_probability**. Output AUC and ROC points/view; do not use prediction, confidence of winning class, or P(No). |
| E05 | Rule Engine ×4 | On E02 rows append indicators `TN,FP,FN,TP`, one per node: e.g. `$actual$ = 1 AND $prediction$ = 1 => 1`, `TRUE => 0`; use 0/0,0/1,1/0 for the others. |
| E06 | GroupBy | E05 → sum each indicator; no grouping (or group by model/scope when concatenated). |
| E07 | Math Formula + Rule Engine | On counts calculate precision `TP/(TP+FP)`, recall `TP/(TP+FN)`, F1 `2*TP/(2*TP+FP+FN)`, using Double arithmetic. Guard each zero denominator with Rule Engine and return **0.0**; do not retain missing or infinity. |
| E08 | Cross Joiner | Join E07's single row with E04's single AUC row (extract the AUC statistic column); append model/scope/fold/candidate variables. |
| E09 | Column Filter + CSV Writer | E02 → `model,customerID,actual,prediction,churn_probability`, sorted by split_order before dropping it. |

If the installed ROC Curve exposes AUC only as a view/flow variable, convert
that AUC variable with **Variable to Table Row** for E08. Do not transcribe a
rounded view value into the metrics file. The following all-native fallback
defines the exact required tie-aware AUC if an unrounded statistic is unavailable:

1. E02 → **GroupBy** by exact `churn_probability`, aggregate Sum(actual)=`pos`
   and Count(actual)=`n`; **Math Formula** `neg=n-pos`.
2. **Sorter** probability ascending → **Moving Aggregator**, cumulative sum
   of `neg` from first row through current row → rename `cum_neg`.
3. **Math Formula** `contribution = pos * (cum_neg - neg + 0.5 * neg)`.
4. **GroupBy**, no keys: sum contribution, sum pos=P, sum neg=N.
5. **Math Formula** `roc_auc = contribution_sum / (P * N)`.

This gives half credit to equal-score positive/negative pairs, including
baseline AUC=0.5. All validation/test scopes here contain both classes.
E07 is authoritative for zero-division behavior even if Scorer displays missing
baseline precision. Confusion matrix output must be explicitly arranged as
**actual rows / predicted columns**, order `[No,Yes]`:
`[[TN,FP],[FN,TP]]`. Never infer orientation from a display.

## 7. Final training, evaluation and comparison

No candidate/fold loops for final fitting:

```text
D12 (all train) ──┐
                 ├─ P FIT/APPLY ─ FIT_LR ─ LR Learner ─ model ─┐
D13 (test) ───────┘                 APPLY_LR ───────────────────┴─ LR Predictor ─ E
                       └──────── FIT_RF ─ RF Learner ─ model ───┐
                                  APPLY_RF ───────────────────┴─ RF Predictor ─ E
D12 ─ majority check; D13 ─ constant P(Yes)=0 ─ E
```

Use the frozen native mappings in section 4, not the CV winner.
Write all outputs to the new run directory with **CSV Writer**, overwrite
disabled; use **Concatenate** to combine the three models' predictions/metrics.
Do not feed Python prediction/metric files into training or threshold selection.

Minimum handoff outputs:

* `test_predictions.csv`: same five columns as Python, rows in saved test-ID
  order within each model.
* `test_metrics.csv`: `model,roc_auc,precision,recall,f1`.
* `confusion_counts.csv`: `model,TN,FP,FN,TP`, with orientation documented.
* `cv_fold_metrics.csv`, `cv_comparison.csv`, `cv_predictions.csv`.
* `preprocessing_parameters.csv`: scope/fold/predictor/median/mean/population SD.
* `encoded_features.csv`, copies/references of original split and fold files.
* `native_settings.csv`: actual version/extensions, all learner settings above,
  threshold rule, paths/hashes, native-only settings, penalty-equivalence status.
* `python_knime_comparison.csv`: join saved Python and new KNIME metrics by
  model, append KNIME−Python differences using Math Formula.
* `prediction_comparison.csv`: join predictions on **model + customerID**,
  compare actuals, probability difference and hard-class disagreement.
* Export the built workflow as `telco-native.knwf` only after validation below.

Original Python test reference (not claimed KNIME results):

| Model | ROC-AUC | Precision | Recall | F1 | TN / FP / FN / TP |
|---|---:|---:|---:|---:|---|
| majority_baseline | .5000000000 | 0 | 0 | 0 | 1035 / 0 / 374 / 0 |
| logistic_regression | .8410912191 | .6572327044 | .5588235294 | .6040462428 | 926 / 109 / 165 / 209 |
| random_forest | .8429951691 | .6759581882 | .5187165775 | .5869894100 | 942 / 93 / 180 / 194 |

The majority baseline should match exactly. Native LR/RF scores are **not**
expected to match exactly; do not modify settings to chase these test scores.
Check metric arithmetic on the *same saved predictions* before interpreting
differences between newly trained models.

## 8. Interpretation and descriptive branches

The required predictive comparison is sections 2–7. The Python project also
contains descriptive outputs and post-hoc interpretation. Do not label a port
of predictive fitting alone as reproduction of those additional analyses.

* Descriptives: D05 → **GroupBy** counts/means/medians/SD by Churn and each
  categorical predictor; **Math Formula** churn fractions; **CSV Writer**.
  Preserve “No internet service” and “No phone service” as distinct categories.
  The offline HTML viewer is not a modelling step and has no native counterpart.
* Logistic contrasts: learner coefficient-table output → select the coefficient
  for Yes vs No. Numerical effect = `exp(beta)` per training SD. For each
  categorical predictor take its first listed category's coefficient as
  reference, join it back by predictor, calculate `exp(beta_level-beta_first)`.
  Use **Table Creator** to map the 46 feature names to predictor/category/reference,
  **Joiner**, **Row Filter**, and **Math Formula**. Do not exponentiate individual
  all-level dummy coefficients and call them category-vs-reference odds ratios.
* RF permutation importance: freeze the trained RF and preprocess FIT once.
  **Table Creator** lists the 19 original predictors, **Table Row to Variable
  Loop Start** iterates predictors, inner table/loop repeats 1–10.
  On the **original** test predictor table, **Counter Generation** adds an
  original position. **Column Filter** selects that predictor; **Shuffle** rows
  with an explicit repeat-specific seed (e.g. 42–51); remove old row keys/position,
  add a fresh sequential position with Counter Generation, rename the shuffled
  value; **Joiner** on the new position assigns it to the unshuffled test rows.
  Replace only that original predictor; apply the already fitted imputer,
  encoding rules and frozen RF, **not** refit. Calculate unpermuted AUC minus
  permuted AUC; collect 10 drops/predictor, **GroupBy** mean and population SD.
  Shuffle an original categorical column **before** encoding, never one dummy
  at a time. This is the same statistic, but KNIME's shuffle RNG/seeding does
  not recreate scikit-learn's exact permutations; save repeat-level drops and
  label them native. Keep negative drops. Never use this branch for selection.
* Native charts need not reproduce Python's hand-written SVG styling. Save
  the underlying metric/importance tables; chart rendering is a presentation
  difference, not a scoring change.

All interpretations are associations, not causal effects. Earlier descriptive
exploration included eventual test customers; this remains a limitation in KNIME.

## 9. Import/execution acceptance checklist — NOT PERFORMED

When KNIME is available:

1. Record platform version, OS, feature versions and node factory versions.
   Build the nodes above; inspect every warning and bound flow variable.
2. Execute input checks; verify counts, ordered IDs, folds, labels, median values,
   all fold category domains, 46-feature order and full-training means/sigmas.
   Compare preprocessing values within floating-point tolerance (e.g. 1e−10
   relative), not by rounded screenshots.
3. Test encoder with an artificial unknown category: its 2–4 indicator columns
   must all be zero. Test probability 0.5 → No and >0.5 → Yes.
4. Read Python's saved test predictions into component E, separately per model.
   The resulting counts and metrics must match saved Python outputs (metrics
   within 1e−12 absolute). This tests metric parity without claiming model parity.
5. Execute CV from reset. Confirm fit/validation counts, no test input inside
   CV, fit-only statistics and exactly five folds per candidate, with 5,634
   OOF rows (one prediction per train customer per candidate).
6. Execute final branches from reset using frozen settings; verify exactly
   1,409 ordered prediction rows/model and finite Yes probabilities in [0,1].
   Check baseline exact agreement; save LR/RF differences without retuning.
7. Export `.knwf`, import into a **fresh workspace**, reset and execute all
   nodes with the same input files. Verify exported CSVs again. An export
   without a successful fresh import/execution is not validation.
8. Record execution date, input hashes, settings, outputs, warnings, unresolved
   penalty differences and import/execution result. Only then call the archive
   tested. Do not describe native LR as identical to Python while the penalty
   conversion remains unresolved.

## Sources

Dataset and attribution: [IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d).
No IBM Cloud, Watson deployment or Flask application is reused.

Node documentation consulted for this specification:

* [Logistic Regression Learner](https://nodepit.com/node/org.knime.base.node.mine.regression.logistic.learner4.LogRegLearnerNodeFactory4):
  SAG/IRLS, Gaussian prior, epsilon and seed semantics.
* [Random Forest Learner](https://nodepit.com/node/org.knime.base.node.mine.treeensemble2.node.randomforest.learner.classification.RandomForestClassificationLearnerNodeFactory2):
  bootstrapping, sqrt feature sampling, depth/child-size and distribution storage.
* [Random Forest Predictor](https://nodepit.com/node/org.knime.base.node.mine.treeensemble2.node.randomforest.predictor.classification.RandomForestClassificationPredictorNodeFactory2):
  soft voting versus hard voting.
* [Missing Value (Apply)](https://nodepit.com/node/org.knime.base.node.preproc.pmml.missingval.apply.MissingValueApplyNodeFactory):
  reuse of training-fitted replacement model.

Documentation is not a substitute for execution in the installed version.
