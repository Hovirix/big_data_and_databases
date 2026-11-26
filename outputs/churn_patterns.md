# Churn across customer groups

Source: [IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d).
Reproduce from the repository root: `nix develop --command uv run python overview.py`.

Overall churn: **26.54%**.
Churn rate = churners / customers within each group × 100.
Difference (pp) = group churn rate minus overall churn, in percentage points.
Counts show the size behind each percentage; rates are not shares of all churners.

Tenure bands are inclusive integer months. SeniorCitizen: 0 = No, 1 = Yes.
Monthly charges use approximately equal-sized quartile groups:

- Q1: ≥18.25 and ≤35.50.
- Q2: >35.50 and ≤70.35.
- Q3: >70.35 and ≤89.85.
- Q4: >89.85 and ≤118.75.

Exact bin boundaries and closure rules are saved in `group_settings.json`.
Full results, including other billing categories, are in `group_churn.csv`.

## Contract

| Group | Customers | Churners | Churn rate | Difference (pp) |
|---|---:|---:|---:|---:|
| Month-to-month | 3,875 | 1,655 | 42.71% | +16.17 |
| One year | 1,473 | 166 | 11.27% | -15.27 |
| Two year | 1,695 | 48 | 2.83% | -23.71 |

Highest observed rate: Month-to-month (42.71%); lowest: Two year (2.83%).

## Tenure band (months)

| Group | Customers | Churners | Churn rate | Difference (pp) |
|---|---:|---:|---:|---:|
| 0–12 | 2,186 | 1,037 | 47.44% | +20.90 |
| 13–24 | 1,024 | 294 | 28.71% | +2.17 |
| 25–48 | 1,594 | 325 | 20.39% | -6.15 |
| 49–72 | 2,239 | 213 | 9.51% | -17.02 |

Highest observed rate: 0–12 (47.44%); lowest: 49–72 (9.51%).

## Monthly charge quartile

| Group | Customers | Churners | Churn rate | Difference (pp) |
|---|---:|---:|---:|---:|
| Q1 | 1,762 | 198 | 11.24% | -15.30 |
| Q2 | 1,766 | 434 | 24.58% | -1.96 |
| Q3 | 1,757 | 659 | 37.51% | +10.97 |
| Q4 | 1,758 | 578 | 32.88% | +6.34 |

Highest observed rate: Q3 (37.51%); lowest: Q1 (11.24%).

## gender

| Group | Customers | Churners | Churn rate | Difference (pp) |
|---|---:|---:|---:|---:|
| Female | 3,488 | 939 | 26.92% | +0.38 |
| Male | 3,555 | 930 | 26.16% | -0.38 |

Highest observed rate: Female (26.92%); lowest: Male (26.16%).

## SeniorCitizen

| Group | Customers | Churners | Churn rate | Difference (pp) |
|---|---:|---:|---:|---:|
| 0 | 5,901 | 1,393 | 23.61% | -2.93 |
| 1 | 1,142 | 476 | 41.68% | +15.14 |

Highest observed rate: 1 (41.68%); lowest: 0 (23.61%).

## Partner

| Group | Customers | Churners | Churn rate | Difference (pp) |
|---|---:|---:|---:|---:|
| No | 3,641 | 1,200 | 32.96% | +6.42 |
| Yes | 3,402 | 669 | 19.66% | -6.87 |

Highest observed rate: No (32.96%); lowest: Yes (19.66%).

## Dependents

| Group | Customers | Churners | Churn rate | Difference (pp) |
|---|---:|---:|---:|---:|
| No | 4,933 | 1,543 | 31.28% | +4.74 |
| Yes | 2,110 | 326 | 15.45% | -11.09 |

Highest observed rate: No (31.28%); lowest: Yes (15.45%).

## PhoneService

| Group | Customers | Churners | Churn rate | Difference (pp) |
|---|---:|---:|---:|---:|
| No | 682 | 170 | 24.93% | -1.61 |
| Yes | 6,361 | 1,699 | 26.71% | +0.17 |

Highest observed rate: Yes (26.71%); lowest: No (24.93%).

## MultipleLines

| Group | Customers | Churners | Churn rate | Difference (pp) |
|---|---:|---:|---:|---:|
| No | 3,390 | 849 | 25.04% | -1.49 |
| No phone service | 682 | 170 | 24.93% | -1.61 |
| Yes | 2,971 | 850 | 28.61% | +2.07 |

Highest observed rate: Yes (28.61%); lowest: No phone service (24.93%).

## InternetService

| Group | Customers | Churners | Churn rate | Difference (pp) |
|---|---:|---:|---:|---:|
| DSL | 2,421 | 459 | 18.96% | -7.58 |
| Fiber optic | 3,096 | 1,297 | 41.89% | +15.36 |
| No | 1,526 | 113 | 7.40% | -19.13 |

Highest observed rate: Fiber optic (41.89%); lowest: No (7.40%).

## OnlineSecurity

| Group | Customers | Churners | Churn rate | Difference (pp) |
|---|---:|---:|---:|---:|
| No | 3,498 | 1,461 | 41.77% | +15.23 |
| No internet service | 1,526 | 113 | 7.40% | -19.13 |
| Yes | 2,019 | 295 | 14.61% | -11.93 |

Highest observed rate: No (41.77%); lowest: No internet service (7.40%).

## OnlineBackup

| Group | Customers | Churners | Churn rate | Difference (pp) |
|---|---:|---:|---:|---:|
| No | 3,088 | 1,233 | 39.93% | +13.39 |
| No internet service | 1,526 | 113 | 7.40% | -19.13 |
| Yes | 2,429 | 523 | 21.53% | -5.01 |

Highest observed rate: No (39.93%); lowest: No internet service (7.40%).

## DeviceProtection

| Group | Customers | Churners | Churn rate | Difference (pp) |
|---|---:|---:|---:|---:|
| No | 3,095 | 1,211 | 39.13% | +12.59 |
| No internet service | 1,526 | 113 | 7.40% | -19.13 |
| Yes | 2,422 | 545 | 22.50% | -4.03 |

Highest observed rate: No (39.13%); lowest: No internet service (7.40%).

## TechSupport

| Group | Customers | Churners | Churn rate | Difference (pp) |
|---|---:|---:|---:|---:|
| No | 3,473 | 1,446 | 41.64% | +15.10 |
| No internet service | 1,526 | 113 | 7.40% | -19.13 |
| Yes | 2,044 | 310 | 15.17% | -11.37 |

Highest observed rate: No (41.64%); lowest: No internet service (7.40%).

## StreamingTV

| Group | Customers | Churners | Churn rate | Difference (pp) |
|---|---:|---:|---:|---:|
| No | 2,810 | 942 | 33.52% | +6.99 |
| No internet service | 1,526 | 113 | 7.40% | -19.13 |
| Yes | 2,707 | 814 | 30.07% | +3.53 |

Highest observed rate: No (33.52%); lowest: No internet service (7.40%).

## StreamingMovies

| Group | Customers | Churners | Churn rate | Difference (pp) |
|---|---:|---:|---:|---:|
| No | 2,785 | 938 | 33.68% | +7.14 |
| No internet service | 1,526 | 113 | 7.40% | -19.13 |
| Yes | 2,732 | 818 | 29.94% | +3.40 |

Highest observed rate: No (33.68%); lowest: No internet service (7.40%).

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
