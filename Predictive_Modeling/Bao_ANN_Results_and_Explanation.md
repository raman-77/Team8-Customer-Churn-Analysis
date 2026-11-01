# ANN Churn Prediction — Results and Plain-English Explanation
**Analyst:** Bao Ngoc Nguyen | **Project:** ACS WIL Customer Churn Analysis

**Source:** `Updated_Dataset_with_Clusters.csv` provided by the team. The notebook is the complete, step-by-step reproducible workflow.

## What was done?

1. **Dataset review:** 7,043 records, no missing cells, 302 duplicate rows. We did not automatically remove duplicates because no real CustomerID exists. Identical input profiles were kept in only one data split to reduce optimistic evaluation.
2. **Encoding:** `gender` (Male=1), `Dependents` (Yes=1), existing boolean columns (True=1), `Churn` (Yes=1). `SeniorCitizen` had been standardised; its higher observed binary value was restored to 1 and its lower value to 0. No separate `raw` dataframe was available.
3. **Class distribution:** Churn occurs in 26.5% of rows. This imbalance makes accuracy alone insufficient.
4. **Train/validation/test:** About 64/16/20 split. Numerical scaling was fitted only on training data, then applied to validation and test.
5. **Simple benchmark:** Logistic Regression was trained so we do not assume ANN is better just because it is more complex.
6. **ANN A:** Dense(32) → Dropout(0.2) → Dense(16) → Sigmoid output, using weighted binary cross-entropy and early stopping.
7. **ANN B:** Same ANN with four additional one-hot cluster indicators. To avoid using test-set information from the supplied full-dataset clusters, the cluster-generation step was re-fitted with K=4 on the training split only.
8. **Evaluation:** Accuracy, precision, recall, F1, ROC-AUC, confusion matrices, ROC curves, model training curves, and feature permutation importance.

## Test results (threshold = 0.5; n = 1,408)

| Metric | Logistic Regression | ANN A (without clusters) | ANN B (with train-only clusters) |
|---|---:|---:|---:|
| Accuracy | 75.1% | 75.0% | 75.9% |
| Precision | 52.1% | 51.9% | 53.2% |
| Recall | **77.7%** | 75.1% | 74.8% |
| F1-score | **62.4%** | 61.4% | 62.2% |
| ROC-AUC | 0.838 | 0.838 | 0.839 |

**Plain English:** Both ANNs identify predictive patterns associated with churn. ANN B has slightly higher accuracy and precision, but it misses slightly more actual churners. Its ROC-AUC gain over ANN A is only about **0.001**, which is too small to claim a meaningful improvement. The simple Logistic Regression benchmark performs competitively, particularly on recall. Model choice should consider the company's priorities and ideally be confirmed through repeated validation or cross-validation.

## Key predictors (ANN A, permutation-based ROC-AUC drop)

- **Tenure:** largest measured drop (~0.241), suggesting strong predictive reliance.
- **MonthlyCharges:** second-largest drop (~0.116).
- All other features show far smaller drops under this particular method and test split.

**Caution:** This is **not proof of causation**, and shuffling features can be affected by correlation and other modelling choices. The cluster results in Jyoti's Part 2 are descriptive, not causal explanations.

## Implications for the Business Analyst

Use churn probabilities to prioritise *potential* retention outreach, not to label real customers with certainty. A high-recall setting may prevent missed churners but also produce more false alerts. Define the cost of those errors before choosing a final decision threshold or recommending campaigns. No retention strategy has been tested experimentally in this dataset.

## Key limitations

Only one cross-sectional snapshot; no real CustomerID; duplicate-profile ambiguity; the original cluster labels were fitted on all rows, so new train-only K-Means was used for fair predictive comparison; one random train/validation/test split; predictive associations are not explanations of *why* churn occurs. The notebook includes no production deployment and does not claim real-time predictive performance.

## Meeting-ready summary

> My role is to predict individual customer churn and check whether the segmentation insights add useful predictive information. Using our prepared dataset, I built a simple benchmark and two ANN models. I evaluated them on held-out customers using accuracy, precision, recall, F1 and ROC-AUC. Adding cluster information brought almost no improvement in ROC-AUC, and Logistic Regression remained competitive. Tenure and monthly charges were the strongest predictors in my permutation analysis. I will report the limitations, including class imbalance, duplicate records and data leakage prevention, and share the prediction results with the Business Analyst for retention planning.
