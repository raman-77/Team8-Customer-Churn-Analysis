# Part 3 — ANN Customer Churn Prediction: Methodology, Evidence and Findings

**Prepared by:** Bao Ngoc Nguyen — Data Analyst (Predictive Modelling)  
**Project:** ACS Work Integrated Learning — Customer Churn Analysis  
**Data sources:** Original Stage 2 dataset (7,043 records) and revised deduplicated Stage 2 dataset (6,741 records)  
**Models:** ANN A (customer attributes only) and ANN B (customer attributes plus training-derived cluster indicators)

## 1. Study objective and scope

This analysis evaluates customer churn prediction using an Artificial Neural Network (ANN). Two research questions are addressed: (1) whether cluster membership provides incremental predictive information, and (2) whether removal of 302 exact duplicate rows changes performance when the original and revised datasets are evaluated against the **same independent test records**.

Removal of exact duplicate rows does not establish that all removed records represented the same real customer, because a unique CustomerID is unavailable. The comparison is therefore treated as a **sensitivity analysis**, not validation of individual customer identity.

## 2. Data readiness and validation design

The revised dataset contains **6,741 rows** after 302 exact duplicate rows were removed from the 7,043-row source. The reported overall churn rate is approximately **26.6%**. The revised customer-attribute file and its clustering counterpart agree on the shared columns; the clustering file additionally contains `Cluster`.

A **stratified, group-aware 70% training / 20% validation / 10% testing** partition was used. Observations with identical input features are assigned to the same partition even when their outcome labels differ. This protects evaluation from the overlap of identical feature profiles across the training and testing sets.

| Partition | Revised records | Churn cases | Fraction |
|---|---:|---:|---:|
| Training | 4,719 | 1,254 | 70.0% |
| Validation | 1,348 | 358 | 20.0% |
| Test | 674 | 179 | 10.0% |

The partitioning procedure uses ten approximately equal stratified, group-aware folds, with seven allocated to training, two to validation and one to testing. The validation set supports early stopping and threshold selection. The **674-record test set remains independent** and is used for final model evaluation.

The original-data experiment uses **4,933 training rows**, consisting of 4,719 distinct revised-data training records and 214 additional duplicate copies associated exclusively with training profiles. The other 88 duplicate copies are associated with validation or test profiles and are excluded from original-data training to prevent overlap with held-out observations. Validation and test records are identical across both experiments.

## 3. Controlled comparison: original versus deduplicated training data

All ANN variants use the same initial architecture (32 and 16 hidden neurons), training configuration, random seed and decision threshold of **0.50**. Class weights and numerical scaling are fitted using the corresponding training partition only. For ANN B, K-Means is fitted to the **training partition only**, rather than using cluster labels derived from the full dataset.

| Dataset and model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Original 7,043 — ANN A, without clusters | 71.4% | 47.5% | 74.9% | 58.1% | 0.7936 |
| Revised 6,741 — ANN A, without clusters | 71.5% | 47.7% | 76.0% | 58.6% | 0.7941 |
| Original 7,043 — ANN B, with clusters | 71.2% | 47.3% | 74.9% | 58.0% | 0.7958 |
| Revised 6,741 — ANN B, with clusters | 71.8% | 48.0% | 72.6% | 57.8% | 0.7924 |

**ANN A (without clusters):** ROC-AUC increased by approximately **0.0005** and recall by **1.12 percentage points** after exact duplicate removal.  
**ANN B (with clusters):** ROC-AUC decreased by approximately **0.0034** and recall by **2.23 percentage points** following the same change.

**Interpretation.** Performance changes were small and not consistently favourable to either data version. A single fixed holdout cannot establish that duplicate removal universally improves churn prediction. Duplicate copies change the relative training weight of particular customer profiles; the derived class weights and, for ANN B, cluster boundaries may also change slightly.

## 4. Verification of 100 identical-feature profiles with different churn labels

The data audit distinguishes **exact duplicates** (all columns, including `Churn`, match) from **conflicting-label feature profiles** (the ten input features match, while `Churn` differs).

The revised dataset contains **100 feature profiles**, each represented by two records with opposite churn outcomes: **200 records in total**. These were identified through a data consistency check, **not by ANN prediction**.

```python
import pandas as pd

df = pd.read_csv("Updated_Dataset (1).csv")
features = [c for c in df.columns if c != "Churn"]
check = df.groupby(features, dropna=False)["Churn"].agg(["size", "nunique"])
conflicting = check[check["nunique"] > 1]
print(len(conflicting))         # 100 input profiles
print(conflicting["size"].sum())  # 200 records
```

**Illustrative evidence:** Excel rows **2 and 5896** have identical values for all ten input variables, but opposite churn labels (`1` and `0`). The companion script `detect_100_conflicting_pairs.py` exports the complete evidence table.

Because deterministic ANN inference assigns the same probability to an identical input vector, both records in such a pair cannot receive different predictions at a fixed threshold. The observations are not automatically considered erroneous; they may represent distinct customers with identical measured characteristics. This constraint is reported as a limitation of the available attributes and labels.

## 5. Treatment of `SeniorCitizen` and other predictors

The Data Engineer indicated that standardisation of `SeniorCitizen` was intended to support the investigation of potential outliers during preprocessing. Such standardised scores can be informative when screening unusual values among **continuous** measurements. However, `SeniorCitizen` denotes two **valid categories**, with `0` indicating non-senior and `1` indicating senior status.

The relatively high standardised score for the less common senior group indicates category prevalence, rather than evidence of an erroneous or unusually measured record. The modelling pipeline therefore **restores `SeniorCitizen` to its original 0/1 representation** for clarity and consistency with the other binary indicators. This is a deliberate modelling choice, not a claim that binary standardisation is mathematically invalid in all applications.

The continuous features `tenure` and `MonthlyCharges` are standardised using **training-set statistics only**. Since the provided files already contain an affine standardisation of these variables, refitting a scaler on the training subset reproduces the effective training-only transformation up to numerical precision. Access to original unscaled values would support a more complete audit of upstream data preparation.

## 6. Rationale for the 70/20/10 model-development split

The **80/20 training/testing split prepared during data engineering is a valid conventional approach**, particularly when a validation partition is subsequently drawn from the training data or cross-validation is applied. The ANN workflow adopts **70% training / 20% validation / 10% testing** to provide distinct datasets for parameter learning, early stopping and model/threshold selection, and independent final evaluation.

The **10% test partition contains 674 records**, including 179 churn cases. This is smaller than a typical 20% test partition and may yield less precise test estimates. Additional group-aware validation splits would strengthen confidence in comparisons between ANN variants. The three-way split is selected for methodological transparency rather than as a categorical rejection of the 80/20 approach.

## 7. Business interpretation and limitations

- **Recall** measures the proportion of actual churners identified; missed churners may represent missed retention opportunities.
- **Precision** indicates how often the predicted at-risk group truly churned and is relevant to retention outreach costs.
- The reported results represent a **single offline holdout on one customer snapshot**, not prospective business performance or proof of causal drivers.
- The absence of CustomerID limits verification of duplicate records, while the 100 identical-feature profiles with opposite labels demonstrate missing predictive distinctions.
- Additional service usage, complaint and payment history could support more informative future models, subject to data availability and appropriate governance.
- Small differences in ROC-AUC or recall should not be described as reliable superiority without further repeated validation.

**Next steps:** Review false negatives, verify modelling decisions and evaluation metrics, consider an outreach-capacity-aware decision threshold, and provide results to the **Business Analyst** for retention recommendations.
