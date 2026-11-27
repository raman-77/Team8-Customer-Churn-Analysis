"""Data audit: identify 100 identical-input pairs with different churn labels.

Usage: python detect_100_conflicting_pairs.py
The procedure is independent of ANN model inference.
"""
from pathlib import Path
import numpy as np
import pandas as pd
path = Path(__file__).with_name("Updated_Dataset (1).csv")
df = pd.read_csv(path)
features = [col for col in df.columns if col != "Churn"]
# Exact duplicates use all columns including Churn.
# The conflicting-label audit excludes Churn when constructing input profiles.
assert df.duplicated().sum() == 0
profile_id = pd.factorize(pd.MultiIndex.from_frame(df[features]), sort=False)[0]
summary = df.assign(Profile_ID=profile_id).groupby("Profile_ID")["Churn"].agg(["size", "nunique"])
conflicts = summary.loc[summary["nunique"] > 1]
evidence = df.assign(Profile_ID=profile_id, Excel_Row=np.arange(len(df))+2)
evidence = evidence[evidence["Profile_ID"].isin(conflicts.index)]
evidence = evidence.sort_values(["Profile_ID", "Churn"])
print(f"Input features: {len(features)}; exact duplicate rows: {df.duplicated().sum()}")
print(f"Conflicting feature profiles: {len(conflicts)}")
print(f"Records affected: {len(evidence)}")
print("First conflicting pair (Excel row number includes CSV header):")
print(evidence[evidence["Profile_ID"]==conflicts.index[0]].to_string(index=False))
assert len(conflicts)==100 and len(evidence)==200 and (conflicts["size"]==2).all()
evidence.to_csv(Path(__file__).with_name("100_pairs_evidence.csv"), index=False)
print("Wrote 100_pairs_evidence.csv")
