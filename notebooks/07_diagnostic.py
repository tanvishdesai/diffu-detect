"""
==========================================================================
  DiffuDetect — Notebook 07: DIAGNOSTIC — Find What's Wrong
==========================================================================

PURPOSE:
  ALL methods produced AUROC < 0.50 (worse than random).
  This notebook diagnoses WHY by inspecting:
    1. The actual MAGE dataset structure and label convention
    2. Mean scores per label class
    3. Correct vs inverted AUROC
    4. Whether the issue is labels, score direction, or both

KAGGLE SETTINGS:
  - GPU: None (CPU is fine)
  - Internet: ON (to download MAGE metadata)

RUN THIS BEFORE re-running any scoring notebooks.
"""

# !pip install -q datasets pandas scikit-learn pyarrow

import os, sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

# =========================================================================
#  PART 1: INSPECT THE RAW MAGE DATASET
# =========================================================================

print("=" * 80)
print("PART 1: RAW MAGE DATASET INSPECTION")
print("=" * 80)

from datasets import load_dataset

ds = load_dataset("yaful/MAGE", trust_remote_code=True)
print(f"\nAvailable splits: {list(ds.keys())}")

for split_name in ds:
    split = ds[split_name]
    print(f"\n--- Split: {split_name} ---")
    print(f"  Num rows: {len(split)}")
    print(f"  Columns: {split.column_names}")
    print(f"  Features: {split.features}")

    # Show first 3 rows
    df_sample = split.to_pandas().head(3)
    print(f"\n  First 3 rows:")
    for idx, row in df_sample.iterrows():
        print(f"    Row {idx}:")
        for col in df_sample.columns:
            val = row[col]
            if isinstance(val, str) and len(val) > 100:
                val = val[:100] + "..."
            print(f"      {col}: {val}")
        print()

# =========================================================================
#  PART 2: LABEL CONVENTION ANALYSIS
# =========================================================================

print("\n" + "=" * 80)
print("PART 2: LABEL CONVENTION — WHAT DO 0 AND 1 MEAN?")
print("=" * 80)

# Use test split if available, else train
if "test" in ds:
    full_df = ds["test"].to_pandas()
    used_split = "test"
elif "train" in ds:
    full_df = ds["train"].to_pandas()
    used_split = "train"
else:
    first = list(ds.keys())[0]
    full_df = ds[first].to_pandas()
    used_split = first

print(f"\nUsing split: {used_split} ({len(full_df)} rows)")
print(f"Columns: {list(full_df.columns)}")

# Show label distribution
print(f"\n--- Label column analysis ---")
if "label" in full_df.columns:
    print(f"Label value counts:\n{full_df['label'].value_counts().to_string()}")
    print(f"Label dtype: {full_df['label'].dtype}")
    print(f"Unique values: {sorted(full_df['label'].unique())}")
else:
    print("WARNING: No 'label' column found!")
    print(f"Available columns: {list(full_df.columns)}")

# Check for generator/model columns
print(f"\n--- Generator/Source columns ---")
for col in full_df.columns:
    if col in ["label", "text"]:
        continue
    nunique = full_df[col].nunique()
    if nunique <= 30:  # Categorical-ish
        print(f"\n  {col} ({nunique} unique values):")
        print(f"    {full_df[col].value_counts().head(15).to_string()}")

# Show sample texts from each label class
print(f"\n--- Sample texts by label ---")
if "label" in full_df.columns:
    text_col = None
    for c in ["text", "content", "passage", "document"]:
        if c in full_df.columns:
            text_col = c
            break

    if text_col:
        for label_val in sorted(full_df["label"].unique()):
            sub = full_df[full_df["label"] == label_val]
            print(f"\n  LABEL = {label_val} ({len(sub)} rows):")
            for i, (_, row) in enumerate(sub.head(2).iterrows()):
                txt = str(row[text_col])[:200]
                print(f"    Sample {i+1}: \"{txt}...\"")
                # Print any other metadata columns
                for c in full_df.columns:
                    if c not in [text_col, "label"] and not isinstance(row[c], str):
                        print(f"      {c} = {row[c]}")
                    elif c not in [text_col, "label"] and isinstance(row[c], str) and len(str(row[c])) < 50:
                        print(f"      {c} = {row[c]}")

# Cross-reference labels with generator info
print(f"\n--- Label × Generator cross-tab ---")
gen_col = None
for c in ["src", "source_model", "model", "generator", "source"]:
    if c in full_df.columns:
        gen_col = c
        break

if gen_col:
    print(f"Using generator column: '{gen_col}'")
    ct = pd.crosstab(full_df["label"], full_df[gen_col])
    print(ct.to_string())
    print(f"\nSo label=0 corresponds to: {full_df[full_df['label']==0][gen_col].unique()}")
    print(f"And label=1 corresponds to: {full_df[full_df['label']==1][gen_col].unique()}")
else:
    print("No generator column found.")
    print("Cannot determine what label=0 and label=1 mean from metadata.")
    print("Will determine from scoring patterns instead.")

# =========================================================================
#  PART 3: ANALYZE EXISTING SCORES (if available)
# =========================================================================

print("\n" + "=" * 80)
print("PART 3: ANALYZE EXISTING SCORE FILES")
print("=" * 80)

# Try to find score files
score_dirs = [
    "/kaggle/input/diffudetect-mre-scores/results",
    "/kaggle/input/diffudetect-baseline-scores/results",
    "/kaggle/input/diffudetect-dc-dtd-scores/results",
    "/kaggle/working/results",
    "./results",
]

import glob
score_files = []
for d in score_dirs:
    if os.path.exists(d):
        score_files.extend(glob.glob(os.path.join(d, "scores_*.parquet")))

if score_files:
    print(f"Found {len(score_files)} score files")

    for sf in score_files:
        print(f"\n--- {os.path.basename(sf)} ---")
        sdf = pd.read_parquet(sf)
        print(f"  Rows: {len(sdf)}, Columns: {list(sdf.columns)}")

        # Find score columns
        meta_cols = {"id", "text", "label", "generator", "domain", "dataset", "attack"}
        s_cols = [c for c in sdf.columns if c not in meta_cols]

        if "label" in sdf.columns and s_cols:
            labels = sdf["label"].values
            print(f"  Labels: {dict(zip(*np.unique(labels, return_counts=True)))}")

            print(f"\n  Mean scores by label class:")
            for sc in s_cols:
                vals = sdf[sc].values
                valid = np.isfinite(vals)
                if valid.sum() < 10:
                    continue

                mean_0 = np.nanmean(vals[labels == 0])
                mean_1 = np.nanmean(vals[labels == 1])

                # Try AUROC both ways
                try:
                    auroc_raw = roc_auc_score(labels[valid], vals[valid])
                    auroc_neg = roc_auc_score(labels[valid], -vals[valid])
                except:
                    auroc_raw = auroc_neg = 0.5

                # Also try with flipped labels
                try:
                    auroc_flipped = roc_auc_score(1 - labels[valid], vals[valid])
                except:
                    auroc_flipped = 0.5

                print(f"\n    {sc}:")
                print(f"      mean(label=0) = {mean_0:.4f}")
                print(f"      mean(label=1) = {mean_1:.4f}")
                print(f"      AUROC(raw scores) = {auroc_raw:.4f}")
                print(f"      AUROC(negated scores) = {auroc_neg:.4f}")
                print(f"      AUROC(flipped labels) = {auroc_flipped:.4f}")

                # Determine which interpretation works
                best_auroc = max(auroc_raw, auroc_neg, auroc_flipped)
                if best_auroc == auroc_raw:
                    print(f"      → Best: raw scores (higher = label 1)")
                elif best_auroc == auroc_neg:
                    print(f"      → Best: negated scores (lower = label 1)")
                else:
                    print(f"      → Best: FLIPPED LABELS (label convention is inverted!)")
else:
    print("No score files found. Run this after scoring notebooks.")
    print("Skipping score analysis, but PART 1 & 2 results above should reveal the issue.")

# =========================================================================
#  PART 4: DIAGNOSIS SUMMARY
# =========================================================================

print("\n" + "=" * 80)
print("PART 4: DIAGNOSIS SUMMARY")
print("=" * 80)

print("""
CHECK THE OUTPUT ABOVE FOR:

1. LABEL CONVENTION:
   - What does label=0 mean? What does label=1 mean?
   - If label=0 = MACHINE and label=1 = HUMAN, our code has them backwards!
   - Look at the 'Label × Generator cross-tab' section.

2. COLUMN MAPPING:
   - What are the actual column names in MAGE?
   - Did 'source_model' or 'category' exist? Or different names?

3. SCORE DIRECTION:
   - Look at 'Mean scores by label class'.
   - For MRE: Is mean(label=0) > or < mean(label=1)?
   - For log-likelihood: Which label has higher LL?
   - The AUROC analysis shows which interpretation gives best results.

4. BASED ON FINDINGS, update the scoring notebooks:
   - If labels are inverted: change `df["label"] = 1 - df["label"]` in data loading
   - If score direction wrong: change the AUROC computation sign
   - If column mapping wrong: fix the column rename logic

AFTER FIXING: Re-run notebooks 02 and 04 on a small sample (MAX_SAMPLES=500)
to verify AUROC > 0.50 before scaling up.
""")
