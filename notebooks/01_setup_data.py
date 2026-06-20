"""
==========================================================================
  DiffuDetect — Kaggle Notebook 1: Environment Setup & Data Preparation
==========================================================================

PURPOSE:
  - Install dependencies
  - Load MAGE and RAID datasets
  - Carve evaluation slices
  - Save preprocessed data as Kaggle Dataset artifacts

RUN THIS FIRST before any scoring notebooks.

KAGGLE SETTINGS:
  - GPU: None needed (CPU is fine)
  - Internet: ON (to download datasets)
  - Accelerator: None
  - Persistence: Save outputs as Dataset
"""

# ─── Cell 1: Install dependencies ────────────────────────────────────────────
# !pip install -q torch transformers datasets accelerate bitsandbytes \
#     scikit-learn pandas pyarrow matplotlib seaborn tqdm huggingface_hub \
#     sentencepiece protobuf

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path

# ─── Cell 2: Setup paths ─────────────────────────────────────────────────────

# Kaggle working directory
WORK_DIR = "/kaggle/working"
DATA_DIR = os.path.join(WORK_DIR, "data")
RESULTS_DIR = os.path.join(WORK_DIR, "results")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Add src to path if running from the repo
sys.path.insert(0, "/kaggle/input/diffudetect")  # If uploaded as a dataset
sys.path.insert(0, ".")

print(f"Working directory: {WORK_DIR}")
print(f"Data directory: {DATA_DIR}")

# ─── Cell 3: Load MAGE dataset ───────────────────────────────────────────────

from datasets import load_dataset

print("Loading MAGE dataset...")
mage_ds = load_dataset("yaful/MAGE")
print(f"MAGE splits: {list(mage_ds.keys())}")

# Convert to pandas
for split_name in mage_ds:
    df = mage_ds[split_name].to_pandas()
    print(f"  {split_name}: {len(df)} rows, columns: {list(df.columns)}")

# Use the test split (or first available)
if "test" in mage_ds:
    mage_df = mage_ds["test"].to_pandas()
elif "train" in mage_ds:
    mage_df = mage_ds["train"].to_pandas()
else:
    first_split = list(mage_ds.keys())[0]
    mage_df = mage_ds[first_split].to_pandas()

print(f"\nMAGE selected: {len(mage_df)} rows")
print(f"Columns: {list(mage_df.columns)}")
print(f"Label distribution (RAW, before flip):\n{mage_df['label'].value_counts()}")
if "src" in mage_df.columns:
    print(f"\nGenerators (src column):\n{mage_df['src'].value_counts().head(20)}")

# ─── Cell 4: Standardize MAGE columns ────────────────────────────────────────

# ======================================================================
# FIX: MAGE label convention is INVERTED:
#   label=0 → machine-generated,  label=1 → human-written
# We flip to our canonical format:
#   label=0 → human,  label=1 → machine
# ======================================================================
mage_df["label"] = 1 - mage_df["label"].astype(int)
print("\nMAGE labels FLIPPED: now label=0→human, label=1→machine")
print(f"Label distribution (after flip):\n{mage_df['label'].value_counts()}")

# FIX: MAGE uses 'src' column (not 'source_model' or 'category')
if "src" in mage_df.columns:
    mage_df["generator"] = mage_df["src"]
    # Extract domain from src (format: domain_human or domain_machine_...)
    mage_df["domain"] = mage_df["src"].str.split("_").str[0]
else:
    # Fallback for other column names
    rename_map = {}
    for src_col, dst_col in [
        ("text", "text"), ("content", "text"),
        ("source_model", "generator"), ("model", "generator"),
        ("category", "domain"),
    ]:
        if src_col in mage_df.columns and dst_col not in rename_map.values():
            rename_map[src_col] = dst_col
    mage_df = mage_df.rename(columns=rename_map)

# Ensure required columns
if "generator" not in mage_df.columns:
    mage_df["generator"] = mage_df["label"].apply(lambda x: "machine" if x == 1 else "human")
if "domain" not in mage_df.columns:
    mage_df["domain"] = "unknown"

mage_df["dataset"] = "mage"
mage_df["attack"] = "none"
mage_df["id"] = [f"mage_{i}" for i in range(len(mage_df))]

print(f"\nStandardized MAGE: {len(mage_df)} rows")
print(f"Columns: {list(mage_df.columns)}")

# ─── Cell 5: Carve MAGE testbed slices ───────────────────────────────────────

slices_info = {}

# Full dataset
slices_info["full"] = {"n": len(mage_df), "n_human": sum(mage_df["label"]==0), "n_machine": sum(mage_df["label"]==1)}

# Per-domain
domains = [d for d in mage_df["domain"].unique() if d != "unknown"]
for domain in domains:
    sub = mage_df[mage_df["domain"] == domain]
    if len(sub) >= 100:
        slices_info[f"domain_{domain}"] = {
            "n": len(sub),
            "n_human": sum(sub["label"]==0),
            "n_machine": sum(sub["label"]==1),
        }

# Per-generator
generators = mage_df[mage_df["label"] == 1]["generator"].unique()
for gen in generators:
    sub = mage_df[(mage_df["label"] == 0) | (mage_df["generator"] == gen)]
    if len(sub) >= 100:
        slices_info[f"gen_{gen}"] = {
            "n": len(sub),
            "n_human": sum(sub["label"]==0),
            "n_machine": sum(sub["label"]==1),
        }

# Calibration split (500 samples, balanced)
cal_size = min(500, len(mage_df) // 10)
cal_human = mage_df[mage_df["label"] == 0].sample(n=min(cal_size//2, sum(mage_df["label"]==0)), random_state=42)
cal_machine = mage_df[mage_df["label"] == 1].sample(n=min(cal_size//2, sum(mage_df["label"]==1)), random_state=42)
cal_df = pd.concat([cal_human, cal_machine]).reset_index(drop=True)
cal_df.to_parquet(os.path.join(DATA_DIR, "mage_calibration.parquet"), index=False)

print(f"\nTestbed slices:")
for name, info in slices_info.items():
    print(f"  {name}: {info['n']} total (H={info['n_human']}, M={info['n_machine']})")

# ─── Cell 6: Load RAID dataset ───────────────────────────────────────────────

print("\nLoading RAID dataset...")
try:
    raid_ds = load_dataset("liamdugan/raid", trust_remote_code=True)
    print(f"RAID splits: {list(raid_ds.keys())}")

    if "test" in raid_ds:
        raid_df = raid_ds["test"].to_pandas()
    elif "train" in raid_ds:
        raid_df = raid_ds["train"].to_pandas()
    else:
        first_split = list(raid_ds.keys())[0]
        raid_df = raid_ds[first_split].to_pandas()

    print(f"RAID: {len(raid_df)} rows")
    print(f"Columns: {list(raid_df.columns)}")

    # Standardize
    rename_map = {}
    for src, dst in [
        ("generation", "text"), ("text", "text"),
        ("model", "generator"), ("generator", "generator"),
        ("attack_name", "attack"), ("attack", "attack"),
        ("domain", "domain"), ("source_domain", "domain"),
    ]:
        if src in raid_df.columns and dst not in rename_map.values():
            rename_map[src] = dst

    raid_df = raid_df.rename(columns=rename_map)

    if "label" not in raid_df.columns:
        if "generator" in raid_df.columns:
            raid_df["label"] = (raid_df["generator"] != "human").astype(int)

    if "domain" not in raid_df.columns:
        raid_df["domain"] = "unknown"
    if "attack" not in raid_df.columns:
        raid_df["attack"] = "none"

    raid_df["attack"] = raid_df["attack"].fillna("none").str.lower().str.strip()
    raid_df["dataset"] = "raid"
    raid_df["id"] = [f"raid_{i}" for i in range(len(raid_df))]
    raid_df["label"] = raid_df["label"].astype(int)

    print(f"\nRAID attacks distribution:")
    print(raid_df["attack"].value_counts())

    raid_df.to_parquet(os.path.join(DATA_DIR, "raid_full.parquet"), index=False)
    print(f"Saved RAID → {DATA_DIR}/raid_full.parquet")

except Exception as e:
    print(f"WARNING: Could not load RAID: {e}")
    print("Continuing with MAGE only.")
    raid_df = None

# ─── Cell 7: Save MAGE data ──────────────────────────────────────────────────

mage_df.to_parquet(os.path.join(DATA_DIR, "mage_full.parquet"), index=False)
print(f"\nSaved MAGE → {DATA_DIR}/mage_full.parquet")

# Save a smaller subsample for quick testing
quick_n = min(2000, len(mage_df))
quick_df = mage_df.groupby("label", group_keys=False).apply(
    lambda x: x.sample(n=min(quick_n//2, len(x)), random_state=42)
).reset_index(drop=True)
quick_df.to_parquet(os.path.join(DATA_DIR, "mage_quick.parquet"), index=False)
print(f"Saved MAGE quick subset ({len(quick_df)} rows) → {DATA_DIR}/mage_quick.parquet")

# Save slice info
with open(os.path.join(DATA_DIR, "slices_info.json"), "w") as f:
    json.dump(slices_info, f, indent=2)

# ─── Cell 8: Summary ─────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("DATA PREPARATION COMPLETE")
print("=" * 60)
print(f"MAGE: {len(mage_df)} passages ({sum(mage_df['label']==0)} human, {sum(mage_df['label']==1)} machine)")
if raid_df is not None:
    print(f"RAID: {len(raid_df)} passages ({sum(raid_df['label']==0)} human, {sum(raid_df['label']==1)} machine)")
    print(f"  Attack types: {sorted(raid_df['attack'].unique())}")
print(f"\nFiles saved to: {DATA_DIR}")
print(f"  - mage_full.parquet")
print(f"  - mage_quick.parquet")
print(f"  - mage_calibration.parquet")
if raid_df is not None:
    print(f"  - raid_full.parquet")
print(f"  - slices_info.json")
print(f"\nNext: Run Notebook 2 (MRE scoring) with the data as a Kaggle Dataset input.")
