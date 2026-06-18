"""
DiffuDetect — MAGE Dataset Loader

Loads the MAGE dataset (yaful/MAGE on HuggingFace) and carves it into:
  - 6 in-distribution testbeds (by domain × generator)
  - 2 wild OOD sets (including the paraphrase set)
  - A calibration split for the logistic combiner

MAGE structure (key columns):
  - text: the passage
  - label: 0 = human, 1 = machine
  - category: domain (e.g., "writing_prompts", "news", "reviews", etc.)
  - source_model: which LLM generated the text (e.g., "chatgpt", "davinci", etc.)
"""

import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from ..config import DATASETS, RESULTS_DIR


@dataclass
class MAGESlice:
    """A slice of the MAGE dataset with metadata."""
    name: str
    df: pd.DataFrame
    domain: Optional[str] = None
    generator: Optional[str] = None
    is_paraphrase: bool = False
    is_ood: bool = False


def load_mage_raw(
    max_samples: Optional[int] = None,
    cache_dir: Optional[str] = None,
    split: str = "test",
) -> pd.DataFrame:
    """
    Load the raw MAGE dataset from HuggingFace.

    Returns a DataFrame with standardized columns:
        id, text, label, generator, domain, dataset, attack
    """
    print("[MAGE] Loading dataset from HuggingFace...")
    spec = DATASETS["mage"]

    try:
        ds = load_dataset(spec.hf_repo, cache_dir=cache_dir, trust_remote_code=True)
    except Exception as e:
        print(f"[MAGE] Error loading from HF: {e}")
        print("[MAGE] Attempting to load from local cache or Kaggle dataset...")
        raise

    # MAGE may have train/test/validation splits or a single split
    if split in ds:
        raw = ds[split]
    elif "test" in ds:
        raw = ds["test"]
    else:
        # Use the first available split
        first_split = list(ds.keys())[0]
        print(f"[MAGE] Using split '{first_split}'")
        raw = ds[first_split]

    df = raw.to_pandas()
    print(f"[MAGE] Raw dataset: {len(df)} rows, columns: {list(df.columns)}")

    # Standardize column names
    col_map = {}
    # text column
    for candidate in ["text", "content", "passage", "document"]:
        if candidate in df.columns:
            col_map[candidate] = "text"
            break

    # label column
    for candidate in ["label", "is_machine", "is_ai"]:
        if candidate in df.columns:
            col_map[candidate] = "label"
            break

    # generator column
    for candidate in ["source_model", "model", "generator", "source"]:
        if candidate in df.columns:
            col_map[candidate] = "generator"
            break

    # domain column
    for candidate in ["category", "domain", "source_domain", "task"]:
        if candidate in df.columns:
            col_map[candidate] = "domain"
            break

    df = df.rename(columns=col_map)

    # Ensure required columns exist
    if "generator" not in df.columns:
        df["generator"] = df["label"].apply(lambda x: "machine" if x == 1 else "human")
    if "domain" not in df.columns:
        df["domain"] = "unknown"

    # Add standard columns
    df["dataset"] = "mage"
    df["attack"] = "none"

    # Create unique IDs
    df["id"] = [f"mage_{i}" for i in range(len(df))]

    # Ensure label is binary int
    df["label"] = df["label"].astype(int)

    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
        print(f"[MAGE] Subsampled to {max_samples} rows")

    # Keep only the columns we need
    keep_cols = ["id", "text", "label", "generator", "domain", "dataset", "attack"]
    extra_cols = [c for c in df.columns if c not in keep_cols]
    df = df[keep_cols + [c for c in extra_cols if c in df.columns]]

    print(f"[MAGE] Final: {len(df)} rows | "
          f"human={sum(df['label']==0)}, machine={sum(df['label']==1)} | "
          f"generators={df['generator'].nunique()} | domains={df['domain'].nunique()}")

    return df


def carve_mage_testbeds(df: pd.DataFrame) -> Dict[str, MAGESlice]:
    """
    Carve the MAGE dataset into evaluation testbeds.

    Returns a dict of named MAGESlice objects:
      - Per-domain slices
      - Per-generator slices
      - Cross-domain/cross-generator slices
      - Paraphrase / OOD slices (if identifiable)
      - A calibration split
    """
    slices: Dict[str, MAGESlice] = {}

    # 1. Full test set
    slices["full"] = MAGESlice(name="full", df=df)

    # 2. Per-domain slices
    domains = df["domain"].unique()
    for domain in domains:
        if domain == "unknown":
            continue
        sub = df[df["domain"] == domain]
        if len(sub) >= 100:  # only if meaningful size
            slices[f"domain_{domain}"] = MAGESlice(
                name=f"domain_{domain}", df=sub, domain=domain
            )

    # 3. Per-generator slices
    generators = df[df["label"] == 1]["generator"].unique()
    for gen in generators:
        # Include all human samples + just this generator's machine samples
        sub = df[(df["label"] == 0) | (df["generator"] == gen)]
        if len(sub) >= 100:
            slices[f"gen_{gen}"] = MAGESlice(
                name=f"gen_{gen}", df=sub, generator=gen
            )

    # 4. Try to identify paraphrase/OOD slices
    # MAGE includes "wild" OOD sets that may contain paraphrase data
    if "attack" in df.columns:
        para = df[df["attack"] != "none"]
        if len(para) > 0:
            slices["paraphrase"] = MAGESlice(
                name="paraphrase", df=para, is_paraphrase=True
            )

    # Check for paraphrase markers in domain or other columns
    for col in ["domain", "generator"]:
        for val in df[col].unique():
            val_lower = str(val).lower()
            if any(kw in val_lower for kw in ["paraph", "rewrite", "ood", "wild"]):
                sub = df[df[col] == val]
                if len(sub) >= 50:
                    slice_name = f"ood_{val}"
                    slices[slice_name] = MAGESlice(
                        name=slice_name, df=sub, is_ood=True
                    )

    # 5. Calibration split (small, balanced, for logistic combiner)
    cal_size = min(500, len(df) // 10)
    if cal_size >= 100:
        cal_human = df[df["label"] == 0].sample(
            n=min(cal_size // 2, sum(df["label"] == 0)), random_state=42
        )
        cal_machine = df[df["label"] == 1].sample(
            n=min(cal_size // 2, sum(df["label"] == 1)), random_state=42
        )
        cal_df = pd.concat([cal_human, cal_machine]).reset_index(drop=True)

        # The "eval" set is everything NOT in calibration
        cal_ids = set(cal_df["id"])
        eval_df = df[~df["id"].isin(cal_ids)].reset_index(drop=True)

        slices["calibration"] = MAGESlice(name="calibration", df=cal_df)
        slices["eval_no_cal"] = MAGESlice(name="eval_no_cal", df=eval_df)

    print(f"[MAGE] Carved {len(slices)} testbed slices:")
    for name, s in slices.items():
        human_n = sum(s.df["label"] == 0)
        machine_n = sum(s.df["label"] == 1)
        flags = []
        if s.is_paraphrase:
            flags.append("PARAPHRASE")
        if s.is_ood:
            flags.append("OOD")
        flag_str = f" [{', '.join(flags)}]" if flags else ""
        print(f"  {name}: {len(s.df)} rows (H={human_n}, M={machine_n}){flag_str}")

    return slices


def get_mage_for_scoring(
    max_samples: Optional[int] = 5000,
    cache_dir: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, MAGESlice]]:
    """
    Convenience function: load MAGE and return (full_df, slices_dict).
    """
    df = load_mage_raw(max_samples=max_samples, cache_dir=cache_dir)
    slices = carve_mage_testbeds(df)
    return df, slices
