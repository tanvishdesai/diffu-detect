"""
DiffuDetect — RAID Dataset Loader

Loads the RAID dataset (liamdugan/raid on HuggingFace) and structures it
for robustness evaluation. RAID contains many adversarial attack types:
  - paraphrase, synonym substitution, whitespace manipulation,
    homoglyph, number/article editing, etc.

The key insight: RAID provides clean AND attacked versions of the same
texts, enabling paired ΔAUROC measurement — the decisive experiment.
"""

import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import pandas as pd
from datasets import load_dataset

from ..config import DATASETS


# Canonical attack categories in RAID for grouping
RAID_ATTACK_GROUPS = {
    "none": ["none"],
    "paraphrase": ["paraphrase"],
    "synonym": ["synonym"],
    "whitespace": ["whitespace"],
    "homoglyph": ["homoglyph"],
    "number": ["number"],
    "article": ["article"],
    "misspelling": ["misspelling"],
    # Composite groups for analysis
    "all_attacks": None,  # all non-none attacks
    "text_level": ["paraphrase", "synonym"],  # semantic-preserving
    "char_level": ["whitespace", "homoglyph", "misspelling"],  # surface-level
}


@dataclass
class RAIDSlice:
    """A slice of the RAID dataset."""
    name: str
    df: pd.DataFrame
    attack_type: Optional[str] = None
    is_clean: bool = False


def load_raid_raw(
    max_samples: Optional[int] = None,
    cache_dir: Optional[str] = None,
    split: str = "test",
) -> pd.DataFrame:
    """
    Load the raw RAID dataset from HuggingFace.

    Returns a DataFrame with standardized columns:
        id, text, label, generator, domain, dataset, attack
    """
    print("[RAID] Loading dataset from HuggingFace...")
    spec = DATASETS["raid"]

    try:
        ds = load_dataset(spec.hf_repo, cache_dir=cache_dir, trust_remote_code=True)
    except Exception as e:
        print(f"[RAID] Error loading: {e}")
        raise

    # Determine the right split
    if split in ds:
        raw = ds[split]
    elif "train" in ds:
        raw = ds["train"]
    else:
        first_split = list(ds.keys())[0]
        print(f"[RAID] Using split '{first_split}'")
        raw = ds[first_split]

    df = raw.to_pandas()
    print(f"[RAID] Raw dataset: {len(df)} rows, columns: {list(df.columns)}")

    # Standardize column names
    col_map = {}
    for candidate in ["generation", "text", "content"]:
        if candidate in df.columns:
            col_map[candidate] = "text"
            break

    for candidate in ["model", "generator", "source_model"]:
        if candidate in df.columns:
            col_map[candidate] = "generator"
            break

    for candidate in ["attack_name", "attack", "perturbation"]:
        if candidate in df.columns:
            col_map[candidate] = "attack"
            break

    for candidate in ["domain", "source_domain", "category"]:
        if candidate in df.columns:
            col_map[candidate] = "domain"
            break

    df = df.rename(columns=col_map)

    # Binarize label: "human" → 0, everything else → 1
    if "label" not in df.columns:
        if "generator" in df.columns:
            df["label"] = (df["generator"] != "human").astype(int)
        else:
            raise ValueError("Cannot determine labels from RAID data")

    # Ensure columns exist
    if "generator" not in df.columns:
        df["generator"] = df["label"].apply(lambda x: "machine" if x == 1 else "human")
    if "domain" not in df.columns:
        df["domain"] = "unknown"
    if "attack" not in df.columns:
        df["attack"] = "none"

    # Standardize attack names
    df["attack"] = df["attack"].fillna("none").str.lower().str.strip()

    df["dataset"] = "raid"
    df["id"] = [f"raid_{i}" for i in range(len(df))]
    df["label"] = df["label"].astype(int)

    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
        print(f"[RAID] Subsampled to {max_samples} rows")

    print(f"[RAID] Final: {len(df)} rows | "
          f"human={sum(df['label']==0)}, machine={sum(df['label']==1)} | "
          f"attacks={df['attack'].nunique()}: {df['attack'].value_counts().to_dict()}")

    return df


def carve_raid_slices(df: pd.DataFrame) -> Dict[str, RAIDSlice]:
    """
    Carve RAID into evaluation slices by attack type.

    Returns a dict of RAIDSlice objects for each attack type and group.
    """
    slices: Dict[str, RAIDSlice] = {}

    # 1. Clean (no attack) slice
    clean = df[df["attack"] == "none"]
    if len(clean) > 0:
        slices["clean"] = RAIDSlice(name="clean", df=clean, attack_type="none", is_clean=True)

    # 2. Per-attack slices
    attack_types = [a for a in df["attack"].unique() if a != "none"]
    for attack in sorted(attack_types):
        sub = df[df["attack"] == attack]
        if len(sub) >= 50:
            slices[f"attack_{attack}"] = RAIDSlice(
                name=f"attack_{attack}", df=sub, attack_type=attack
            )

    # 3. Grouped slices
    # All attacks combined
    all_attacked = df[df["attack"] != "none"]
    if len(all_attacked) > 0:
        slices["all_attacks"] = RAIDSlice(
            name="all_attacks", df=all_attacked, attack_type="all"
        )

    # Text-level attacks (semantic-preserving)
    text_level_types = {"paraphrase", "synonym"}
    text_level = df[df["attack"].isin(text_level_types)]
    if len(text_level) > 0:
        slices["text_level_attacks"] = RAIDSlice(
            name="text_level_attacks", df=text_level, attack_type="text_level"
        )

    # Character-level attacks
    char_level_types = {"whitespace", "homoglyph", "misspelling"}
    char_level = df[df["attack"].isin(char_level_types)]
    if len(char_level) > 0:
        slices["char_level_attacks"] = RAIDSlice(
            name="char_level_attacks", df=char_level, attack_type="char_level"
        )

    # 4. Per-generator slices (within clean)
    if len(clean) > 0:
        generators = clean[clean["label"] == 1]["generator"].unique()
        for gen in generators:
            sub = clean[(clean["label"] == 0) | (clean["generator"] == gen)]
            if len(sub) >= 100:
                slices[f"clean_gen_{gen}"] = RAIDSlice(
                    name=f"clean_gen_{gen}", df=sub, attack_type="none",
                    is_clean=True
                )

    print(f"[RAID] Carved {len(slices)} evaluation slices:")
    for name, s in slices.items():
        human_n = sum(s.df["label"] == 0)
        machine_n = sum(s.df["label"] == 1)
        flags = "CLEAN" if s.is_clean else f"ATTACK={s.attack_type}"
        print(f"  {name}: {len(s.df)} rows (H={human_n}, M={machine_n}) [{flags}]")

    return slices


def build_clean_attacked_pairs(
    df: pd.DataFrame,
) -> Optional[pd.DataFrame]:
    """
    Build clean↔attacked paired passages for ΔAUROC computation.

    If RAID contains paired data (same generation with and without attack),
    this returns a DataFrame with columns:
        id_clean, id_attacked, attack_type, generator, domain

    If no pairing info is available, returns None (we'll compute unpaired ΔAUROC).
    """
    # Check if there's a pairing column (e.g., generation_id, pair_id)
    pair_cols = [c for c in df.columns if "pair" in c.lower() or "generation_id" in c.lower()]

    if pair_cols:
        pair_col = pair_cols[0]
        clean = df[df["attack"] == "none"].set_index(pair_col)
        attacked = df[df["attack"] != "none"]

        pairs = []
        for _, row in attacked.iterrows():
            pid = row.get(pair_col)
            if pid in clean.index:
                pairs.append({
                    "id_clean": clean.loc[pid, "id"],
                    "id_attacked": row["id"],
                    "attack_type": row["attack"],
                    "generator": row["generator"],
                    "domain": row.get("domain", "unknown"),
                })

        if pairs:
            print(f"[RAID] Built {len(pairs)} clean↔attacked pairs")
            return pd.DataFrame(pairs)

    print("[RAID] No explicit pairing found; will use unpaired ΔAUROC")
    return None


def get_raid_for_scoring(
    max_samples: Optional[int] = 10000,
    cache_dir: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, RAIDSlice]]:
    """
    Convenience: load RAID and return (full_df, slices_dict).
    """
    df = load_raid_raw(max_samples=max_samples, cache_dir=cache_dir)
    slices = carve_raid_slices(df)
    return df, slices
