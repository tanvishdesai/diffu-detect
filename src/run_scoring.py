"""
DiffuDetect — Main Scoring Pipeline

End-to-end script that:
  1. Loads the dataset (MAGE or RAID)
  2. Loads a diffusion model
  3. Runs all three DiffuDetect scorers (MRE, DC, DTD)
  4. Saves results to Parquet

Usage:
    python -m src.run_scoring \
        --dataset mage \
        --model smdm-1.1b \
        --scorers mre dc \
        --max_samples 1000 \
        --output_dir ./results
"""

import argparse
import os
import sys
import time
import torch
import pandas as pd
import numpy as np
from typing import List, Dict, Any

from .config import (
    MODELS, DATASETS, ScoringConfig, MREConfig, DCConfig, DTDConfig,
    RESULTS_DIR, get_device, get_results_path,
)
from .utils import (
    seed_everything, load_masked_diffusion_model,
    save_scores_parquet, truncate_text, Timer,
)
from .data.mage_loader import load_mage_raw, carve_mage_testbeds
from .data.raid_loader import load_raid_raw, carve_raid_slices
from .scorers.mre import MREScorer
from .scorers.dc import DCScorer
from .scorers.dtd import DTDScorer


def run_scoring_pipeline(
    dataset_name: str = "mage",
    model_name: str = "smdm-1.1b",
    scorers_to_run: List[str] = None,
    max_samples: int = 1000,
    output_dir: str = None,
    config: ScoringConfig = None,
    cache_dir: str = None,
):
    """
    Run the full DiffuDetect scoring pipeline.

    Args:
        dataset_name: "mage" or "raid"
        model_name: key from MODELS dict
        scorers_to_run: list of ["mre", "dc", "dtd"]
        max_samples: max passages to score
        output_dir: where to save Parquet results
        config: scoring hyperparameters
        cache_dir: HuggingFace cache directory
    """
    scorers_to_run = scorers_to_run or ["mre"]
    output_dir = output_dir or RESULTS_DIR
    config = config or ScoringConfig()

    seed_everything(config.seed)
    device = get_device()
    print(f"[pipeline] Device: {device}")
    print(f"[pipeline] Dataset: {dataset_name}, Model: {model_name}")
    print(f"[pipeline] Scorers: {scorers_to_run}, Max samples: {max_samples}")

    # ── 1. Load dataset ──────────────────────────────────────────────────────
    with Timer("Dataset loading"):
        if dataset_name == "mage":
            df = load_mage_raw(max_samples=max_samples, cache_dir=cache_dir)
        elif dataset_name == "raid":
            df = load_raid_raw(max_samples=max_samples, cache_dir=cache_dir)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

    print(f"[pipeline] Loaded {len(df)} passages")

    # ── 2. Load model ────────────────────────────────────────────────────────
    model_spec = MODELS[model_name]
    with Timer("Model loading"):
        model, tokenizer = load_masked_diffusion_model(
            hf_repo=model_spec.hf_repo,
            quantize_bits=model_spec.quantize_bits,
            device=device,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )

    # Determine mask token ID
    mask_token_id = None
    if hasattr(tokenizer, 'mask_token_id') and tokenizer.mask_token_id is not None:
        mask_token_id = tokenizer.mask_token_id
    elif hasattr(model.config, 'mask_token_id'):
        mask_token_id = model.config.mask_token_id

    print(f"[pipeline] Mask token ID: {mask_token_id}")

    # ── 3. Run scorers ───────────────────────────────────────────────────────
    texts = df["text"].tolist()
    all_records = []

    if "mre" in scorers_to_run:
        print("\n" + "=" * 60)
        print("Running MRE scorer...")
        print("=" * 60)
        mre_scorer = MREScorer(
            model=model,
            tokenizer=tokenizer,
            config=config.mre,
            device=device,
            mask_token_id=mask_token_id,
        )
        with Timer("MRE scoring"):
            mre_scores = mre_scorer.score_batch(
                texts,
                max_length=config.max_length,
            )

        # Build records for MRE
        for i, (_, row) in enumerate(df.iterrows()):
            for score_key, score_val in mre_scores[i].items():
                record = {
                    "id": row["id"],
                    "text": truncate_text(str(row["text"])),
                    "label": int(row["label"]),
                    "generator": row.get("generator", "unknown"),
                    "domain": row.get("domain", "unknown"),
                    "dataset": row.get("dataset", dataset_name),
                    "attack": row.get("attack", "none"),
                    "method": score_key,
                    "score": float(score_val),
                    "model_used": model_name,
                }
                all_records.append(record)

        # Also save as wide-format
        _save_wide_scores(df, mre_scores, "mre", model_name, dataset_name, output_dir)

    if "dc" in scorers_to_run:
        print("\n" + "=" * 60)
        print("Running DC scorer...")
        print("=" * 60)
        dc_scorer = DCScorer(
            model=model,
            tokenizer=tokenizer,
            config=config.dc,
            device=device,
            mask_token_id=mask_token_id,
        )
        with Timer("DC scoring"):
            dc_scores = dc_scorer.score_batch(
                texts,
                max_length=config.max_length,
            )

        for i, (_, row) in enumerate(df.iterrows()):
            for score_key, score_val in dc_scores[i].items():
                record = {
                    "id": row["id"],
                    "text": truncate_text(str(row["text"])),
                    "label": int(row["label"]),
                    "generator": row.get("generator", "unknown"),
                    "domain": row.get("domain", "unknown"),
                    "dataset": row.get("dataset", dataset_name),
                    "attack": row.get("attack", "none"),
                    "method": score_key,
                    "score": float(score_val),
                    "model_used": model_name,
                }
                all_records.append(record)

        _save_wide_scores(df, dc_scores, "dc", model_name, dataset_name, output_dir)

    if "dtd" in scorers_to_run:
        if not model_spec.supports_iterative:
            print(f"[pipeline] WARNING: {model_name} does not support iterative denoising.")
            print("[pipeline] DTD features may not be meaningful. Proceeding anyway...")

        print("\n" + "=" * 60)
        print("Running DTD scorer...")
        print("=" * 60)
        dtd_scorer = DTDScorer(
            model=model,
            tokenizer=tokenizer,
            config=config.dtd,
            device=device,
            mask_token_id=mask_token_id,
        )
        with Timer("DTD scoring"):
            dtd_scores = dtd_scorer.score_batch(
                texts,
                max_length=config.max_length,
            )

        for i, (_, row) in enumerate(df.iterrows()):
            for score_key, score_val in dtd_scores[i].items():
                record = {
                    "id": row["id"],
                    "text": truncate_text(str(row["text"])),
                    "label": int(row["label"]),
                    "generator": row.get("generator", "unknown"),
                    "domain": row.get("domain", "unknown"),
                    "dataset": row.get("dataset", dataset_name),
                    "attack": row.get("attack", "none"),
                    "method": score_key,
                    "score": float(score_val),
                    "model_used": model_name,
                }
                all_records.append(record)

        _save_wide_scores(df, dtd_scores, "dtd", model_name, dataset_name, output_dir)

    # Save long-format records
    if all_records:
        long_path = os.path.join(output_dir, f"scores_long_{dataset_name}_{model_name}.parquet")
        save_scores_parquet(all_records, long_path)
        print(f"[pipeline] Saved {len(all_records)} long-format records")

    print("\n[pipeline] ✅ Scoring complete!")


def _save_wide_scores(
    df: pd.DataFrame,
    scores_list: List[Dict[str, float]],
    method_prefix: str,
    model_name: str,
    dataset_name: str,
    output_dir: str,
):
    """Save scores in wide format (one row per passage, score columns)."""
    scores_df = pd.DataFrame(scores_list)
    meta_cols = ["id", "text", "label", "generator", "domain", "dataset", "attack"]
    wide_df = df[meta_cols].copy()
    for col in scores_df.columns:
        wide_df[col] = scores_df[col].values

    path = os.path.join(output_dir, f"scores_{dataset_name}_{method_prefix}_{model_name}.parquet")
    os.makedirs(output_dir, exist_ok=True)
    wide_df.to_parquet(path, index=False, engine="pyarrow")
    print(f"[pipeline] Wide-format scores saved → {path}")


def main():
    parser = argparse.ArgumentParser(description="DiffuDetect Scoring Pipeline")
    parser.add_argument("--dataset", type=str, default="mage",
                        choices=["mage", "raid"], help="Dataset to score")
    parser.add_argument("--model", type=str, default="smdm-1.1b",
                        choices=list(MODELS.keys()), help="Model to use")
    parser.add_argument("--scorers", type=str, nargs="+", default=["mre"],
                        choices=["mre", "dc", "dtd"], help="Scorers to run")
    parser.add_argument("--max_samples", type=int, default=1000,
                        help="Max passages to score")
    parser.add_argument("--output_dir", type=str, default=RESULTS_DIR,
                        help="Output directory for results")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="HuggingFace cache directory")
    parser.add_argument("--mask_draws", type=int, default=16,
                        help="Number of random mask draws (K)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Max token length")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()

    config = ScoringConfig(
        mre=MREConfig(num_mask_draws=args.mask_draws),
        dc=DCConfig(num_mask_draws=args.mask_draws),
        dtd=DTDConfig(num_mask_draws=max(4, args.mask_draws // 2)),
        batch_size=args.batch_size,
        max_length=args.max_length,
        seed=args.seed,
    )

    run_scoring_pipeline(
        dataset_name=args.dataset,
        model_name=args.model,
        scorers_to_run=args.scorers,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
        config=config,
        cache_dir=args.cache_dir,
    )


if __name__ == "__main__":
    main()
