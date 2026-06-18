"""
DiffuDetect — Baseline Scoring Pipeline

Runs all AR-based baselines:
  1. Fast-DetectGPT (primary baseline)
  2. DetectGPT (weak OOD reference)
  3. Binoculars (strong zero-shot)
  4. Classical (log-lik, rank, entropy)

Usage:
    python -m src.run_baselines \
        --dataset mage \
        --baselines fast_detectgpt classical \
        --max_samples 1000 \
        --output_dir ./results
"""

import argparse
import os
import sys
import torch
import pandas as pd
import numpy as np
from typing import List, Dict

from .config import (
    MODELS, DATASETS, RESULTS_DIR, get_device,
)
from .utils import (
    seed_everything, load_model_and_tokenizer,
    save_scores_parquet, truncate_text, Timer,
)
from .data.mage_loader import load_mage_raw
from .data.raid_loader import load_raid_raw
from .baselines.fast_detectgpt import FastDetectGPTScorer
from .baselines.detectgpt import DetectGPTScorer
from .baselines.binoculars import BinocularsScorer
from .baselines.classical import ClassicalScorer


def run_baselines_pipeline(
    dataset_name: str = "mage",
    baselines_to_run: List[str] = None,
    max_samples: int = 1000,
    output_dir: str = None,
    cache_dir: str = None,
    seed: int = 42,
    max_length: int = 512,
    num_perturbations: int = 50,
):
    """
    Run all baseline detectors.
    """
    baselines_to_run = baselines_to_run or ["classical"]
    output_dir = output_dir or RESULTS_DIR
    os.makedirs(output_dir, exist_ok=True)

    seed_everything(seed)
    device = get_device()

    print(f"[baselines] Device: {device}")
    print(f"[baselines] Dataset: {dataset_name}")
    print(f"[baselines] Baselines: {baselines_to_run}")

    # ── Load dataset ──
    with Timer("Dataset loading"):
        if dataset_name == "mage":
            df = load_mage_raw(max_samples=max_samples, cache_dir=cache_dir)
        elif dataset_name == "raid":
            df = load_raid_raw(max_samples=max_samples, cache_dir=cache_dir)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

    texts = df["text"].tolist()
    meta_cols = ["id", "text", "label", "generator", "domain", "dataset", "attack"]

    # ── Classical baselines ──
    if "classical" in baselines_to_run:
        print("\n" + "=" * 60)
        print("Running Classical baselines (log-lik, rank, entropy)...")
        print("=" * 60)

        ar_spec = MODELS["gpt-neo-2.7b"]
        with Timer("AR model loading"):
            ar_model, ar_tokenizer = load_model_and_tokenizer(
                hf_repo=ar_spec.hf_repo,
                quantize_bits=ar_spec.quantize_bits,
                device=device,
                cache_dir=cache_dir,
            )

        scorer = ClassicalScorer(ar_model, ar_tokenizer, device=device)
        with Timer("Classical scoring"):
            scores = scorer.score_batch(texts, max_length=max_length)

        _save_baseline_scores(df, scores, "classical", "gpt-neo-2.7b", dataset_name, output_dir, meta_cols)

        # Free memory
        del ar_model, scorer
        torch.cuda.empty_cache()

    # ── Fast-DetectGPT ──
    if "fast_detectgpt" in baselines_to_run:
        print("\n" + "=" * 60)
        print("Running Fast-DetectGPT...")
        print("=" * 60)

        ar_spec = MODELS["gpt-neo-2.7b"]
        with Timer("AR model loading"):
            ar_model, ar_tokenizer = load_model_and_tokenizer(
                hf_repo=ar_spec.hf_repo,
                quantize_bits=ar_spec.quantize_bits,
                device=device,
                cache_dir=cache_dir,
            )

        scorer = FastDetectGPTScorer(
            scoring_model=ar_model,
            scoring_tokenizer=ar_tokenizer,
            num_perturbations=num_perturbations,
            device=device,
        )
        with Timer("Fast-DetectGPT scoring"):
            scores = scorer.score_batch(texts, max_length=max_length)

        _save_baseline_scores(df, scores, "fast_detectgpt", "gpt-neo-2.7b", dataset_name, output_dir, meta_cols)

        del ar_model, scorer
        torch.cuda.empty_cache()

    # ── DetectGPT ──
    if "detectgpt" in baselines_to_run:
        print("\n" + "=" * 60)
        print("Running DetectGPT...")
        print("=" * 60)

        ar_spec = MODELS["gpt-neo-2.7b"]
        with Timer("AR model loading"):
            ar_model, ar_tokenizer = load_model_and_tokenizer(
                hf_repo=ar_spec.hf_repo,
                quantize_bits=ar_spec.quantize_bits,
                device=device,
                cache_dir=cache_dir,
            )

        # Optionally load T5 for perturbation
        t5_model = None
        t5_tokenizer = None
        try:
            from transformers import T5ForConditionalGeneration, T5Tokenizer
            print("[baselines] Loading T5-small for perturbation...")
            t5_tokenizer = T5Tokenizer.from_pretrained("t5-small", cache_dir=cache_dir)
            t5_model = T5ForConditionalGeneration.from_pretrained(
                "t5-small", cache_dir=cache_dir
            ).to(device)
            t5_model.eval()
        except Exception as e:
            print(f"[baselines] T5 not available, using word-shuffle fallback: {e}")

        scorer = DetectGPTScorer(
            scoring_model=ar_model,
            scoring_tokenizer=ar_tokenizer,
            perturbation_model=t5_model,
            perturbation_tokenizer=t5_tokenizer,
            num_perturbations=min(num_perturbations, 25),
            device=device,
        )
        with Timer("DetectGPT scoring"):
            scores = scorer.score_batch(texts, max_length=max_length)

        _save_baseline_scores(df, scores, "detectgpt", "gpt-neo-2.7b", dataset_name, output_dir, meta_cols)

        del ar_model, t5_model, scorer
        torch.cuda.empty_cache()

    # ── Binoculars ──
    if "binoculars" in baselines_to_run:
        print("\n" + "=" * 60)
        print("Running Binoculars...")
        print("=" * 60)

        obs_spec = MODELS["falcon-7b-instruct"]
        perf_spec = MODELS["falcon-7b"]

        with Timer("Binoculars model loading"):
            obs_model, obs_tokenizer = load_model_and_tokenizer(
                hf_repo=obs_spec.hf_repo,
                quantize_bits=obs_spec.quantize_bits,
                device=device,
                cache_dir=cache_dir,
            )
            perf_model, perf_tokenizer = load_model_and_tokenizer(
                hf_repo=perf_spec.hf_repo,
                quantize_bits=perf_spec.quantize_bits,
                device=device,
                cache_dir=cache_dir,
            )

        scorer = BinocularsScorer(
            observer_model=obs_model,
            observer_tokenizer=obs_tokenizer,
            performer_model=perf_model,
            performer_tokenizer=perf_tokenizer,
            device=device,
        )
        with Timer("Binoculars scoring"):
            scores = scorer.score_batch(texts, max_length=max_length)

        _save_baseline_scores(df, scores, "binoculars", "falcon-7b", dataset_name, output_dir, meta_cols)

        del obs_model, perf_model, scorer
        torch.cuda.empty_cache()

    print("\n[baselines] ✅ All baselines complete!")


def _save_baseline_scores(
    df: pd.DataFrame,
    scores_list: List[Dict[str, float]],
    method_name: str,
    model_name: str,
    dataset_name: str,
    output_dir: str,
    meta_cols: List[str],
):
    """Save baseline scores in wide format."""
    scores_df = pd.DataFrame(scores_list)
    wide_df = df[meta_cols].copy()
    for col in scores_df.columns:
        wide_df[col] = scores_df[col].values

    path = os.path.join(output_dir, f"scores_{dataset_name}_{method_name}_{model_name}.parquet")
    wide_df.to_parquet(path, index=False, engine="pyarrow")
    print(f"[baselines] Saved → {path}")


def main():
    parser = argparse.ArgumentParser(description="DiffuDetect Baseline Pipeline")
    parser.add_argument("--dataset", type=str, default="mage",
                        choices=["mage", "raid"])
    parser.add_argument("--baselines", type=str, nargs="+", default=["classical"],
                        choices=["classical", "fast_detectgpt", "detectgpt", "binoculars"])
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default=RESULTS_DIR)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--num_perturbations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    run_baselines_pipeline(
        dataset_name=args.dataset,
        baselines_to_run=args.baselines,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        seed=args.seed,
        max_length=args.max_length,
        num_perturbations=args.num_perturbations,
    )


if __name__ == "__main__":
    main()
