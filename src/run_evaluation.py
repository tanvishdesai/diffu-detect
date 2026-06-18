"""
DiffuDetect — Evaluation Pipeline

Reads all scored Parquet files, computes metrics, generates tables & figures.

Usage:
    python -m src.run_evaluation \
        --results_dir ./results \
        --output_dir ./results/analysis
"""

import argparse
import os
import sys
import pandas as pd

from .config import RESULTS_DIR
from .eval.aggregator import aggregate_results, generate_tables, generate_figures
from .eval.metrics import compute_combined_score, SCORE_DIRECTIONS
from .eval.robustness import robustness_report


def run_evaluation(
    results_dir: str = None,
    output_dir: str = None,
):
    """
    Run the full evaluation pipeline: aggregate → metrics → tables → figures.
    """
    results_dir = results_dir or RESULTS_DIR
    output_dir = output_dir or os.path.join(results_dir, "analysis")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("DiffuDetect — Evaluation Pipeline")
    print("=" * 80)

    # ── 1. Aggregate all score files ──
    print("\n[1/4] Aggregating results...")
    try:
        df = aggregate_results(results_dir)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Run scoring pipelines first to generate results.")
        return

    print(f"Total: {len(df)} passages, {len(df.columns)} columns")

    # ── 2. Identify score columns ──
    score_columns = [c for c in df.columns if c in SCORE_DIRECTIONS]
    print(f"\nDetected score columns: {score_columns}")

    if not score_columns:
        print("ERROR: No recognized score columns found in the data!")
        return

    # ── 3. Fit logistic combiner if we have multiple DiffuDetect scores ──
    dd_features = [c for c in score_columns if c.startswith(("mre_", "dc_", "dtd_"))]
    if len(dd_features) >= 2:
        print(f"\n[2/4] Fitting logistic combiner on {len(dd_features)} features...")

        # Use calibration split if available, else use full data
        if "calibration" in df.get("domain", pd.Series()).values:
            cal_df = df[df["domain"] == "calibration"]
            eval_df = df[df["domain"] != "calibration"]
        else:
            # Split 10% for calibration
            cal_df = df.sample(frac=0.1, random_state=42)
            eval_df = df.drop(cal_df.index)

        combined_scores = compute_combined_score(
            df, dd_features,
            label_col="label",
            calibration_df=cal_df if len(cal_df) > 50 else None,
        )
        df["combined_logistic"] = combined_scores
        score_columns.append("combined_logistic")
    else:
        print("\n[2/4] Skipping combiner (need ≥2 DiffuDetect features)")

    # ── 4. Generate tables ──
    print("\n[3/4] Generating tables...")
    tables = generate_tables(df, output_dir, score_columns)

    # ── 5. Generate figures ──
    print("\n[4/4] Generating figures...")
    try:
        generate_figures(tables, os.path.join(output_dir, "figures"))
    except Exception as e:
        print(f"[eval] Figure generation failed (non-critical): {e}")

    # ── Save the full aggregated DataFrame ──
    full_path = os.path.join(output_dir, "full_aggregated_scores.parquet")
    df.to_parquet(full_path, index=False, engine="pyarrow")
    print(f"\n[eval] Full aggregated data saved → {full_path}")

    # ── Summary ──
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"Tables generated: {list(tables.keys())}")
    if "clean_auroc" in tables:
        best = tables["clean_auroc"].iloc[0]
        print(f"Best clean AUROC: {best['method']} = {best['auroc']:.4f}")
    if "robustness" in tables:
        best_robust = tables["robustness"].sort_values("delta_auroc").iloc[0]
        print(f"Most robust (smallest ΔAUROC): {best_robust['method']} = {best_robust['delta_auroc']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="DiffuDetect Evaluation Pipeline")
    parser.add_argument("--results_dir", type=str, default=RESULTS_DIR,
                        help="Directory containing score Parquet files")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for tables/figures (default: results_dir/analysis)")

    args = parser.parse_args()
    run_evaluation(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
