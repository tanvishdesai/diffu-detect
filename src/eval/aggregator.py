"""
DiffuDetect — Results Aggregator

Loads all Parquet score files from the results directory,
aggregates them, computes metrics, and generates publication-ready tables.
"""

import os
import glob
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from .metrics import compute_all_metrics, compute_combined_score, SCORE_DIRECTIONS
from .robustness import (
    compute_robustness_delta,
    compute_per_attack_robustness,
    robustness_report,
    robustness_curve_data,
)


def aggregate_results(
    results_dir: str,
    pattern: str = "scores_*.parquet",
) -> pd.DataFrame:
    """
    Load and merge all score Parquet files from the results directory.

    Each file contributes columns for one (method × model) combination.
    Files are merged on the 'id' column.
    """
    files = sorted(glob.glob(os.path.join(results_dir, pattern)))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} in {results_dir}")

    print(f"[aggregator] Found {len(files)} result files")

    # Load and merge
    dfs = []
    for f in files:
        df = pd.read_parquet(f, engine="pyarrow")
        print(f"  {os.path.basename(f)}: {len(df)} rows, "
              f"cols={[c for c in df.columns if c not in ['id','text','label','generator','domain','dataset','attack']]}")
        dfs.append(df)

    if len(dfs) == 1:
        return dfs[0]

    # Merge all on 'id'
    merged = dfs[0]
    for df in dfs[1:]:
        # Only add new score columns (avoid duplicating metadata)
        meta_cols = {"id", "text", "label", "generator", "domain", "dataset", "attack"}
        new_cols = [c for c in df.columns if c not in meta_cols or c == "id"]
        merged = merged.merge(df[new_cols], on="id", how="outer", suffixes=("", "_dup"))

        # Drop duplicate columns
        dup_cols = [c for c in merged.columns if c.endswith("_dup")]
        merged = merged.drop(columns=dup_cols)

    print(f"[aggregator] Merged: {len(merged)} rows, {len(merged.columns)} columns")
    return merged


def generate_tables(
    df: pd.DataFrame,
    output_dir: str,
    score_columns: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Generate all publication-ready tables:
      1. Clean-text competitiveness (Table 1)
      2. Robustness under attack (Table 2 — the headline)
      3. Per-generator AUROC (Table 3)
      4. Per-attack robustness breakdown (Table 4)
      5. Cost/latency table (Table 5)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Auto-detect score columns
    if score_columns is None:
        score_columns = [c for c in df.columns if c in SCORE_DIRECTIONS]

    if not score_columns:
        print("[aggregator] WARNING: No recognized score columns found!")
        return {}

    tables = {}

    # ── Table 1: Clean-text AUROC ──
    clean_df = df[df.get("attack", "none") == "none"] if "attack" in df.columns else df
    if len(clean_df) > 0:
        table1 = compute_all_metrics(clean_df, score_columns)
        table1 = table1.sort_values("auroc", ascending=False)
        table1.to_csv(os.path.join(output_dir, "table1_clean_auroc.csv"), index=False)
        tables["clean_auroc"] = table1
        print("\n=== Table 1: Clean-Text Detection Performance ===")
        print(table1[["method", "auroc", "tpr_at_1fpr", "tpr_at_5fpr", "accuracy", "f1"]].to_string(index=False))

    # ── Table 2: Robustness ΔAUROC ──
    if "attack" in df.columns:
        attacked_df = df[df["attack"] != "none"]
        if len(clean_df) > 0 and len(attacked_df) > 0:
            table2 = compute_robustness_delta(clean_df, attacked_df, score_columns)
            table2 = table2.sort_values("delta_auroc", ascending=True)
            table2.to_csv(os.path.join(output_dir, "table2_robustness.csv"), index=False)
            tables["robustness"] = table2
            print("\n=== Table 2: Robustness (ΔAUROC) — THE HEADLINE ===")
            print(table2[["method", "auroc_clean", "auroc_attacked", "delta_auroc", "delta_tpr1"]].to_string(index=False))

            # Generate report
            report = robustness_report(table2)
            with open(os.path.join(output_dir, "robustness_report.txt"), "w") as f:
                f.write(report)

    # ── Table 3: Per-generator AUROC ──
    if "generator" in df.columns:
        generators = df[df["label"] == 1]["generator"].unique()
        gen_rows = []
        for gen in generators:
            gen_df = df[(df["label"] == 0) | (df["generator"] == gen)]
            if len(gen_df) < 100:
                continue
            gen_metrics = compute_all_metrics(gen_df, score_columns)
            gen_metrics["generator"] = gen
            gen_rows.append(gen_metrics)

        if gen_rows:
            table3 = pd.concat(gen_rows, ignore_index=True)
            table3.to_csv(os.path.join(output_dir, "table3_per_generator.csv"), index=False)
            tables["per_generator"] = table3
            print("\n=== Table 3: Per-Generator AUROC ===")
            pivot = table3.pivot_table(
                values="auroc", index="method", columns="generator", aggfunc="first"
            )
            print(pivot.to_string())

    # ── Table 4: Per-attack robustness ──
    if "attack" in df.columns:
        table4 = compute_per_attack_robustness(df, score_columns)
        if len(table4) > 0:
            table4.to_csv(os.path.join(output_dir, "table4_per_attack.csv"), index=False)
            tables["per_attack"] = table4

    # ── Table 5: Robustness curves data ──
    if "attack" in df.columns:
        curve_data = robustness_curve_data(df, score_columns)
        if len(curve_data) > 0:
            curve_data.to_csv(os.path.join(output_dir, "robustness_curve_data.csv"), index=False)
            tables["curve_data"] = curve_data

    print(f"\n[aggregator] Saved {len(tables)} tables to {output_dir}")
    return tables


def generate_figures(
    tables: Dict[str, pd.DataFrame],
    output_dir: str,
):
    """Generate publication figures from computed tables."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    os.makedirs(output_dir, exist_ok=True)
    sns.set_style("whitegrid")
    plt.rcParams.update({"font.size": 12, "figure.dpi": 150})

    # ── Figure 1: Clean AUROC bar chart ──
    if "clean_auroc" in tables:
        df = tables["clean_auroc"].copy()
        fig, ax = plt.subplots(figsize=(12, 6))

        # Color-code DiffuDetect vs baselines
        dd_methods = {"mre_mean", "dc_normalized", "dtd_entropy_auc",
                      "dtd_mean_commit_time", "combined_logistic"}
        colors = ["#2196F3" if m in dd_methods else "#9E9E9E" for m in df["method"]]

        bars = ax.barh(df["method"], df["auroc"], color=colors)
        ax.set_xlabel("AUROC")
        ax.set_title("Clean-Text Detection Performance")
        ax.set_xlim(0.4, 1.0)
        ax.axvline(x=0.5, color="red", linestyle="--", alpha=0.5, label="Random")

        # Add value labels
        for bar, val in zip(bars, df["auroc"]):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "fig1_clean_auroc.png"), dpi=150, bbox_inches="tight")
        plt.close()

    # ── Figure 2: Robustness ΔAUROC (THE HEADLINE FIGURE) ──
    if "robustness" in tables:
        df = tables["robustness"].copy()
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Panel A: ΔAUROC
        dd_methods = {"mre_mean", "dc_normalized", "dtd_entropy_auc",
                      "dtd_mean_commit_time", "combined_logistic"}
        colors = ["#2196F3" if m in dd_methods else "#FF5722" for m in df["method"]]

        axes[0].barh(df["method"], df["delta_auroc"], color=colors)
        axes[0].set_xlabel("ΔAUROC (clean → attacked)")
        axes[0].set_title("Robustness: ΔAUROC")
        axes[0].axvline(x=0, color="black", linestyle="-", alpha=0.3)

        # Panel B: Clean vs Attacked AUROC
        x = np.arange(len(df))
        width = 0.35
        axes[1].bar(x - width / 2, df["auroc_clean"], width, label="Clean", color="#4CAF50")
        axes[1].bar(x + width / 2, df["auroc_attacked"], width, label="Attacked", color="#F44336")
        axes[1].set_xlabel("Method")
        axes[1].set_ylabel("AUROC")
        axes[1].set_title("Clean vs Attacked AUROC")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(df["method"], rotation=45, ha="right", fontsize=8)
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "fig2_robustness_headline.png"), dpi=150, bbox_inches="tight")
        plt.close()

    # ── Figure 3: Per-attack AUROC heatmap ──
    if "per_attack" in tables:
        df = tables["per_attack"]
        if len(df) > 0:
            pivot = df.pivot_table(
                values="delta_auroc",
                index="method",
                columns="attack_type",
                aggfunc="first",
            )
            fig, ax = plt.subplots(figsize=(14, 8))
            sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn_r",
                        center=0, ax=ax, linewidths=0.5)
            ax.set_title("ΔAUROC by Method × Attack Type")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "fig3_per_attack_heatmap.png"), dpi=150, bbox_inches="tight")
            plt.close()

    # ── Figure 4: Robustness curves ──
    if "curve_data" in tables:
        df = tables["curve_data"]
        if len(df) > 0:
            fig, ax = plt.subplots(figsize=(12, 6))

            dd_methods = {"mre_mean", "dc_normalized", "dtd_entropy_auc", "combined_logistic"}
            for method in df["method"].unique():
                sub = df[df["method"] == method]
                style = "-" if method in dd_methods else "--"
                ax.plot(sub["attack"], sub["auroc"], marker="o", linestyle=style, label=method)

            ax.set_xlabel("Attack Type")
            ax.set_ylabel("AUROC")
            ax.set_title("Detection AUROC Across Attack Types")
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "fig4_robustness_curves.png"), dpi=150, bbox_inches="tight")
            plt.close()

    print(f"[aggregator] Figures saved to {output_dir}")
