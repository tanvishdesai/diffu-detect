"""
DiffuDetect — Robustness Analysis

THE decisive experiment: compute ΔAUROC between clean and attacked text.

Key metric: ΔAUROC = AUROC(clean) - AUROC(attacked)
  - Smaller ΔAUROC = more robust to attacks.
  - If DiffuDetect's ΔAUROC is 8-10 points better than Fast-DetectGPT's,
    that IS the paper.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from .metrics import compute_detection_metrics, SCORE_DIRECTIONS


def compute_robustness_delta(
    clean_df: pd.DataFrame,
    attacked_df: pd.DataFrame,
    score_columns: List[str],
    label_col: str = "label",
) -> pd.DataFrame:
    """
    Compute ΔAUROC and ΔTPR for each method between clean and attacked data.

    Args:
        clean_df: DataFrame with clean (no attack) data
        attacked_df: DataFrame with attacked/paraphrased data
        score_columns: list of score columns to evaluate
        label_col: label column

    Returns:
        DataFrame with columns: method, auroc_clean, auroc_attacked, delta_auroc,
        tpr1_clean, tpr1_attacked, delta_tpr1, etc.
    """
    rows = []

    for score_col in score_columns:
        if score_col not in clean_df.columns or score_col not in attacked_df.columns:
            continue

        direction = SCORE_DIRECTIONS.get(score_col, "higher_is_machine")

        # Clean metrics
        clean_metrics = compute_detection_metrics(
            clean_df[label_col].values,
            clean_df[score_col].values,
            direction,
        )

        # Attacked metrics
        attacked_metrics = compute_detection_metrics(
            attacked_df[label_col].values,
            attacked_df[score_col].values,
            direction,
        )

        row = {
            "method": score_col,
            "auroc_clean": clean_metrics["auroc"],
            "auroc_attacked": attacked_metrics["auroc"],
            "delta_auroc": clean_metrics["auroc"] - attacked_metrics["auroc"],
            "tpr1_clean": clean_metrics["tpr_at_1fpr"],
            "tpr1_attacked": attacked_metrics["tpr_at_1fpr"],
            "delta_tpr1": clean_metrics["tpr_at_1fpr"] - attacked_metrics["tpr_at_1fpr"],
            "tpr5_clean": clean_metrics["tpr_at_5fpr"],
            "tpr5_attacked": attacked_metrics["tpr_at_5fpr"],
            "delta_tpr5": clean_metrics["tpr_at_5fpr"] - attacked_metrics["tpr_at_5fpr"],
            "accuracy_clean": clean_metrics["accuracy"],
            "accuracy_attacked": attacked_metrics["accuracy"],
            "delta_accuracy": clean_metrics["accuracy"] - attacked_metrics["accuracy"],
            "n_clean": clean_metrics["n_samples"],
            "n_attacked": attacked_metrics["n_samples"],
        }
        rows.append(row)

    return pd.DataFrame(rows)


def compute_per_attack_robustness(
    df: pd.DataFrame,
    score_columns: List[str],
    label_col: str = "label",
    attack_col: str = "attack",
) -> pd.DataFrame:
    """
    Compute robustness metrics per attack type.

    Returns a DataFrame with metrics for each (method × attack_type) combination.
    """
    clean_df = df[df[attack_col] == "none"]
    if len(clean_df) == 0:
        print("[robustness] WARNING: No clean data found!")
        return pd.DataFrame()

    attack_types = [a for a in df[attack_col].unique() if a != "none"]

    all_results = []
    for attack in attack_types:
        attacked_df = df[df[attack_col] == attack]
        if len(attacked_df) < 50:
            continue

        delta_df = compute_robustness_delta(
            clean_df, attacked_df, score_columns, label_col
        )
        delta_df["attack_type"] = attack
        all_results.append(delta_df)

    if not all_results:
        return pd.DataFrame()

    return pd.concat(all_results, ignore_index=True)


def robustness_report(
    delta_df: pd.DataFrame,
    diffudetect_methods: Optional[List[str]] = None,
    baseline_methods: Optional[List[str]] = None,
) -> str:
    """
    Generate a human-readable robustness report.

    Highlights the GO/NO-GO decision for Phase 3.
    """
    if diffudetect_methods is None:
        diffudetect_methods = [
            "mre_mean", "dc_normalized",
            "dtd_entropy_auc", "dtd_mean_commit_time",
            "combined_logistic",
        ]

    if baseline_methods is None:
        baseline_methods = [
            "fdgpt_curvature", "dgpt_curvature",
            "bino_score", "cls_log_likelihood",
        ]

    lines = [
        "=" * 80,
        "DIFFUDETECT ROBUSTNESS REPORT — GO/NO-GO #2",
        "=" * 80,
        "",
    ]

    # Summary table
    lines.append("Method                  | AUROC Clean | AUROC Attack | ΔAUROC | ΔTPR@1%")
    lines.append("-" * 80)

    for _, row in delta_df.iterrows():
        method = row["method"]
        marker = "★" if method in diffudetect_methods else " "
        lines.append(
            f"{marker} {method:<22s} | "
            f"{row['auroc_clean']:.4f}      | "
            f"{row['auroc_attacked']:.4f}       | "
            f"{row['delta_auroc']:+.4f} | "
            f"{row['delta_tpr1']:+.4f}"
        )

    lines.append("")

    # Compute average deltas for DiffuDetect vs baselines
    dd_rows = delta_df[delta_df["method"].isin(diffudetect_methods)]
    bl_rows = delta_df[delta_df["method"].isin(baseline_methods)]

    if len(dd_rows) > 0 and len(bl_rows) > 0:
        dd_mean_delta = dd_rows["delta_auroc"].mean()
        bl_mean_delta = bl_rows["delta_auroc"].mean()
        advantage = bl_mean_delta - dd_mean_delta

        lines.append(f"DiffuDetect avg ΔAUROC:  {dd_mean_delta:+.4f}")
        lines.append(f"Baseline avg ΔAUROC:     {bl_mean_delta:+.4f}")
        lines.append(f"Robustness advantage:    {advantage:+.4f}")
        lines.append("")

        # GO/NO-GO decision
        if advantage >= 0.08:
            lines.append("🟢 GO: Robustness advantage ≥ 8 points. THIS IS THE PAPER.")
            lines.append("   → Push to AAAI-27. Lock the headline figure.")
        elif advantage >= 0.04:
            lines.append("🟡 MARGINAL: 4-8 point advantage. Workshop quality.")
            lines.append("   → Consider workshop submission or further optimization.")
        else:
            lines.append("🔴 NO-GO: Advantage < 4 points. Not a strong result.")
            lines.append("   → Demote to workshop or pivot to Proposal 5.")

    report = "\n".join(lines)
    print(report)
    return report


def robustness_curve_data(
    df: pd.DataFrame,
    score_columns: List[str],
    label_col: str = "label",
    attack_col: str = "attack",
    attack_strength_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Prepare data for robustness curves (AUROC vs attack strength/type).

    Returns a DataFrame suitable for plotting with matplotlib/seaborn.
    """
    rows = []
    attack_types = sorted(df[attack_col].unique())

    for attack in attack_types:
        sub = df[df[attack_col] == attack]
        if len(sub) < 50:
            continue

        for score_col in score_columns:
            if score_col not in sub.columns:
                continue

            direction = SCORE_DIRECTIONS.get(score_col, "higher_is_machine")
            metrics = compute_detection_metrics(
                sub[label_col].values,
                sub[score_col].values,
                direction,
            )

            rows.append({
                "attack": attack,
                "method": score_col,
                "auroc": metrics["auroc"],
                "tpr_at_1fpr": metrics["tpr_at_1fpr"],
                "tpr_at_5fpr": metrics["tpr_at_5fpr"],
                "n_samples": metrics["n_samples"],
            })

    return pd.DataFrame(rows)
