"""
==========================================================================
  DiffuDetect — Notebook 08: Head-to-Head MRE vs FDGPT Comparison
==========================================================================

PURPOSE:
  - Load MRE scores (from Notebook 02) and FDGPT/Classical scores (from Notebook 04)
  - Compute per-generator AUROC for BOTH methods
  - Scatter-plot MRE-AUROC vs FDGPT-AUROC to show complementarity
  - Train a logistic combiner and measure combined AUROC
  - This produces THE key figure for the paper

KAGGLE SETTINGS:
  - GPU: None (CPU is fine)
  - Internet: OFF
  - Attach as input datasets:
      - Notebook 02 output (MRE scores)
      - Notebook 04 output (baseline scores)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

# ─── Config ──────────────────────────────────────────────────────────────────

RESULTS_DIR = "/kaggle/working/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Paths to score files from previous notebooks
# Adjust these paths based on your Kaggle dataset inputs
MRE_PATHS = [
    "/kaggle/input/datasets/vasuaashadesai/diffudetect-mre-scores/results/scores_mage_mre_smdm-1.1b.parquet",
    "/kaggle/working/results/scores_mage_mre_smdm-1.1b.parquet",
]
BASELINE_PATHS = [
    "/kaggle/input/datasets/shilpavdesai/04-baseline-scoring-scores/results/scores_mage_classical_gpt-neo-2.7b.parquet",
    "/kaggle/working/results/scores_mage_classical_gpt-neo-2.7b.parquet",
]
FDGPT_PATHS = [
    "/kaggle/input/datasets/shilpavdesai/04-baseline-scoring-scores/results/scores_mage_fast_detectgpt_gpt-neo-2.7b.parquet",
    "/kaggle/working/results/scores_mage_fast_detectgpt_gpt-neo-2.7b.parquet",
]


def find_file(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None

# ─── Load scores ─────────────────────────────────────────────────────────────

print("Loading score files...")

mre_file = find_file(MRE_PATHS)
baseline_file = find_file(BASELINE_PATHS)
fdgpt_file = find_file(FDGPT_PATHS)

if mre_file is None:
    raise FileNotFoundError("MRE scores not found. Run Notebook 02 first.")

df = pd.read_parquet(mre_file)
print(f"MRE scores: {len(df)} rows from {mre_file}")

# Merge baseline scores if available
if baseline_file:
    cls_df = pd.read_parquet(baseline_file)
    cls_cols = [c for c in cls_df.columns if c.startswith("cls_")]
    if "id" in cls_df.columns and "id" in df.columns:
        df = df.merge(cls_df[["id"] + cls_cols], on="id", how="left")
    else:
        # Fallback: align by index
        for c in cls_cols:
            if len(cls_df) == len(df):
                df[c] = cls_df[c].values
    print(f"  Merged classical scores: {cls_cols}")

if fdgpt_file:
    fdgpt_df = pd.read_parquet(fdgpt_file)
    fdgpt_cols = [c for c in fdgpt_df.columns if c.startswith("fdgpt_")]
    if "id" in fdgpt_df.columns and "id" in df.columns:
        df = df.merge(fdgpt_df[["id"] + fdgpt_cols], on="id", how="left")
    else:
        for c in fdgpt_cols:
            if len(fdgpt_df) == len(df):
                df[c] = fdgpt_df[c].values
    print(f"  Merged FDGPT scores: {fdgpt_cols}")

print(f"\nFinal merged dataset: {len(df)} rows")
print(f"Columns: {list(df.columns)}")
print(f"Labels: {df['label'].value_counts().to_dict()}")

# =========================================================================
#  PART 1: Per-Generator AUROC Comparison
# =========================================================================

print("\n" + "=" * 70)
print("PART 1: PER-GENERATOR AUROC — MRE vs FDGPT vs CLASSICAL")
print("=" * 70)

def best_auroc(labels, scores):
    """Auto-detect direction and return best AUROC."""
    valid = np.isfinite(scores)
    if valid.sum() < 20:
        return np.nan
    try:
        a1 = roc_auc_score(labels[valid], scores[valid])
        a2 = roc_auc_score(labels[valid], -scores[valid])
        return max(a1, a2)
    except ValueError:
        return np.nan

# Methods to compare
methods = {}
if "mre_mean" in df.columns:
    methods["MRE (SMDM)"] = "mre_mean"
if "fdgpt_curvature" in df.columns:
    methods["Fast-DetectGPT"] = "fdgpt_curvature"
if "cls_mean_rank" in df.columns:
    methods["Classical Rank"] = "cls_mean_rank"
if "cls_log_likelihood" in df.columns:
    methods["Classical LL"] = "cls_log_likelihood"

# Extract generator from 'generator' or 'src' column
gen_col = None
for c in ["generator", "src"]:
    if c in df.columns and df[c].nunique() > 1:
        gen_col = c
        break

if gen_col is None:
    print("WARNING: No generator column found. Skipping per-generator analysis.")
else:
    # Only use machine generators (generators that have label=1 samples)
    machine_gens = df[df["label"] == 1][gen_col].unique()
    print(f"Found {len(machine_gens)} machine generators")

    rows = []
    for gen in machine_gens:
        # Slice: all humans + this generator's machine text
        gen_mask = (df["label"] == 0) | (df[gen_col] == gen)
        sub = df[gen_mask]

        if sub["label"].nunique() < 2 or len(sub) < 50:
            continue

        row = {"generator": gen, "n": len(sub)}

        # Extract model family and domain from generator name
        parts = gen.split("_")
        row["domain"] = parts[0] if parts else "unknown"

        # Determine model family
        gen_lower = gen.lower()
        if "llama" in gen_lower or any(f"_{s}" in gen_lower for s in ["7b", "13b", "30b", "65b"]):
            if "13b" in gen_lower:
                row["model_family"] = "LLaMA-13B"
            elif "30b" in gen_lower:
                row["model_family"] = "LLaMA-30B"
            elif "65b" in gen_lower:
                row["model_family"] = "LLaMA-65B"
            elif "7b" in gen_lower and "bloom" not in gen_lower and "opt" not in gen_lower:
                row["model_family"] = "LLaMA-7B"
            else:
                row["model_family"] = gen_lower.split("_")[-1] if "_" in gen_lower else "other"
        elif "gpt-3.5" in gen_lower or "gpt_3.5" in gen_lower:
            row["model_family"] = "GPT-3.5"
        elif "gpt4" in gen_lower:
            row["model_family"] = "GPT-4"
        elif "davinci-002" in gen_lower:
            row["model_family"] = "text-davinci-002"
        elif "davinci-003" in gen_lower:
            row["model_family"] = "text-davinci-003"
        elif "bloom" in gen_lower:
            row["model_family"] = "BLOOM-7B"
        elif "glm" in gen_lower:
            row["model_family"] = "GLM-130B"
        elif "flan_t5" in gen_lower:
            size = "base"
            for s in ["small", "base", "large", "xl", "xxl"]:
                if s in gen_lower:
                    size = s
            row["model_family"] = f"Flan-T5-{size}"
        elif "t0_" in gen_lower:
            size = "3b" if "3b" in gen_lower else "11b"
            row["model_family"] = f"T0-{size}"
        elif "opt_" in gen_lower:
            for s in ["125m", "350m", "1.3b", "2.7b", "6.7b", "13b", "30b"]:
                if s in gen_lower:
                    row["model_family"] = f"OPT-{s}"
                    break
            else:
                row["model_family"] = "OPT-other"
        elif "gpt_j" in gen_lower:
            row["model_family"] = "GPT-J-6B"
        elif "gpt_neox" in gen_lower or "gpt-neox" in gen_lower:
            row["model_family"] = "GPT-NeoX-20B"
        elif "human" in gen_lower:
            continue  # Skip human entries
        else:
            row["model_family"] = gen_lower.split("_")[-1] if "_" in gen_lower else "other"

        for method_name, col in methods.items():
            if col in sub.columns:
                row[f"auroc_{method_name}"] = best_auroc(sub["label"].values, sub[col].values)

        rows.append(row)

    gen_df = pd.DataFrame(rows)

    if len(gen_df) > 0:
        # Print top-level summary by model family
        print(f"\n{'='*70}")
        print("PER-MODEL-FAMILY MEAN AUROC")
        print(f"{'='*70}")

        family_agg = gen_df.groupby("model_family").agg(
            **{f"auroc_{name}": (f"auroc_{name}", "mean")
               for name in methods.keys() if f"auroc_{name}" in gen_df.columns},
            n_generators=("generator", "count"),
        ).round(4)

        # Sort by MRE AUROC
        mre_col = f"auroc_MRE (SMDM)"
        if mre_col in family_agg.columns:
            family_agg = family_agg.sort_values(mre_col, ascending=False)

        print(family_agg.to_string())
        family_agg.to_csv(os.path.join(RESULTS_DIR, "per_family_auroc.csv"))

        # ─── KEY FIGURE: MRE vs FDGPT scatter ───────────────────────────
        if "auroc_MRE (SMDM)" in gen_df.columns and "auroc_Fast-DetectGPT" in gen_df.columns:
            print(f"\n{'='*70}")
            print("KEY FIGURE: MRE vs Fast-DetectGPT per-generator scatter")
            print(f"{'='*70}")

            fig, ax = plt.subplots(1, 1, figsize=(10, 10))

            # Color by model family
            families = gen_df["model_family"].unique()
            cmap = plt.cm.get_cmap("tab20", len(families))
            family_colors = {f: cmap(i) for i, f in enumerate(sorted(families))}

            for family in sorted(families):
                sub = gen_df[gen_df["model_family"] == family]
                mre_vals = sub["auroc_MRE (SMDM)"].values
                fdgpt_vals = sub["auroc_Fast-DetectGPT"].values
                ax.scatter(fdgpt_vals, mre_vals, c=[family_colors[family]],
                          label=family, alpha=0.7, s=40, edgecolors="k", linewidth=0.3)

            # Reference lines
            ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Equal")
            ax.axhline(y=0.85, color="green", linestyle=":", alpha=0.4, label="AUROC=0.85")
            ax.axvline(x=0.85, color="green", linestyle=":", alpha=0.4)

            ax.set_xlabel("Fast-DetectGPT AUROC", fontsize=14)
            ax.set_ylabel("MRE (SMDM-1.1B) AUROC", fontsize=14)
            ax.set_title("Per-Generator Detection: MRE vs Fast-DetectGPT\n"
                        "(Points above diagonal = MRE wins)", fontsize=14)
            ax.set_xlim(0.45, 1.02)
            ax.set_ylim(0.45, 1.02)
            ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.2)

            plt.tight_layout()
            fig.savefig(os.path.join(RESULTS_DIR, "mre_vs_fdgpt_scatter.png"), dpi=150, bbox_inches="tight")
            plt.show()
            print(f"Saved → {RESULTS_DIR}/mre_vs_fdgpt_scatter.png")

            # Count how many generators each method wins
            both_valid = gen_df[["auroc_MRE (SMDM)", "auroc_Fast-DetectGPT"]].dropna()
            mre_wins = (both_valid["auroc_MRE (SMDM)"] > both_valid["auroc_Fast-DetectGPT"]).sum()
            fdgpt_wins = (both_valid["auroc_Fast-DetectGPT"] > both_valid["auroc_MRE (SMDM)"]).sum()
            ties = (both_valid["auroc_MRE (SMDM)"] == both_valid["auroc_Fast-DetectGPT"]).sum()
            print(f"\n  MRE wins: {mre_wins}, FDGPT wins: {fdgpt_wins}, ties: {ties}")
            print(f"  MRE mean AUROC: {both_valid['auroc_MRE (SMDM)'].mean():.4f}")
            print(f"  FDGPT mean AUROC: {both_valid['auroc_Fast-DetectGPT'].mean():.4f}")

            # Complementarity analysis
            # How many generators are detectable (>0.85) by at least one method?
            mre_detectable = (both_valid["auroc_MRE (SMDM)"] >= 0.85).sum()
            fdgpt_detectable = (both_valid["auroc_Fast-DetectGPT"] >= 0.85).sum()
            either_detectable = ((both_valid["auroc_MRE (SMDM)"] >= 0.85) |
                                (both_valid["auroc_Fast-DetectGPT"] >= 0.85)).sum()
            both_detectable = ((both_valid["auroc_MRE (SMDM)"] >= 0.85) &
                              (both_valid["auroc_Fast-DetectGPT"] >= 0.85)).sum()
            mre_only = mre_detectable - both_detectable
            fdgpt_only = fdgpt_detectable - both_detectable

            print(f"\n  Detectable (AUROC≥0.85) by:")
            print(f"    MRE only:  {mre_only}")
            print(f"    FDGPT only: {fdgpt_only}")
            print(f"    Both:      {both_detectable}")
            print(f"    Either:    {either_detectable} / {len(both_valid)} ({100*either_detectable/len(both_valid):.1f}%)")

# =========================================================================
#  PART 2: Logistic Combiner
# =========================================================================

print("\n" + "=" * 70)
print("PART 2: LOGISTIC COMBINER — MRE + FDGPT + Classical")
print("=" * 70)

# Features for the combiner
feature_candidates = ["mre_mean", "mre_r0.15", "mre_r0.30", "mre_r0.50",
                      "fdgpt_curvature", "cls_log_likelihood", "cls_mean_rank",
                      "cls_mean_entropy", "cls_perplexity"]

available_features = [f for f in feature_candidates if f in df.columns]
print(f"Available features: {available_features}")

if len(available_features) >= 2 and "label" in df.columns:
    labels = df["label"].values
    
    # Clean and impute features
    usable_features = []
    for f in available_features:
        # Convert to numeric, handle infs
        df[f] = pd.to_numeric(df[f], errors="coerce")
        df[f] = df[f].replace([np.inf, -np.inf], np.nan)
        
        if df[f].isna().all():
            print(f"Warning: Feature '{f}' is completely NaN. Dropping it.")
        else:
            df[f] = df[f].fillna(df[f].median())
            usable_features.append(f)
            
    available_features = usable_features
        
    if not available_features:
        print("No valid features remaining for combiner.")
        valid = np.zeros(len(df), dtype=bool)
    else:
        valid = np.all([np.isfinite(df[f].values) for f in available_features], axis=0)
        
    X = df.loc[valid, available_features].values
    y = labels[valid]

    print(f"Valid samples: {valid.sum()} / {len(df)}")

    # Feature combinations to try
    combos = {
        "MRE only": [f for f in available_features if f.startswith("mre_")],
        "FDGPT only": [f for f in available_features if f.startswith("fdgpt_")],
        "Classical only": [f for f in available_features if f.startswith("cls_")],
        "MRE + FDGPT": [f for f in available_features if f.startswith("mre_") or f.startswith("fdgpt_")],
        "MRE + Classical": [f for f in available_features if f.startswith("mre_") or f.startswith("cls_")],
        "All features": available_features,
    }

    print(f"\n{'Combination':<25s} {'Features':>3s}  {'5-fold CV AUROC':>15s}")
    print("-" * 50)

    for name, feats in combos.items():
        feats = [f for f in feats if f in available_features]
        if not feats:
            continue
        
        # Double check if we still have samples
        if valid.sum() == 0:
            continue

        X_combo = df.loc[valid, feats].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_combo)

        clf = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
        cv_scores = cross_val_score(clf, X_scaled, y, cv=5, scoring="roc_auc")

        print(f"  {name:<25s} {len(feats):>3d}  {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Train final combiner with all features
    print("\n--- Final Combined Detector ---")
    if valid.sum() > 0 and available_features:
        X_all = df.loc[valid, available_features].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_all)
        clf = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
        clf.fit(X_scaled, y)

        combined_scores = clf.predict_proba(X_scaled)[:, 1]
        combined_auroc = roc_auc_score(y, combined_scores)
        print(f"  Combined AUROC (train): {combined_auroc:.4f}")

        # Feature importances
        print(f"\n  Feature weights:")
        for f, w in sorted(zip(available_features, clf.coef_[0]), key=lambda x: abs(x[1]), reverse=True):
            print(f"    {f:<25s}: {w:+.4f}")

        # Save combined scores
        df.loc[valid, "combined_logistic"] = combined_scores
        meta = ["id", "text", "label", "generator", "domain", "dataset", "attack"]
        save_cols = [c for c in meta if c in df.columns] + available_features + ["combined_logistic"]
        df[save_cols].to_parquet(os.path.join(RESULTS_DIR, "scores_combined.parquet"), index=False)

        # Plot ROC curves for individual vs combined
        fig, ax = plt.subplots(figsize=(8, 8))

        for method_name, col in methods.items():
            if col not in df.columns:
                continue
            scores = df.loc[valid, col].values
            a1 = roc_auc_score(y, scores)
            a2 = roc_auc_score(y, -scores)
            if a2 > a1:
                scores = -scores
                a1 = a2
            fpr, tpr, _ = roc_curve(y, scores)
            ax.plot(fpr, tpr, label=f"{method_name} (AUROC={a1:.3f})", alpha=0.7)

        # Combined
        fpr_c, tpr_c, _ = roc_curve(y, combined_scores)
        ax.plot(fpr_c, tpr_c, "k-", linewidth=2.5, label=f"Combined (AUROC={combined_auroc:.3f})")

        ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax.set_xlabel("False Positive Rate", fontsize=13)
        ax.set_ylabel("True Positive Rate", fontsize=13)
        ax.set_title("Detection ROC: Individual Methods vs Combined", fontsize=14)
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, "roc_combined.png"), dpi=150)
        plt.show()
        print(f"Saved → {RESULTS_DIR}/roc_combined.png")
    else:
        print("Not enough valid samples to train the final combined detector.")

else:
    print("Not enough features available for combiner. Need MRE + at least one baseline.")

# =========================================================================
#  PART 3: Summary
# =========================================================================

print("\n" + "=" * 70)
print("NOTEBOOK 08 COMPLETE — Head-to-Head Comparison")
print("=" * 70)
print(f"Results in: {RESULTS_DIR}/")
print("Key outputs:")
print("  - per_family_auroc.csv: AUROC by model family")
print("  - mre_vs_fdgpt_scatter.png: Complementarity scatter plot")
print("  - roc_combined.png: Combined vs individual ROC curves")
print("  - scores_combined.parquet: Combined scores for downstream use")
