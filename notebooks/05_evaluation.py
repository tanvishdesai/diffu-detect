"""
==========================================================================
  DiffuDetect — Kaggle Notebook 5: Evaluation, Robustness & Figures
==========================================================================

PURPOSE:
  - Aggregate all score files (from notebooks 2-4)
  - Compute clean-text AUROC table (M2 milestone)
  - Compute robustness ΔAUROC (M3 — THE DECISIVE EXPERIMENT)
  - Fit logistic combiner
  - Generate publication figures
  - Output GO/NO-GO #2 decision

KAGGLE SETTINGS:
  - GPU: None needed (CPU is fine)
  - Internet: OFF is fine
  - Accelerator: None

INPUT DATASETS:
  - Attach outputs from Notebooks 2, 3, and 4 as Kaggle Datasets
"""

# !pip install -q scikit-learn pandas pyarrow matplotlib seaborn

import os, sys, glob, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ─── Config ──────────────────────────────────────────────────────────────────

# Point these to where your scoring outputs are stored
SCORE_DIRS = [
    "/kaggle/input/diffudetect-mre-scores/results",
    "/kaggle/input/diffudetect-dc-dtd-scores/results",
    "/kaggle/input/diffudetect-baseline-scores/results",
    "/kaggle/working/results",  # If some scores are in the working directory
]

OUTPUT_DIR = "/kaggle/working/analysis"
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ─── Score directions ────────────────────────────────────────────────────────

SCORE_DIRECTIONS = {
    "mre_mean": "lower_is_machine",
    "mre_r0.15": "lower_is_machine",
    "mre_r0.30": "lower_is_machine",
    "mre_r0.50": "lower_is_machine",
    "dc_curvature": "higher_is_machine",
    "dc_normalized": "higher_is_machine",
    "dtd_entropy_auc": "lower_is_machine",
    "dtd_mean_commit_time": "lower_is_machine",
    "dtd_trajectory_curvature": "lower_is_machine",
    "dtd_mean_flips": "lower_is_machine",
    "dtd_final_entropy": "lower_is_machine",
    "dtd_entropy_drop": "higher_is_machine",
    "fdgpt_curvature": "higher_is_machine",
    "dgpt_curvature": "higher_is_machine",
    "bino_score": "lower_is_machine",
    "cls_log_likelihood": "higher_is_machine",
    "cls_mean_rank": "lower_is_machine",
    "cls_mean_log_rank": "lower_is_machine",
    "cls_mean_entropy": "lower_is_machine",
    "cls_perplexity": "lower_is_machine",
    "combined_logistic": "higher_is_machine",
}

# DiffuDetect methods vs baselines
DD_METHODS = {"mre_mean", "dc_normalized", "dtd_entropy_auc", "dtd_mean_commit_time", "combined_logistic"}
BL_METHODS = {"fdgpt_curvature", "cls_log_likelihood", "cls_mean_entropy", "bino_score"}

# ─── Cell 1: Load and aggregate all scores ───────────────────────────────────

print("=" * 60)
print("STEP 1: Aggregating scores from all notebooks")
print("=" * 60)

all_files = []
for d in SCORE_DIRS:
    if os.path.exists(d):
        files = glob.glob(os.path.join(d, "scores_*.parquet"))
        all_files.extend(files)
        print(f"  {d}: {len(files)} files")

if not all_files:
    print("ERROR: No score files found! Run scoring notebooks first.")
    sys.exit(1)

print(f"\nTotal files: {len(all_files)}")

# Load and merge
meta_cols = {"id", "text", "label", "generator", "domain", "dataset", "attack"}
merged = None

for f in all_files:
    df_part = pd.read_parquet(f)
    score_cols = [c for c in df_part.columns if c not in meta_cols]
    print(f"  {os.path.basename(f)}: {len(df_part)} rows, scores: {score_cols}")

    if merged is None:
        merged = df_part
    else:
        new_cols = [c for c in df_part.columns if c not in meta_cols] + ["id"]
        merged = merged.merge(df_part[new_cols], on="id", how="outer", suffixes=("", "_dup"))
        dup_cols = [c for c in merged.columns if c.endswith("_dup")]
        merged = merged.drop(columns=dup_cols)

df = merged
print(f"\nMerged: {len(df)} rows, {len(df.columns)} columns")

# Identify available score columns
score_cols = [c for c in df.columns if c in SCORE_DIRECTIONS]
print(f"Score columns found: {score_cols}")

# ─── Cell 2: Compute AUROC for each method ───────────────────────────────────

print("\n" + "=" * 60)
print("STEP 2: Clean-Text Detection Performance (Table 1)")
print("=" * 60)

def compute_auroc(labels, scores, direction):
    valid = np.isfinite(scores)
    if valid.sum() < 10: return np.nan, np.nan, np.nan
    s = -scores[valid] if direction == "lower_is_machine" else scores[valid]
    auroc = roc_auc_score(labels[valid], s)
    fpr, tpr, _ = roc_curve(labels[valid], s)
    idx1 = max(0, np.searchsorted(fpr, 0.01, side="right") - 1)
    idx5 = max(0, np.searchsorted(fpr, 0.05, side="right") - 1)
    return auroc, tpr[idx1], tpr[idx5]

labels = df["label"].values
results_rows = []

for col in score_cols:
    direction = SCORE_DIRECTIONS[col]
    auroc, tpr1, tpr5 = compute_auroc(labels, df[col].values, direction)
    is_dd = "★" if col in DD_METHODS else " "
    results_rows.append({
        "marker": is_dd, "method": col, "auroc": auroc,
        "tpr@1%fpr": tpr1, "tpr@5%fpr": tpr5,
    })
    print(f"  {is_dd} {col:<30s} AUROC={auroc:.4f}  TPR@1%={tpr1:.4f}  TPR@5%={tpr5:.4f}")

table1 = pd.DataFrame(results_rows).sort_values("auroc", ascending=False)
table1.to_csv(os.path.join(OUTPUT_DIR, "table1_clean_auroc.csv"), index=False)

# ─── Cell 3: Fit logistic combiner ───────────────────────────────────────────

dd_features = [c for c in score_cols if c.startswith(("mre_", "dc_", "dtd_"))]
if len(dd_features) >= 2:
    print(f"\n{'='*60}")
    print(f"STEP 2b: Logistic combiner ({len(dd_features)} features)")
    print("=" * 60)

    # Prepare features
    X = df[dd_features].copy()
    for c in dd_features: X[c] = X[c].fillna(X[c].median())
    X = X.values; y = labels

    # Split calibration (10%)
    np.random.seed(42)
    cal_idx = np.random.choice(len(X), size=len(X)//10, replace=False)
    eval_idx = np.setdiff1d(np.arange(len(X)), cal_idx)

    scaler = StandardScaler()
    X_cal = scaler.fit_transform(X[cal_idx])
    X_eval = scaler.transform(X[eval_idx])

    clf = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    clf.fit(X_cal, y[cal_idx])

    # Score everything
    X_all = scaler.transform(X)
    df["combined_logistic"] = clf.predict_proba(X_all)[:, 1]
    score_cols.append("combined_logistic")

    auroc_comb = roc_auc_score(y, df["combined_logistic"].values)
    print(f"  Combined AUROC: {auroc_comb:.4f}")
    print(f"  Feature weights: {dict(zip(dd_features, clf.coef_[0].round(3)))}")

# ─── Cell 4: Robustness analysis (THE DECISIVE EXPERIMENT) ───────────────────

print("\n" + "=" * 60)
print("STEP 3: ROBUSTNESS ANALYSIS — GO/NO-GO #2")
print("=" * 60)

if "attack" in df.columns:
    clean_df = df[df["attack"] == "none"]
    attacks = [a for a in df["attack"].unique() if a != "none"]

    if len(clean_df) > 0 and len(attacks) > 0:
        print(f"Clean samples: {len(clean_df)}")
        print(f"Attack types: {attacks}")

        # Overall robustness: clean vs all-attacks
        attacked_df = df[df["attack"] != "none"]

        rob_rows = []
        for col in score_cols:
            direction = SCORE_DIRECTIONS.get(col, "higher_is_machine")
            auroc_clean, _, _ = compute_auroc(clean_df["label"].values, clean_df[col].values, direction)
            auroc_att, _, _ = compute_auroc(attacked_df["label"].values, attacked_df[col].values, direction)
            delta = auroc_clean - auroc_att if np.isfinite(auroc_clean) and np.isfinite(auroc_att) else np.nan
            is_dd = "★" if col in DD_METHODS else " "
            rob_rows.append({
                "marker": is_dd, "method": col,
                "auroc_clean": auroc_clean, "auroc_attacked": auroc_att,
                "delta_auroc": delta,
            })
            print(f"  {is_dd} {col:<30s} Clean={auroc_clean:.4f}  Attacked={auroc_att:.4f}  ΔAUROC={delta:+.4f}")

        table2 = pd.DataFrame(rob_rows).sort_values("delta_auroc", ascending=True)
        table2.to_csv(os.path.join(OUTPUT_DIR, "table2_robustness.csv"), index=False)

        # Per-attack breakdown
        for attack in attacks:
            att_df = df[df["attack"] == attack]
            if len(att_df) < 50: continue
            print(f"\n  --- Attack: {attack} (n={len(att_df)}) ---")
            for col in score_cols:
                direction = SCORE_DIRECTIONS.get(col, "higher_is_machine")
                auroc_c, _, _ = compute_auroc(clean_df["label"].values, clean_df[col].values, direction)
                auroc_a, _, _ = compute_auroc(att_df["label"].values, att_df[col].values, direction)
                delta = auroc_c - auroc_a if np.isfinite(auroc_c) and np.isfinite(auroc_a) else np.nan
                if col in DD_METHODS or col in BL_METHODS:
                    print(f"    {col:<30s} Δ={delta:+.4f}")

        # GO/NO-GO Decision
        dd_rows = table2[table2["method"].isin(DD_METHODS)]
        bl_rows = table2[table2["method"].isin(BL_METHODS)]
        if len(dd_rows) > 0 and len(bl_rows) > 0:
            dd_mean = dd_rows["delta_auroc"].mean()
            bl_mean = bl_rows["delta_auroc"].mean()
            advantage = bl_mean - dd_mean

            print(f"\n{'='*60}")
            print(f"GO/NO-GO #2 — THE DECISIVE VERDICT")
            print(f"{'='*60}")
            print(f"DiffuDetect avg ΔAUROC: {dd_mean:+.4f}")
            print(f"Baseline avg ΔAUROC:    {bl_mean:+.4f}")
            print(f"Robustness advantage:   {advantage:+.4f}")

            if advantage >= 0.08:
                print("🟢 GO: ≥8pt advantage. THIS IS THE PAPER → Push to AAAI-27")
            elif advantage >= 0.04:
                print("🟡 MARGINAL: 4-8pt. Workshop quality")
            else:
                print("🔴 NO-GO: <4pt. Not strong enough → Workshop or pivot")
    else:
        print("No attack/clean split available for robustness analysis.")
        print("Run RAID scoring to enable robustness experiments.")
else:
    print("No 'attack' column — skipping robustness analysis.")
    print("Add RAID dataset scores for the decisive experiment.")

# ─── Cell 5: Publication figures ──────────────────────────────────────────────

print(f"\n{'='*60}")
print("STEP 4: Generating figures")
print("=" * 60)

sns.set_style("whitegrid")
plt.rcParams.update({"font.size": 12, "figure.dpi": 150})

# Figure 1: Clean AUROC bar chart
fig, ax = plt.subplots(figsize=(12, 6))
plot_data = table1[table1["auroc"].notna()].copy()
colors = ["#2196F3" if m in DD_METHODS else "#9E9E9E" for m in plot_data["method"]]

bars = ax.barh(range(len(plot_data)), plot_data["auroc"], color=colors)
ax.set_yticks(range(len(plot_data)))
ax.set_yticklabels(plot_data["method"], fontsize=9)
ax.set_xlabel("AUROC")
ax.set_title("Clean-Text Detection Performance (★ = DiffuDetect)")
ax.set_xlim(0.4, 1.0)
ax.axvline(x=0.5, color="red", linestyle="--", alpha=0.3)

for i, (bar, val) in enumerate(zip(bars, plot_data["auroc"])):
    ax.text(val + 0.005, i, f"{val:.3f}", va="center", fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "fig1_clean_auroc.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig1_clean_auroc.png")

# Figure 2: Robustness (if available)
if "attack" in df.columns and 'table2' in dir():
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    plot_data = table2[table2["delta_auroc"].notna()].copy()
    colors = ["#2196F3" if m in DD_METHODS else "#FF5722" for m in plot_data["method"]]

    # Panel A: ΔAUROC bars
    axes[0].barh(range(len(plot_data)), plot_data["delta_auroc"], color=colors)
    axes[0].set_yticks(range(len(plot_data)))
    axes[0].set_yticklabels(plot_data["method"], fontsize=8)
    axes[0].set_xlabel("ΔAUROC (clean → attacked)")
    axes[0].set_title("Robustness: ΔAUROC (smaller = more robust)")
    axes[0].axvline(x=0, color="black", alpha=0.3)

    # Panel B: Clean vs Attacked
    x = np.arange(len(plot_data))
    w = 0.35
    axes[1].bar(x - w/2, plot_data["auroc_clean"], w, label="Clean", color="#4CAF50")
    axes[1].bar(x + w/2, plot_data["auroc_attacked"], w, label="Attacked", color="#F44336")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(plot_data["method"], rotation=45, ha="right", fontsize=7)
    axes[1].set_ylabel("AUROC")
    axes[1].set_title("Clean vs Attacked AUROC")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig2_robustness.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved fig2_robustness.png")

# Figure 3: Score distributions
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
main_methods = [c for c in ["mre_mean","dc_normalized","dtd_entropy_auc",
                            "fdgpt_curvature","cls_log_likelihood","combined_logistic"] if c in df.columns]

for i, col in enumerate(main_methods[:6]):
    ax = axes[i//3, i%3]
    human = df[df["label"]==0][col].dropna()
    machine = df[df["label"]==1][col].dropna()
    ax.hist(human, bins=50, alpha=0.5, label="Human", color="blue", density=True)
    ax.hist(machine, bins=50, alpha=0.5, label="Machine", color="red", density=True)
    ax.set_title(col, fontsize=10)
    ax.legend(fontsize=8)

plt.suptitle("Score Distributions: Human vs Machine", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "fig3_distributions.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig3_distributions.png")

# Figure 4: ROC curves
fig, ax = plt.subplots(figsize=(10, 10))
for col in score_cols:
    if col not in df.columns: continue
    scores = df[col].values
    direction = SCORE_DIRECTIONS.get(col, "higher_is_machine")
    valid = np.isfinite(scores)
    if valid.sum() < 10: continue
    s = -scores[valid] if direction == "lower_is_machine" else scores[valid]
    fpr, tpr, _ = roc_curve(labels[valid], s)
    auroc = roc_auc_score(labels[valid], s)
    style = "-" if col in DD_METHODS else "--"
    lw = 2 if col in DD_METHODS else 1
    ax.plot(fpr, tpr, linestyle=style, linewidth=lw, label=f"{col} ({auroc:.3f})")

ax.plot([0,1],[0,1], "k--", alpha=0.3)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — All Methods")
ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "fig4_roc_curves.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig4_roc_curves.png")

# ─── Cell 6: Save final aggregated data ──────────────────────────────────────

df.to_parquet(os.path.join(OUTPUT_DIR, "full_aggregated.parquet"), index=False)

# ─── Summary ─────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)
print(f"Output directory: {OUTPUT_DIR}")
print(f"Figures: {FIGURES_DIR}")
print(f"\nFiles generated:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
    print(f"  {f} ({size/1024:.0f} KB)")
if os.path.exists(FIGURES_DIR):
    for f in sorted(os.listdir(FIGURES_DIR)):
        print(f"  figures/{f}")

print("\n★ DiffuDetect methods are marked with ★ in the tables above.")
print("Review the GO/NO-GO decision to determine next steps.")
