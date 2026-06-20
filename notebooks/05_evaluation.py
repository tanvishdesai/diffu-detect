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

# ─── Within-testbed evaluation helpers ───────────────────────────────────────
#
# THE KEY FIX (v1 → v2). In v1 the headline metric was a single AUROC over the
# fully-POOLED set (all human domains vs all machine generators). That floored
# the number to ~0.60 even though per-generator AUROC was 0.9+ — a textbook
# Simpson's-paradox/pooling artifact, because the absolute score scale shifts
# across domains so one global threshold can't separate them. The standard
# MAGE / Fast-DetectGPT protocol evaluates WITHIN each testbed and averages.
# We make mean within-testbed AUROC the primary metric here.

def _extract_domain(gen):
    # MAGE generators encode domain as the first underscore field:
    #   "cmv_human", "cmv_machine_continuation_13B" -> "cmv"
    return str(gen).split("_")[0]

if "domain" not in df.columns or df["domain"].isna().all() or (df["domain"] == "unknown").all():
    if "generator" in df.columns:
        df["domain"] = df["generator"].map(_extract_domain)

# Composite testbed id keeps MAGE and RAID domains from colliding (e.g. both may
# have an "xsum"); each (dataset, domain) is its own within-testbed cell.
if "dataset" not in df.columns:
    df["dataset"] = "mage"
df["testbed"] = df["dataset"].astype(str) + "/" + df["domain"].astype(str)

def _orient(col):
    """Return scores oriented so that HIGHER = more machine, using the declared
    SCORE_DIRECTIONS. Falls back to pooled-sign detection for unknown columns."""
    s = pd.to_numeric(df[col], errors="coerce").values.astype(float)
    direction = SCORE_DIRECTIONS.get(col)
    if direction == "lower_is_machine":
        return -s
    if direction == "higher_is_machine":
        return s
    # Unknown: orient once on the pooled data (best-effort, logged as such)
    valid = np.isfinite(s)
    if valid.sum() >= 10:
        try:
            if roc_auc_score(df["label"].values[valid], s[valid]) < 0.5:
                return -s
        except ValueError:
            pass
    return s

def compute_auroc(labels, scores, direction=None):
    """Pooled AUROC + TPR (kept for the reference 'naive pooled' column and for
    per-generator/robustness slices). Auto-detects direction within the slice."""
    valid = np.isfinite(scores)
    if valid.sum() < 10: return np.nan, np.nan, np.nan
    auroc_pos = roc_auc_score(labels[valid], scores[valid])
    auroc_neg = roc_auc_score(labels[valid], -scores[valid])
    if auroc_pos >= auroc_neg:
        s = scores[valid]; auroc = auroc_pos
    else:
        s = -scores[valid]; auroc = auroc_neg
    fpr, tpr, _ = roc_curve(labels[valid], s)
    idx1 = max(0, np.searchsorted(fpr, 0.01, side="right") - 1)
    idx5 = max(0, np.searchsorted(fpr, 0.05, side="right") - 1)
    return auroc, tpr[idx1], tpr[idx5]

def within_testbed_auroc(frame, col, testbed_col="domain", min_per_class=20):
    """Mean within-testbed AUROC. Orientation is FIXED globally (no per-testbed
    flipping → no optimistic bias); AUROC is computed inside each testbed and
    averaged (both unweighted and sample-weighted)."""
    s_all = _orient(col)
    sub = frame.index
    aurocs, weights = [], []
    for _, g in frame.groupby(testbed_col):
        idx = g.index
        y = g["label"].values
        s = s_all[idx]
        valid = np.isfinite(s)
        y, s = y[valid], s[valid]
        if (y == 0).sum() < min_per_class or (y == 1).sum() < min_per_class:
            continue
        try:
            aurocs.append(roc_auc_score(y, s)); weights.append(len(y))
        except ValueError:
            continue
    if not aurocs:
        return np.nan, np.nan, 0
    aurocs = np.array(aurocs); weights = np.array(weights)
    return float(aurocs.mean()), float(np.average(aurocs, weights=weights)), len(aurocs)

# ─── Cell 2: Clean-text detection performance — WITHIN-TESTBED (primary) ──────

print("\n" + "=" * 60)
print("STEP 2: Clean-Text Detection Performance (Table 1)")
print("  PRIMARY metric = mean within-testbed (per-domain) AUROC")
print("  'pooled' shown only for reference (known to be floored by domain mixing)")
print("=" * 60)

# df is indexed 0..N-1 after merge; ensure a clean RangeIndex so .index aligns
df = df.reset_index(drop=True)
labels = df["label"].values
results_rows = []

# Clean-text table is evaluated on un-attacked rows only (so adding RAID's
# attacked rows later doesn't contaminate the clean numbers).
clean_frame = df[df["attack"] == "none"] if "attack" in df.columns else df

for col in score_cols:
    pooled, tpr1, tpr5 = compute_auroc(
        clean_frame["label"].values, pd.to_numeric(clean_frame[col], errors="coerce").values)
    wt_mean, wt_wmean, n_tb = within_testbed_auroc(clean_frame, col, testbed_col="testbed")
    is_dd = "★" if col in DD_METHODS else " "
    results_rows.append({
        "marker": is_dd, "method": col,
        "auroc_within_testbed": wt_mean,     # PRIMARY
        "auroc_within_testbed_wt": wt_wmean,  # sample-weighted
        "n_testbeds": n_tb,
        "auroc_pooled": pooled,               # reference only
        "tpr@1%fpr": tpr1, "tpr@5%fpr": tpr5,
    })
    print(f"  {is_dd} {col:<26s} within-testbed={wt_mean:.4f} (n_tb={n_tb})  "
          f"pooled={pooled:.4f}  TPR@1%={tpr1:.4f}")

table1 = pd.DataFrame(results_rows).sort_values("auroc_within_testbed", ascending=False)
table1.to_csv(os.path.join(OUTPUT_DIR, "table1_clean_auroc.csv"), index=False)

# GO/NO-GO #1 re-evaluated on the CORRECT (within-testbed) metric
dd_in_table = table1[table1["method"].isin(DD_METHODS) & table1["auroc_within_testbed"].notna()]
if len(dd_in_table) > 0:
    best_dd = dd_in_table.iloc[0]
    print(f"\n  GO/NO-GO #1 (within-testbed): best DiffuDetect = "
          f"{best_dd['method']} @ {best_dd['auroc_within_testbed']:.4f}")
    if best_dd["auroc_within_testbed"] >= 0.85:
        print("  🟢 GO: ≥0.85 within-testbed. Core premise validated.")
    elif best_dd["auroc_within_testbed"] >= 0.70:
        print("  🟡 MARGINAL: 0.70–0.85. Real signal; scale model (LLaDA/Dream) to push higher.")
    else:
        print("  🔴 NO-GO: <0.70 even within-testbed. Premise weak.")

# ─── Cell 2b: Per-generator AUROC (the meaningful metric) ────────────────────

if "generator" in df.columns and df["generator"].nunique() > 1:
    print(f"\n{'='*60}")
    print("STEP 2b: Per-Generator AUROC (more meaningful than aggregate)")
    print("=" * 60)

    machine_gens = df[df["label"] == 1]["generator"].unique()
    gen_rows = []
    for gen in sorted(machine_gens):
        gen_mask = (df["label"] == 0) | (df["generator"] == gen)
        sub = df[gen_mask]
        if sub["label"].nunique() < 2 or len(sub) < 50:
            continue
        row = {"generator": gen, "n": len(sub)}
        for col in score_cols:
            if col not in sub.columns: continue
            a, _, _ = compute_auroc(sub["label"].values, sub[col].values)
            row[f"auroc_{col}"] = a
        gen_rows.append(row)

    if gen_rows:
        gen_table = pd.DataFrame(gen_rows)
        auroc_cols = [c for c in gen_table.columns if c.startswith("auroc_")]
        if auroc_cols:
            gen_table = gen_table.sort_values(auroc_cols[0], ascending=False)

        # Summary by tier
        if auroc_cols:
            main_col = auroc_cols[0]
            tier_hi = gen_table[gen_table[main_col] >= 0.90]
            tier_md = gen_table[(gen_table[main_col] >= 0.70) & (gen_table[main_col] < 0.90)]
            tier_lo = gen_table[gen_table[main_col] < 0.70]
            print(f"\n  AUROC ≥ 0.90 (easy):    {len(tier_hi)} generators")
            print(f"  AUROC 0.70–0.90 (medium): {len(tier_md)} generators")
            print(f"  AUROC < 0.70 (hard):    {len(tier_lo)} generators")

            # Print top and bottom 10
            print(f"\n  Top 10 generators:")
            for _, r in gen_table.head(10).iterrows():
                vals = "  ".join(f"{c.replace('auroc_','')}={r[c]:.3f}" for c in auroc_cols if pd.notna(r[c]))
                print(f"    {r['generator'][:50]:50s} {vals}")
            print(f"\n  Bottom 10 generators:")
            for _, r in gen_table.tail(10).iterrows():
                vals = "  ".join(f"{c.replace('auroc_','')}={r[c]:.3f}" for c in auroc_cols if pd.notna(r[c]))
                print(f"    {r['generator'][:50]:50s} {vals}")

        gen_table.to_csv(os.path.join(OUTPUT_DIR, "table1b_per_generator_auroc.csv"), index=False)
        print(f"\n  Saved per-generator table → table1b_per_generator_auroc.csv")

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

def robustness_testbed(col, attack, testbed_col="testbed", min_per_class=10):
    # NOTE: RAID (domain × attack) cells are smaller than MAGE's. If you see very
    # few n_testbeds per attack, raise MAX_SAMPLES in Notebook 6 (≥6000) so each
    # (domain, attack) cell has enough human+machine to compute a stable AUROC.
    """ΔAUROC for one attack, computed WITHIN-TESTBED.

    For each testbed present in BOTH the clean and attacked splits, compute
    clean AUROC and attacked AUROC with a globally-fixed orientation, then
    average the per-testbed deltas. This is the only way the robustness number
    is meaningful: v1 measured pooled clean (≈0.64, already floored) so there
    was no headroom for any method to 'collapse' under paraphrase.
    """
    s_all = _orient(col)
    clean_mask = (df["attack"] == "none").values
    att_mask = (df["attack"] == attack).values
    clean_tb, att_tb, deltas = [], [], []
    for tb in df[testbed_col].unique():
        tb_mask = (df[testbed_col] == tb).values
        ci = np.where(tb_mask & clean_mask)[0]
        ai = np.where(tb_mask & att_mask)[0]
        if len(ci) == 0 or len(ai) == 0:
            continue
        yc, sc = df["label"].values[ci], s_all[ci]
        ya, sa = df["label"].values[ai], s_all[ai]
        vc, va = np.isfinite(sc), np.isfinite(sa)
        yc, sc, ya, sa = yc[vc], sc[vc], ya[va], sa[va]
        if min((yc == 0).sum(), (yc == 1).sum(), (ya == 0).sum(), (ya == 1).sum()) < min_per_class:
            continue
        try:
            ac = roc_auc_score(yc, sc); aa = roc_auc_score(ya, sa)
        except ValueError:
            continue
        clean_tb.append(ac); att_tb.append(aa); deltas.append(ac - aa)
    if not deltas:
        return np.nan, np.nan, np.nan, 0
    return (float(np.mean(clean_tb)), float(np.mean(att_tb)),
            float(np.mean(deltas)), len(deltas))

if "attack" in df.columns and (df["attack"] != "none").any() and (df["attack"] == "none").any():
    attacks = sorted(a for a in df["attack"].unique() if a != "none")
    print(f"Clean samples: {int((df['attack']=='none').sum())} | Attack types: {attacks}")
    print("All ΔAUROC below are WITHIN-TESTBED (per-domain) means.\n")

    # Overall robustness: average ΔAUROC across all attacks, per method
    rob_rows = []
    for col in score_cols:
        per_attack = [robustness_testbed(col, a) for a in attacks]
        per_attack = [r for r in per_attack if np.isfinite(r[2])]
        if not per_attack:
            continue
        cln = np.mean([r[0] for r in per_attack])
        att = np.mean([r[1] for r in per_attack])
        dlt = np.mean([r[2] for r in per_attack])
        is_dd = "★" if col in DD_METHODS else " "
        rob_rows.append({"marker": is_dd, "method": col,
                         "auroc_clean": cln, "auroc_attacked": att, "delta_auroc": dlt})
        print(f"  {is_dd} {col:<26s} clean={cln:.4f}  attacked={att:.4f}  ΔAUROC={dlt:+.4f}")

    table2 = pd.DataFrame(rob_rows).sort_values("delta_auroc", ascending=True)
    table2.to_csv(os.path.join(OUTPUT_DIR, "table2_robustness.csv"), index=False)

    # Per-attack breakdown (paraphrase is the headline attack)
    per_attack_rows = []
    for attack in attacks:
        print(f"\n  --- Attack: {attack} (within-testbed ΔAUROC) ---")
        for col in score_cols:
            if col not in DD_METHODS and col not in BL_METHODS:
                continue
            cln, att, dlt, n_tb = robustness_testbed(col, attack)
            if not np.isfinite(dlt):
                continue
            per_attack_rows.append({"attack": attack, "method": col,
                                    "auroc_clean": cln, "auroc_attacked": att,
                                    "delta_auroc": dlt, "n_testbeds": n_tb})
            mk = "★" if col in DD_METHODS else " "
            print(f"    {mk} {col:<26s} clean={cln:.4f} att={att:.4f} Δ={dlt:+.4f} (n_tb={n_tb})")
    if per_attack_rows:
        pd.DataFrame(per_attack_rows).to_csv(
            os.path.join(OUTPUT_DIR, "table2b_per_attack_robustness.csv"), index=False)

    # GO/NO-GO #2 — decisive verdict, on the paraphrase attack specifically + overall
    dd_rows = table2[table2["method"].isin(DD_METHODS)]
    bl_rows = table2[table2["method"].isin(BL_METHODS)]
    if len(dd_rows) > 0 and len(bl_rows) > 0:
        dd_mean = dd_rows["delta_auroc"].mean()
        bl_mean = bl_rows["delta_auroc"].mean()
        advantage = bl_mean - dd_mean
        print(f"\n{'='*60}\nGO/NO-GO #2 — THE DECISIVE VERDICT (within-testbed)\n{'='*60}")
        print(f"DiffuDetect avg ΔAUROC: {dd_mean:+.4f}  (smaller = more robust)")
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
    print("Run RAID scoring (Notebook 6) to enable the decisive experiment.")

# ─── Cell 5: Publication figures ──────────────────────────────────────────────

print(f"\n{'='*60}")
print("STEP 4: Generating figures")
print("=" * 60)

sns.set_style("whitegrid")
plt.rcParams.update({"font.size": 12, "figure.dpi": 150})

# Figure 1: Clean AUROC bar chart (within-testbed = the honest primary metric)
fig, ax = plt.subplots(figsize=(12, 6))
plot_data = table1[table1["auroc_within_testbed"].notna()].copy()
colors = ["#2196F3" if m in DD_METHODS else "#9E9E9E" for m in plot_data["method"]]

bars = ax.barh(range(len(plot_data)), plot_data["auroc_within_testbed"], color=colors)
ax.set_yticks(range(len(plot_data)))
ax.set_yticklabels(plot_data["method"], fontsize=9)
ax.set_xlabel("Mean within-testbed AUROC")
ax.set_title("Clean-Text Detection Performance (★ = DiffuDetect)")
ax.set_xlim(0.4, 1.0)
ax.axvline(x=0.5, color="red", linestyle="--", alpha=0.3)

for i, (bar, val) in enumerate(zip(bars, plot_data["auroc_within_testbed"])):
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
