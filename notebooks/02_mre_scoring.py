"""
==========================================================================
  DiffuDetect — Kaggle Notebook 2: MRE Scoring (Phase 1 — Kill-Early Gate)
==========================================================================

PURPOSE:
  - Load SMDM-1.1B (or MDLM-110M for quick test)
  - Score passages with Masked Reconstruction Error (MRE)
  - Sweep mask ratios {0.15, 0.30, 0.50}
  - Compute initial AUROC → GO/NO-GO #1
  - Save scores to Parquet

KAGGLE SETTINGS:
  - GPU: T4 x1 (required)
  - Internet: ON (to download model)
  - Accelerator: GPU T4
  - Persistence: Save outputs as Dataset

INPUT DATASETS:
  - Attach the output from Notebook 1 as a Kaggle Dataset
"""

# ─── Cell 1: Install & imports ───────────────────────────────────────────────
# !pip install -q torch transformers datasets accelerate bitsandbytes \
#     scikit-learn pandas pyarrow tqdm huggingface_hub sentencepiece protobuf

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")

# ─── Cell 2: Configuration ───────────────────────────────────────────────────

# === CHANGE THESE FOR YOUR RUN ===
MODEL_NAME = "smdm-1.1b"           # "smdm-1.1b" or "mdlm-110m" for quick test
MODEL_HF_REPO = "nieshen/SMDM-1.1b"  # HF repo
QUANTIZE_BITS = None                 # None for full precision
MAX_SAMPLES = 2000                   # Start small, scale up later
MAX_LENGTH = 512
MASK_RATIOS = [0.15, 0.30, 0.50]
NUM_MASK_DRAWS = 16                  # K mask draws per ratio
SEED = 42
DATASET = "mage"                     # "mage" or "raid"

# Paths
DATA_DIR = "/kaggle/input/diffudetect-data/data"  # ← adjust to your dataset name
WORK_DIR = "/kaggle/working"
RESULTS_DIR = os.path.join(WORK_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
np.random.seed(SEED)
torch.manual_seed(SEED)

# ─── Cell 3: Load preprocessed data ──────────────────────────────────────────

data_file = os.path.join(DATA_DIR, f"{DATASET}_quick.parquet")
if not os.path.exists(data_file):
    data_file = os.path.join(DATA_DIR, f"{DATASET}_full.parquet")

if os.path.exists(data_file):
    df = pd.read_parquet(data_file)
    print(f"Loaded {len(df)} passages from {data_file}")
else:
    # Fallback: load directly from HuggingFace
    print("Loading directly from HuggingFace...")
    from datasets import load_dataset
    ds = load_dataset("yaful/MAGE", trust_remote_code=True)
    split = "test" if "test" in ds else list(ds.keys())[0]
    df = ds[split].to_pandas()

    # Standardize
    for src, dst in [("source_model", "generator"), ("category", "domain")]:
        if src in df.columns:
            df = df.rename(columns={src: dst})
    if "generator" not in df.columns:
        df["generator"] = df["label"].apply(lambda x: "machine" if x == 1 else "human")
    if "domain" not in df.columns:
        df["domain"] = "unknown"
    df["dataset"] = "mage"
    df["attack"] = "none"
    df["id"] = [f"mage_{i}" for i in range(len(df))]
    df["label"] = df["label"].astype(int)

# Subsample if needed
if MAX_SAMPLES and len(df) > MAX_SAMPLES:
    df = df.groupby("label", group_keys=False).apply(
        lambda x: x.sample(n=min(MAX_SAMPLES//2, len(x)), random_state=SEED)
    ).reset_index(drop=True)
    print(f"Subsampled to {len(df)} passages")

print(f"Labels: {df['label'].value_counts().to_dict()}")
if "generator" in df.columns:
    print(f"Generators: {df['generator'].value_counts().to_dict()}")

# ─── Cell 4: Load diffusion model ────────────────────────────────────────────

from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM

print(f"\nLoading model: {MODEL_HF_REPO}...")
start_time = time.time()

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_HF_REPO, trust_remote_code=True
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Try different model loading strategies
model = None
for ModelClass in [AutoModelForMaskedLM, AutoModel]:
    try:
        model = ModelClass.from_pretrained(
            MODEL_HF_REPO,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map={"": DEVICE},
        )
        print(f"Loaded with {ModelClass.__name__}")
        break
    except Exception as e:
        print(f"  {ModelClass.__name__} failed: {e}")
        continue

if model is None:
    raise RuntimeError(f"Could not load model {MODEL_HF_REPO}")

model.eval()
load_time = time.time() - start_time
print(f"Model loaded in {load_time:.1f}s")
print(f"GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

# Determine mask token ID
MASK_TOKEN_ID = None
if hasattr(tokenizer, 'mask_token_id') and tokenizer.mask_token_id is not None:
    MASK_TOKEN_ID = tokenizer.mask_token_id
elif hasattr(model.config, 'mask_token_id'):
    MASK_TOKEN_ID = model.config.mask_token_id
elif "[MASK]" in tokenizer.get_vocab():
    MASK_TOKEN_ID = tokenizer.convert_tokens_to_ids("[MASK]")
elif "<mask>" in tokenizer.get_vocab():
    MASK_TOKEN_ID = tokenizer.convert_tokens_to_ids("<mask>")
else:
    MASK_TOKEN_ID = tokenizer.unk_token_id or (len(tokenizer) - 1)

print(f"Mask token ID: {MASK_TOKEN_ID}")

# Special tokens to never mask
SPECIAL_IDS = set()
for attr in ['bos_token_id', 'eos_token_id', 'pad_token_id', 'cls_token_id', 'sep_token_id']:
    tid = getattr(tokenizer, attr, None)
    if tid is not None:
        SPECIAL_IDS.add(tid)

# ─── Cell 5: MRE scoring function ────────────────────────────────────────────

@torch.no_grad()
def compute_mre(text, mask_ratio, num_draws=16, max_length=512):
    """
    Compute Masked Reconstruction Error for a single text.

    Returns the mean NLL of true tokens at masked positions,
    averaged over num_draws random mask samples.
    """
    # Tokenize
    encoding = tokenizer(
        text, max_length=max_length, truncation=True,
        padding="max_length", return_tensors="pt"
    )
    input_ids = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)

    batch_size, seq_len = input_ids.shape
    draw_nlls = []

    for _ in range(num_draws):
        # Create random mask
        eligible = torch.ones(seq_len, dtype=torch.bool, device=DEVICE)
        if tokenizer.pad_token_id is not None:
            eligible &= (input_ids[0] != tokenizer.pad_token_id)
        for sid in SPECIAL_IDS:
            eligible &= (input_ids[0] != sid)

        eligible_idx = eligible.nonzero(as_tuple=True)[0]
        n_mask = max(1, int(len(eligible_idx) * mask_ratio))
        perm = torch.randperm(len(eligible_idx), device=DEVICE)[:n_mask]
        mask_positions = eligible_idx[perm]

        # Apply mask
        masked_input = input_ids.clone()
        masked_input[0, mask_positions] = MASK_TOKEN_ID

        # Forward pass
        try:
            outputs = model(input_ids=masked_input, attention_mask=attention_mask)
        except TypeError:
            outputs = model(input_ids=masked_input)

        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        elif isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs

        # NLL at masked positions
        log_probs = F.log_softmax(logits, dim=-1)
        true_log_probs = log_probs[0, mask_positions, :]
        true_token_ids = input_ids[0, mask_positions]
        token_nlls = -true_log_probs[torch.arange(len(mask_positions)), true_token_ids]

        draw_nlls.append(token_nlls.mean().item())

    return np.mean(draw_nlls)

# ─── Cell 6: Run MRE scoring ─────────────────────────────────────────────────

print(f"\nStarting MRE scoring: {len(df)} passages × {len(MASK_RATIOS)} ratios × {NUM_MASK_DRAWS} draws")
print(f"Estimated: {len(df) * len(MASK_RATIOS) * NUM_MASK_DRAWS} forward passes total")

all_scores = {f"mre_r{r:.2f}": [] for r in MASK_RATIOS}
all_scores["mre_mean"] = []
errors = []

scoring_start = time.time()

for idx, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="MRE Scoring")):
    text = str(row["text"])

    try:
        ratio_scores = {}
        for ratio in MASK_RATIOS:
            nll = compute_mre(text, mask_ratio=ratio, num_draws=NUM_MASK_DRAWS, max_length=MAX_LENGTH)
            ratio_scores[f"mre_r{ratio:.2f}"] = nll

        # Mean across ratios
        ratio_scores["mre_mean"] = np.mean(list(ratio_scores.values()))

        for key, val in ratio_scores.items():
            all_scores[key].append(val)

    except Exception as e:
        errors.append((idx, str(e)))
        for key in all_scores:
            all_scores[key].append(np.nan)

    # Progress report every 100 passages
    if (idx + 1) % 100 == 0:
        elapsed = time.time() - scoring_start
        rate = (idx + 1) / elapsed
        remaining = (len(df) - idx - 1) / rate
        print(f"  [{idx+1}/{len(df)}] {rate:.1f} passages/s, ~{remaining/60:.0f} min remaining")

scoring_time = time.time() - scoring_start
print(f"\nScoring complete: {scoring_time:.0f}s ({scoring_time/len(df):.2f}s per passage)")
if errors:
    print(f"Errors: {len(errors)}")

# ─── Cell 7: Save scores ─────────────────────────────────────────────────────

# Add scores to DataFrame
for key, vals in all_scores.items():
    df[key] = vals

# Save wide-format results
output_path = os.path.join(RESULTS_DIR, f"scores_{DATASET}_mre_{MODEL_NAME}.parquet")
meta_cols = ["id", "text", "label", "generator", "domain", "dataset", "attack"]
score_cols = list(all_scores.keys())
save_cols = [c for c in meta_cols if c in df.columns] + score_cols
df[save_cols].to_parquet(output_path, index=False)
print(f"Saved scores → {output_path}")

# ─── Cell 8: Compute AUROC (GO/NO-GO #1) ─────────────────────────────────────

print("\n" + "=" * 60)
print("GO/NO-GO GATE #1: MRE AUROC on clean MAGE")
print("=" * 60)

labels = df["label"].values

for score_col in score_cols:
    scores = df[score_col].values
    valid = ~np.isnan(scores)

    if valid.sum() < 10:
        print(f"  {score_col}: insufficient valid scores")
        continue

    # MRE: lower = more machine-like, so negate for AUROC
    try:
        auroc = roc_auc_score(labels[valid], -scores[valid])
    except ValueError:
        auroc = 0.5

    # TPR at low FPR
    fpr, tpr, _ = roc_curve(labels[valid], -scores[valid])
    idx_1 = np.searchsorted(fpr, 0.01, side="right") - 1
    idx_5 = np.searchsorted(fpr, 0.05, side="right") - 1
    tpr_1 = tpr[max(0, idx_1)]
    tpr_5 = tpr[max(0, idx_5)]

    print(f"  {score_col}: AUROC={auroc:.4f}  TPR@1%FPR={tpr_1:.4f}  TPR@5%FPR={tpr_5:.4f}")

# Best score
best_col = max(score_cols, key=lambda c: roc_auc_score(labels[~np.isnan(df[c].values)], -df[c].values[~np.isnan(df[c].values)]) if (~np.isnan(df[c].values)).sum() > 10 else 0)
best_auroc = roc_auc_score(labels[~np.isnan(df[best_col].values)], -df[best_col].values[~np.isnan(df[best_col].values)])

print(f"\nBest: {best_col} = {best_auroc:.4f}")

if best_auroc >= 0.85:
    print("🟢 GO: AUROC ≥ 0.85. Core premise validated. Proceed to Phase 2.")
elif best_auroc >= 0.70:
    print("🟡 MARGINAL: AUROC 0.70-0.85. Might improve with DC/DTD. Proceed cautiously.")
else:
    print("🔴 NO-GO: AUROC < 0.70. Core premise is weak. Consider pivoting.")

# ─── Cell 9: Per-generator breakdown ─────────────────────────────────────────

if "generator" in df.columns:
    print("\nPer-generator AUROC:")
    generators = df[df["label"] == 1]["generator"].unique()
    for gen in sorted(generators):
        gen_df = df[(df["label"] == 0) | (df["generator"] == gen)]
        if len(gen_df) < 50:
            continue
        gen_labels = gen_df["label"].values
        gen_scores = gen_df["mre_mean"].values
        valid = ~np.isnan(gen_scores)
        if valid.sum() < 10:
            continue
        try:
            auroc = roc_auc_score(gen_labels[valid], -gen_scores[valid])
            print(f"  {gen}: AUROC={auroc:.4f} (n={len(gen_df)})")
        except ValueError:
            print(f"  {gen}: AUROC computation failed (n={len(gen_df)})")

# ─── Cell 10: Quick visualization ────────────────────────────────────────────

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Distribution of MRE scores by label
for i, ratio in enumerate(MASK_RATIOS):
    col = f"mre_r{ratio:.2f}"
    human_scores = df[df["label"]==0][col].dropna()
    machine_scores = df[df["label"]==1][col].dropna()

    axes[i].hist(human_scores, bins=50, alpha=0.5, label="Human", color="blue", density=True)
    axes[i].hist(machine_scores, bins=50, alpha=0.5, label="Machine", color="red", density=True)
    axes[i].set_title(f"MRE (mask ratio={ratio})")
    axes[i].set_xlabel("NLL")
    axes[i].legend()
    axes[i].set_ylabel("Density")

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "mre_distributions.png"), dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved distribution plot → {RESULTS_DIR}/mre_distributions.png")

# ROC curves
fig, ax = plt.subplots(figsize=(8, 8))
for score_col in score_cols:
    scores = df[score_col].values
    valid = ~np.isnan(scores)
    if valid.sum() < 10:
        continue
    fpr, tpr, _ = roc_curve(labels[valid], -scores[valid])
    auroc = roc_auc_score(labels[valid], -scores[valid])
    ax.plot(fpr, tpr, label=f"{score_col} (AUROC={auroc:.3f})")

ax.plot([0,1], [0,1], "k--", alpha=0.3, label="Random")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("MRE ROC Curves")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "mre_roc_curves.png"), dpi=150, bbox_inches="tight")
plt.show()

print("\n" + "=" * 60)
print("NOTEBOOK 2 COMPLETE")
print("=" * 60)
print(f"Results saved to: {RESULTS_DIR}/")
print("Next: Run Notebook 3 (DC + DTD scoring) or Notebook 4 (baselines)")
