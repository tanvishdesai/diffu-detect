"""
==========================================================================
  DiffuDetect — Kaggle Notebook 4: Baseline Scoring
==========================================================================

PURPOSE:
  - Run Fast-DetectGPT (the primary baseline to beat)
  - Run Classical baselines (log-lik, rank, entropy)
  - Optionally run DetectGPT (T5 perturbation) and Binoculars
  - Save all baseline scores to Parquet

KAGGLE SETTINGS:
  - GPU: T4 x1 (required)
  - Internet: ON
  - Accelerator: GPU T4
"""

# !pip install -q torch transformers datasets accelerate bitsandbytes \
#     scikit-learn pandas pyarrow tqdm huggingface_hub sentencepiece protobuf

import os, sys, time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42; np.random.seed(SEED); torch.manual_seed(SEED)

# ─── Config ──────────────────────────────────────────────────────────────────

RUN_CLASSICAL = True
RUN_FAST_DETECTGPT = True
RUN_DETECTGPT = False       # Slow; set True if time permits
RUN_BINOCULARS = False       # Needs 2 models; set True if VRAM allows

AR_MODEL_REPO = "EleutherAI/gpt-neo-2.7B"
AR_MODEL_NAME = "gpt-neo-2.7b"
FDGPT_NUM_PERTURBATIONS = 50
MAX_SAMPLES = 2000
MAX_LENGTH = 512

DATA_DIR = "/kaggle/input/diffudetect-data/data"
RESULTS_DIR = "/kaggle/working/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─── Load data ───────────────────────────────────────────────────────────────

data_file = os.path.join(DATA_DIR, "mage_quick.parquet")
if not os.path.exists(data_file):
    from datasets import load_dataset
    ds = load_dataset("yaful/MAGE", trust_remote_code=True)
    split = "test" if "test" in ds else list(ds.keys())[0]
    df = ds[split].to_pandas()
    for src, dst in [("source_model","generator"),("category","domain")]:
        if src in df.columns: df = df.rename(columns={src: dst})
    if "generator" not in df.columns: df["generator"] = df["label"].apply(lambda x: "machine" if x == 1 else "human")
    if "domain" not in df.columns: df["domain"] = "unknown"
    df["dataset"]="mage"; df["attack"]="none"
    df["id"]=[f"mage_{i}" for i in range(len(df))]; df["label"]=df["label"].astype(int)
else:
    df = pd.read_parquet(data_file)

if MAX_SAMPLES and len(df) > MAX_SAMPLES:
    df = df.groupby("label", group_keys=False).apply(
        lambda x: x.sample(n=min(MAX_SAMPLES//2, len(x)), random_state=SEED)
    ).reset_index(drop=True)

print(f"Data: {len(df)} passages")

# ─── Load AR model ───────────────────────────────────────────────────────────

from transformers import AutoTokenizer, AutoModelForCausalLM

print(f"\nLoading AR model: {AR_MODEL_REPO}...")
ar_tokenizer = AutoTokenizer.from_pretrained(AR_MODEL_REPO)
if ar_tokenizer.pad_token is None: ar_tokenizer.pad_token = ar_tokenizer.eos_token

ar_model = AutoModelForCausalLM.from_pretrained(
    AR_MODEL_REPO, torch_dtype=torch.float16, device_map={"": DEVICE}
)
ar_model.eval()
print(f"AR model loaded. GPU mem: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

# =========================================================================
#  CLASSICAL BASELINES
# =========================================================================

if RUN_CLASSICAL:
    print("\n" + "=" * 60)
    print("CLASSICAL BASELINES (log-lik, rank, entropy)")
    print("=" * 60)

    @torch.no_grad()
    def score_classical(text):
        enc = ar_tokenizer(text, max_length=MAX_LENGTH, truncation=True, padding="max_length", return_tensors="pt")
        ids = enc["input_ids"].to(DEVICE); attn = enc["attention_mask"].to(DEVICE)

        out = ar_model(input_ids=ids, attention_mask=attn)
        logits = out.logits[:, :-1, :]; labels = ids[:, 1:]; mask = attn[:, 1:].float()
        n = mask.sum().item()
        if n == 0: return {"cls_log_likelihood":0,"cls_mean_rank":0,"cls_mean_log_rank":0,"cls_mean_entropy":0,"cls_perplexity":0}

        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        # Log-likelihood
        tok_lp = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        mean_ll = (tok_lp * mask).sum() / n

        # Rank
        _, sorted_idx = probs.sort(dim=-1, descending=True)
        ranks = torch.zeros_like(labels, dtype=torch.float)
        for p in range(labels.shape[1]):
            if mask[0,p] == 0: continue
            r = (sorted_idx[0,p] == labels[0,p]).nonzero(as_tuple=True)[0]
            ranks[0,p] = (r[0].float()+1) if len(r) > 0 else probs.shape[-1]
        mean_rank = (ranks * mask).sum() / n
        mean_log_rank = ((ranks+1).log() * mask).sum() / n

        # Entropy
        ent = -(probs * (probs+1e-10).log()).sum(dim=-1)
        mean_ent = (ent * mask).sum() / n

        return {
            "cls_log_likelihood": mean_ll.item(),
            "cls_mean_rank": mean_rank.item(),
            "cls_mean_log_rank": mean_log_rank.item(),
            "cls_mean_entropy": mean_ent.item(),
            "cls_perplexity": torch.exp(-mean_ll).item(),
        }

    cls_results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Classical"):
        try: cls_results.append(score_classical(str(row["text"])))
        except: cls_results.append({k:np.nan for k in ["cls_log_likelihood","cls_mean_rank","cls_mean_log_rank","cls_mean_entropy","cls_perplexity"]})

    cls_df = pd.DataFrame(cls_results)
    for c in cls_df.columns: df[c] = cls_df[c].values

    meta = ["id","text","label","generator","domain","dataset","attack"]
    save_cols = [c for c in meta if c in df.columns] + list(cls_df.columns)
    df[save_cols].to_parquet(os.path.join(RESULTS_DIR, f"scores_mage_classical_{AR_MODEL_NAME}.parquet"), index=False)

    # Classical AUROCs
    print("\nClassical AUROCs:")
    for col, direction in [("cls_log_likelihood","higher"),("cls_mean_rank","lower"),
                          ("cls_mean_entropy","lower"),("cls_perplexity","lower")]:
        if col not in df.columns: continue
        v = df[col].values; valid = ~np.isnan(v)
        if valid.sum() < 10: continue
        s = v[valid] if direction == "higher" else -v[valid]
        try: print(f"  {col}: AUROC={roc_auc_score(df['label'].values[valid], s):.4f}")
        except: pass

# =========================================================================
#  FAST-DETECTGPT
# =========================================================================

if RUN_FAST_DETECTGPT:
    print("\n" + "=" * 60)
    print("FAST-DETECTGPT")
    print("=" * 60)

    @torch.no_grad()
    def score_fast_detectgpt(text, n_perturb=50):
        enc = ar_tokenizer(text, max_length=MAX_LENGTH, truncation=True, padding="max_length", return_tensors="pt")
        ids = enc["input_ids"].to(DEVICE); attn = enc["attention_mask"].to(DEVICE)
        mask = attn[:, 1:].float(); n = mask.sum()

        # Original log-probs
        out = ar_model(input_ids=ids, attention_mask=attn)
        logits = out.logits[:, :-1, :]
        log_probs = F.log_softmax(logits, dim=-1)
        orig_lp = log_probs.gather(dim=-1, index=ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        orig_ll = (orig_lp * mask).sum() / n

        # Sampling distribution
        samp_probs = F.softmax(logits, dim=-1)

        # Perturbation log-likelihoods
        p_lls = []
        for _ in range(n_perturb):
            perturbed = ids.clone()
            for pos in range(samp_probs.shape[1]):
                if attn[0, pos+1] == 0: continue
                perturbed[0, pos+1] = torch.multinomial(samp_probs[0, pos], 1).item()

            p_out = ar_model(input_ids=perturbed, attention_mask=attn)
            p_logits = p_out.logits[:, :-1, :]
            p_log_probs = F.log_softmax(p_logits, dim=-1)
            p_lp = p_log_probs.gather(dim=-1, index=perturbed[:, 1:].unsqueeze(-1)).squeeze(-1)
            p_ll = (p_lp * mask).sum() / n
            p_lls.append(p_ll.item())

        p_mean = np.mean(p_lls); p_std = np.std(p_lls) + 1e-8
        return {
            "fdgpt_curvature": (orig_ll.item() - p_mean) / p_std,
            "fdgpt_original_ll": orig_ll.item(),
            "fdgpt_perturb_mean_ll": p_mean,
        }

    fdgpt_results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Fast-DetectGPT"):
        try: fdgpt_results.append(score_fast_detectgpt(str(row["text"]), FDGPT_NUM_PERTURBATIONS))
        except: fdgpt_results.append({k:np.nan for k in ["fdgpt_curvature","fdgpt_original_ll","fdgpt_perturb_mean_ll"]})

    fdgpt_df = pd.DataFrame(fdgpt_results)
    for c in fdgpt_df.columns: df[c] = fdgpt_df[c].values

    meta = ["id","text","label","generator","domain","dataset","attack"]
    save_cols = [c for c in meta if c in df.columns] + list(fdgpt_df.columns)
    df[save_cols].to_parquet(os.path.join(RESULTS_DIR, f"scores_mage_fast_detectgpt_{AR_MODEL_NAME}.parquet"), index=False)

    valid = ~np.isnan(df["fdgpt_curvature"].values)
    if valid.sum() > 10:
        auroc = roc_auc_score(df["label"].values[valid], df["fdgpt_curvature"].values[valid])
        print(f"\nFast-DetectGPT Curvature AUROC: {auroc:.4f}")

# ─── Summary ─────────────────────────────────────────────────────────────────

del ar_model; torch.cuda.empty_cache()

print("\n" + "=" * 60)
print("NOTEBOOK 4 COMPLETE — Baselines Done")
print("=" * 60)
print(f"Results saved to: {RESULTS_DIR}/")
print("Next: Run Notebook 5 (evaluation and robustness analysis)")
