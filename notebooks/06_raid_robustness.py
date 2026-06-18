"""
==========================================================================
  DiffuDetect — Kaggle Notebook 6: RAID Robustness Scoring
  THE DECISIVE EXPERIMENT — Phase 3
==========================================================================

PURPOSE:
  - Score RAID dataset (clean + attacked passages) with DiffuDetect + baselines
  - Compute ΔAUROC per attack type
  - This is THE experiment that determines if the paper is publishable

KAGGLE SETTINGS:
  - GPU: T4 x1 (required)
  - Internet: ON
  - Accelerator: GPU T4

STRATEGY:
  Split this across multiple Kaggle accounts by changing MODEL/METHOD below.
  Each run handles one (model × method) combination on the RAID dataset.
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

# ─── Config — CHANGE THESE PER RUN ──────────────────────────────────────────

# Which method to run in this notebook instance:
#   "mre"              → DiffuDetect MRE with SMDM
#   "fast_detectgpt"   → Fast-DetectGPT baseline
#   "classical"        → Classical baselines
RUN_METHOD = "mre"

# Model config (matches RUN_METHOD)
MODEL_CONFIGS = {
    "mre": {
        "repo": "nieshen/SMDM-1.1b",
        "name": "smdm-1.1b",
        "type": "diffusion",
        "quantize": None,
    },
    "fast_detectgpt": {
        "repo": "EleutherAI/gpt-neo-2.7B",
        "name": "gpt-neo-2.7b",
        "type": "ar",
        "quantize": None,
    },
    "classical": {
        "repo": "EleutherAI/gpt-neo-2.7B",
        "name": "gpt-neo-2.7b",
        "type": "ar",
        "quantize": None,
    },
}

MAX_SAMPLES = 3000          # RAID can be large; subsample for speed
MAX_LENGTH = 512
MRE_MASK_RATIOS = [0.15, 0.30, 0.50]
MRE_NUM_DRAWS = 12          # Slightly fewer for speed on RAID
FDGPT_NUM_PERTURBATIONS = 30

DATA_DIR = "/kaggle/input/diffudetect-data/data"
RESULTS_DIR = "/kaggle/working/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

config = MODEL_CONFIGS[RUN_METHOD]
print(f"Method: {RUN_METHOD}")
print(f"Model: {config['name']} ({config['repo']})")

# ─── Load RAID data ──────────────────────────────────────────────────────────

raid_file = os.path.join(DATA_DIR, "raid_full.parquet")
if os.path.exists(raid_file):
    df = pd.read_parquet(raid_file)
    print(f"Loaded RAID from Parquet: {len(df)} rows")
else:
    print("Loading RAID from HuggingFace...")
    from datasets import load_dataset
    ds = load_dataset("liamdugan/raid", trust_remote_code=True)
    split = "test" if "test" in ds else ("train" if "train" in ds else list(ds.keys())[0])
    df = ds[split].to_pandas()

    # Standardize columns
    for src, dst in [("generation","text"),("model","generator"),("attack_name","attack"),("domain","domain")]:
        if src in df.columns: df = df.rename(columns={src: dst})
    if "label" not in df.columns:
        if "generator" in df.columns:
            df["label"] = (df["generator"] != "human").astype(int)
    if "domain" not in df.columns: df["domain"] = "unknown"
    if "attack" not in df.columns: df["attack"] = "none"
    df["attack"] = df["attack"].fillna("none").str.lower().str.strip()
    df["dataset"] = "raid"
    df["id"] = [f"raid_{i}" for i in range(len(df))]
    df["label"] = df["label"].astype(int)

# Subsample: keep proportional attack distribution
if MAX_SAMPLES and len(df) > MAX_SAMPLES:
    # Stratified sampling by attack type
    df = df.groupby("attack", group_keys=False).apply(
        lambda x: x.sample(n=min(MAX_SAMPLES // max(1, df["attack"].nunique()), len(x)), random_state=SEED)
    ).reset_index(drop=True)
    print(f"Subsampled to {len(df)} passages")

print(f"\nRAID data summary:")
print(f"  Total: {len(df)}")
print(f"  Labels: {df['label'].value_counts().to_dict()}")
print(f"  Attacks:\n{df['attack'].value_counts().to_string()}")

clean_df = df[df["attack"] == "none"]
attacked_df = df[df["attack"] != "none"]
print(f"\n  Clean: {len(clean_df)}, Attacked: {len(attacked_df)}")

# ─── Load model ──────────────────────────────────────────────────────────────

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM, AutoModel

print(f"\nLoading model: {config['repo']}...")

tokenizer = AutoTokenizer.from_pretrained(config["repo"], trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

if config["type"] == "ar":
    model = AutoModelForCausalLM.from_pretrained(
        config["repo"], torch_dtype=torch.float16, device_map={"": DEVICE}
    )
else:
    model = None
    for Cls in [AutoModelForMaskedLM, AutoModel]:
        try:
            model = Cls.from_pretrained(
                config["repo"], trust_remote_code=True,
                torch_dtype=torch.float16, device_map={"": DEVICE}
            )
            break
        except: continue
    if model is None: raise RuntimeError(f"Cannot load {config['repo']}")

model.eval()
print(f"Model loaded. GPU: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

# Get mask token (for diffusion models)
MASK_TOKEN_ID = None
if config["type"] == "diffusion":
    for check in [
        lambda: tokenizer.mask_token_id if hasattr(tokenizer, 'mask_token_id') and tokenizer.mask_token_id else None,
        lambda: model.config.mask_token_id if hasattr(model.config, 'mask_token_id') else None,
        lambda: tokenizer.convert_tokens_to_ids("[MASK]") if "[MASK]" in tokenizer.get_vocab() else None,
        lambda: tokenizer.convert_tokens_to_ids("<mask>") if "<mask>" in tokenizer.get_vocab() else None,
        lambda: tokenizer.unk_token_id,
    ]:
        v = check()
        if v is not None: MASK_TOKEN_ID = v; break
    if MASK_TOKEN_ID is None: MASK_TOKEN_ID = len(tokenizer) - 1
    print(f"Mask token ID: {MASK_TOKEN_ID}")

SPECIAL_IDS = set()
for attr in ['bos_token_id','eos_token_id','pad_token_id','cls_token_id','sep_token_id']:
    v = getattr(tokenizer, attr, None)
    if v is not None: SPECIAL_IDS.add(v)

# ─── Scoring functions ───────────────────────────────────────────────────────

@torch.no_grad()
def score_mre(text):
    """MRE scoring for diffusion model."""
    enc = tokenizer(text, max_length=MAX_LENGTH, truncation=True, padding="max_length", return_tensors="pt")
    ids = enc["input_ids"].to(DEVICE); attn = enc["attention_mask"].to(DEVICE)
    seq_len = ids.shape[1]

    results = {}
    all_vals = []
    for ratio in MRE_MASK_RATIOS:
        draw_nlls = []
        for _ in range(MRE_NUM_DRAWS):
            eligible = torch.ones(seq_len, dtype=torch.bool, device=DEVICE)
            if tokenizer.pad_token_id is not None: eligible &= (ids[0] != tokenizer.pad_token_id)
            for sid in SPECIAL_IDS: eligible &= (ids[0] != sid)
            eidx = eligible.nonzero(as_tuple=True)[0]
            n = max(1, int(len(eidx) * ratio))
            perm = torch.randperm(len(eidx), device=DEVICE)[:n]
            mpos = eidx[perm]

            m_ids = ids.clone(); m_ids[0, mpos] = MASK_TOKEN_ID
            try: out = model(input_ids=m_ids, attention_mask=attn)
            except TypeError: out = model(input_ids=m_ids)
            logits = out.logits if hasattr(out, 'logits') else (out[0] if isinstance(out, tuple) else out)

            lp = F.log_softmax(logits, dim=-1)
            nll = -lp[0, mpos, :][torch.arange(len(mpos)), ids[0, mpos]].mean().item()
            draw_nlls.append(nll)

        mean_nll = np.mean(draw_nlls)
        results[f"mre_r{ratio:.2f}"] = mean_nll
        all_vals.append(mean_nll)

    results["mre_mean"] = np.mean(all_vals)
    return results


@torch.no_grad()
def score_classical(text):
    """Classical AR baselines."""
    enc = tokenizer(text, max_length=MAX_LENGTH, truncation=True, padding="max_length", return_tensors="pt")
    ids = enc["input_ids"].to(DEVICE); attn = enc["attention_mask"].to(DEVICE)
    out = model(input_ids=ids, attention_mask=attn)
    logits = out.logits[:, :-1, :]; labels = ids[:, 1:]; mask = attn[:, 1:].float()
    n = mask.sum().item()
    if n == 0: return {k: 0.0 for k in ["cls_log_likelihood","cls_mean_rank","cls_mean_entropy","cls_perplexity"]}

    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    tok_lp = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    mean_ll = (tok_lp * mask).sum() / n

    _, sorted_idx = probs.sort(dim=-1, descending=True)
    ranks = torch.zeros_like(labels, dtype=torch.float)
    for p in range(labels.shape[1]):
        if mask[0,p] == 0: continue
        r = (sorted_idx[0,p] == labels[0,p]).nonzero(as_tuple=True)[0]
        ranks[0,p] = (r[0].float()+1) if len(r)>0 else probs.shape[-1]
    mean_rank = (ranks * mask).sum() / n

    ent = -(probs * (probs+1e-10).log()).sum(dim=-1)
    mean_ent = (ent * mask).sum() / n

    return {
        "cls_log_likelihood": mean_ll.item(),
        "cls_mean_rank": mean_rank.item(),
        "cls_mean_entropy": mean_ent.item(),
        "cls_perplexity": torch.exp(-mean_ll).item(),
    }


@torch.no_grad()
def score_fast_detectgpt(text, n_perturb=30):
    """Fast-DetectGPT scoring."""
    enc = tokenizer(text, max_length=MAX_LENGTH, truncation=True, padding="max_length", return_tensors="pt")
    ids = enc["input_ids"].to(DEVICE); attn = enc["attention_mask"].to(DEVICE)
    mask = attn[:, 1:].float(); n = mask.sum()

    out = model(input_ids=ids, attention_mask=attn)
    logits = out.logits[:, :-1, :]
    log_probs = F.log_softmax(logits, dim=-1)
    orig_lp = log_probs.gather(dim=-1, index=ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    orig_ll = (orig_lp * mask).sum() / n

    samp_probs = F.softmax(logits, dim=-1)
    p_lls = []
    for _ in range(n_perturb):
        perturbed = ids.clone()
        for pos in range(samp_probs.shape[1]):
            if attn[0, pos+1] == 0: continue
            perturbed[0, pos+1] = torch.multinomial(samp_probs[0, pos], 1).item()
        p_out = model(input_ids=perturbed, attention_mask=attn)
        p_lp = F.log_softmax(p_out.logits[:, :-1, :], dim=-1)
        p_tok_lp = p_lp.gather(dim=-1, index=perturbed[:, 1:].unsqueeze(-1)).squeeze(-1)
        p_ll = (p_tok_lp * mask).sum() / n
        p_lls.append(p_ll.item())

    p_mean = np.mean(p_lls); p_std = np.std(p_lls) + 1e-8
    return {
        "fdgpt_curvature": (orig_ll.item() - p_mean) / p_std,
        "fdgpt_original_ll": orig_ll.item(),
    }

# ─── Run scoring ─────────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print(f"SCORING RAID with {RUN_METHOD} ({len(df)} passages)")
print(f"{'='*60}")

score_func = {"mre": score_mre, "classical": score_classical, "fast_detectgpt": score_fast_detectgpt}[RUN_METHOD]

results = []
errors = 0
start = time.time()

for idx, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc=f"RAID {RUN_METHOD}")):
    try:
        scores = score_func(str(row["text"]))
        results.append(scores)
    except Exception as e:
        errors += 1
        # Fill with NaN
        if results:
            results.append({k: np.nan for k in results[0].keys()})
        else:
            results.append({})

    if (idx+1) % 200 == 0:
        elapsed = time.time() - start
        rate = (idx+1) / elapsed
        print(f"  [{idx+1}/{len(df)}] {rate:.1f}/s, ~{(len(df)-idx-1)/rate/60:.0f}min left, {errors} errors")

total_time = time.time() - start
print(f"\nScoring done: {total_time:.0f}s ({total_time/len(df):.2f}s/passage), {errors} errors")

# Add scores to DataFrame
scores_df = pd.DataFrame(results)
for c in scores_df.columns:
    df[c] = scores_df[c].values

# Save
meta = ["id","text","label","generator","domain","dataset","attack"]
save_cols = [c for c in meta if c in df.columns] + list(scores_df.columns)
outfile = os.path.join(RESULTS_DIR, f"scores_raid_{RUN_METHOD}_{config['name']}.parquet")
df[save_cols].to_parquet(outfile, index=False)
print(f"Saved → {outfile}")

# ─── Quick robustness analysis ───────────────────────────────────────────────

print(f"\n{'='*60}")
print("QUICK ROBUSTNESS ANALYSIS")
print(f"{'='*60}")

SCORE_DIRECTIONS = {
    "mre_mean": "lower_is_machine", "mre_r0.15": "lower_is_machine",
    "mre_r0.30": "lower_is_machine", "mre_r0.50": "lower_is_machine",
    "fdgpt_curvature": "higher_is_machine",
    "cls_log_likelihood": "higher_is_machine",
    "cls_mean_rank": "lower_is_machine",
    "cls_mean_entropy": "lower_is_machine",
}

score_cols = [c for c in scores_df.columns if c in SCORE_DIRECTIONS]
clean_data = df[df["attack"] == "none"]
attacks = [a for a in df["attack"].unique() if a != "none"]

for col in score_cols:
    direction = SCORE_DIRECTIONS.get(col, "higher_is_machine")
    cv = clean_data[col].values; valid_c = np.isfinite(cv)
    if valid_c.sum() < 10: continue
    s_c = -cv[valid_c] if direction == "lower_is_machine" else cv[valid_c]
    try: auroc_c = roc_auc_score(clean_data["label"].values[valid_c], s_c)
    except: continue

    print(f"\n  {col} — Clean AUROC: {auroc_c:.4f}")
    for attack in sorted(attacks):
        att_data = df[df["attack"] == attack]
        if len(att_data) < 30: continue
        av = att_data[col].values; valid_a = np.isfinite(av)
        if valid_a.sum() < 10: continue
        s_a = -av[valid_a] if direction == "lower_is_machine" else av[valid_a]
        try:
            auroc_a = roc_auc_score(att_data["label"].values[valid_a], s_a)
            delta = auroc_c - auroc_a
            print(f"    {attack:<20s} AUROC={auroc_a:.4f}  ΔAUROC={delta:+.4f}")
        except: pass

print(f"\n{'='*60}")
print("NOTEBOOK 6 COMPLETE — RAID Robustness Scoring Done")
print("=" * 60)
print(f"Upload {outfile} and combine with other notebooks in Notebook 5")
print("to compute the final GO/NO-GO #2 decision.")
