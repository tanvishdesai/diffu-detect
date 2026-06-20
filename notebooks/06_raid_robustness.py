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
        "loader": "smdm",
        "hf_repo": "nieshen/SMDM",
        "hf_checkpoint": "mdm_safetensors/mdm-1028M-1600e18.safetensors",
        "mask_token_id": 32000,
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
model_label = config.get("hf_repo") or config.get("repo")
print(f"Method: {RUN_METHOD}")
print(f"Model: {config['name']} ({model_label})")

# ─── Load RAID data ──────────────────────────────────────────────────────────

raid_file = os.path.join(DATA_DIR, "raid_full.parquet")
if os.path.exists(raid_file):
    df = pd.read_parquet(raid_file)
    print(f"Loaded RAID from Parquet: {len(df)} rows")
else:
    print("Loading RAID from HuggingFace...")
    from datasets import load_dataset
    ds = load_dataset("liamdugan/raid")
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
from types import SimpleNamespace

print(f"\nLoading model: {model_label}...")

if config.get("loader") == "smdm":
    import re
    import subprocess
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    smdm_root = "/tmp/SMDM"
    if not os.path.isdir(os.path.join(smdm_root, "lit_gpt")):
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/ML-GSAI/SMDM.git", smdm_root],
            check=True,
        )

    # ── Patch SMDM source (same as notebooks 02/03) ──────────────────────
    def _patch_smdm_source_nb6(smdm_root):
        """Patch SMDM source to remove flash_attn, xformers, rotary, lightning deps."""
        model_py = os.path.join(smdm_root, "lit_gpt", "model.py")
        if os.path.exists(model_py):
            with open(model_py, "r") as f: src = f.read()
            if "flash_attn" in src:
                src = src.replace("from flash_attn import flash_attn_func",
                    "# flash_attn patched out — using PyTorch SDPA fallback")
                src = re.sub(r'flash_attn_func\s*\(([^)]*)\)', r'F.scaled_dot_product_attention(\1)', src)
                if "import torch.nn.functional as F" not in src:
                    src = "import torch.nn.functional as F\n" + src
                src = src.replace("from xformers.ops import SwiGLU",
                    "# xformers patched out\n"
                    "class SwiGLU(torch.nn.Module):\n"
                    "    def __init__(self, in_features, hidden_features, out_features, bias=True):\n"
                    "        super().__init__()\n"
                    "        self.w1 = torch.nn.Linear(in_features, hidden_features, bias=bias)\n"
                    "        self.w2 = torch.nn.Linear(in_features, hidden_features, bias=bias)\n"
                    "        self.w3 = torch.nn.Linear(hidden_features, out_features, bias=bias)\n"
                    "        self.act = torch.nn.SiLU()\n"
                    "    def forward(self, x):\n"
                    "        return self.w3(self.act(self.w1(x)) * self.w2(x))\n")
                with open(model_py, "w") as f: f.write(src)
                print("  Patched model.py")
        diff_py = os.path.join(smdm_root, "lit_gpt", "diffmodel.py")
        if os.path.exists(diff_py):
            with open(diff_py, "r") as f: src = f.read()
            changed = False
            if "from flash_attn import flash_attn_func" in src:
                src = src.replace("from flash_attn import flash_attn_func",
                    "# flash_attn patched out"); changed = True
            if "from xformers.ops import SwiGLU" in src:
                src = src.replace("from xformers.ops import SwiGLU",
                    "# xformers patched out\n"
                    "class SwiGLU(torch.nn.Module):\n"
                    "    def __init__(self, in_features, hidden_features, out_features=None, bias=True, _pack_weights=False):\n"
                    "        super().__init__()\n"
                    "        out_features = out_features or in_features\n"
                    "        self.w1 = torch.nn.Linear(in_features, hidden_features, bias=bias)\n"
                    "        self.w2 = torch.nn.Linear(in_features, hidden_features, bias=bias)\n"
                    "        self.w3 = torch.nn.Linear(hidden_features, out_features, bias=bias)\n"
                    "        self.act = torch.nn.SiLU()\n"
                    "    def forward(self, x):\n"
                    "        return self.w3(self.act(self.w1(x)) * self.w2(x))\n"); changed = True
            if "from .fused_rotary_embedding import apply_rotary_emb_func" in src:
                src = src.replace("from .fused_rotary_embedding import apply_rotary_emb_func",
                    "# fused_rotary_embedding patched out\n"
                    "def apply_rotary_emb_func(x, cos, sin, interleaved=False, inplace=False):\n"
                    "    rot_dim = cos.shape[-1] * 2\n"
                    "    x_rot = x[..., :rot_dim]; x_pass = x[..., rot_dim:]\n"
                    "    x1 = x_rot[..., : rot_dim // 2]; x2 = x_rot[..., rot_dim // 2 :]\n"
                    "    cos = cos[:x.shape[1]].unsqueeze(0).unsqueeze(2)\n"
                    "    sin = sin[:x.shape[1]].unsqueeze(0).unsqueeze(2)\n"
                    "    o1 = x1 * cos - x2 * sin; o2 = x2 * cos + x1 * sin\n"
                    "    return torch.cat([torch.cat([o1, o2], dim=-1), x_pass], dim=-1).to(x.dtype)\n"); changed = True
            if "from lightning_utilities.core.imports import RequirementCache" in src:
                src = src.replace("from lightning_utilities.core.imports import RequirementCache",
                    "# lightning_utilities patched out\n"
                    "class RequirementCache:\n"
                    "    def __init__(self, *a, **kw): pass\n"
                    "    def __bool__(self): return False\n"); changed = True
            if changed:
                if "import torch.nn.functional as F" not in src:
                    src = "import torch.nn.functional as F\n" + src
                with open(diff_py, "w") as f: f.write(src)
                print("  Patched diffmodel.py")
        config_py = os.path.join(smdm_root, "lit_gpt", "config.py")
        if os.path.exists(config_py):
            with open(config_py, "r") as f: src = f.read()
            if "# config-fully-patched" not in src:
                new_src = ("# config-fully-patched\n"
                    "from dataclasses import dataclass\n"
                    "from typing import Any, Literal, Optional, Type\n\n"
                    "import torch\nfrom typing_extensions import Self\n\n"
                    "def find_multiple(n: int, k: int) -> int:\n"
                    "    if n % k == 0: return n\n"
                    "    return n + k - (n % k)\n\n")
                idx = src.find("@dataclass")
                if idx != -1:
                    body = src[idx:]
                    body = re.sub(r'@property\s+def norm_class\(self\).*?(?=\n    @|\nconfigs)',
                        '@property\n    def norm_class(self) -> Type:\n'
                        '        if "RMSNorm" in self._norm_class:\n'
                        '            if hasattr(torch.nn, "RMSNorm"): return torch.nn.RMSNorm\n'
                        '            class _RMSNorm(torch.nn.Module):\n'
                        '                def __init__(self, d, eps=1e-5):\n'
                        '                    super().__init__()\n'
                        '                    self.eps = eps\n'
                        '                    self.weight = torch.nn.Parameter(torch.ones(d))\n'
                        '                def forward(self, x):\n'
                        '                    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight\n'
                        '            return _RMSNorm\n'
                        '        return getattr(torch.nn, self._norm_class)\n\n',
                        body, flags=re.DOTALL)
                    body = re.sub(r'@property\s+def mlp_class\(self\).*?(?=\n    @|\nconfigs)',
                        '@property\n    def mlp_class(self) -> Type:\n'
                        '        import lit_gpt.diffmodel as _dm\n'
                        '        return getattr(_dm, self._mlp_class)\n\n',
                        body, flags=re.DOTALL)
                    new_src += body
                with open(config_py, "w") as f: f.write(new_src)
                print("  Patched config.py")
        init_py = os.path.join(smdm_root, "lit_gpt", "__init__.py")
        if os.path.exists(init_py):
            with open(init_py, "r") as f: src = f.read()
            if "# patched-init" not in src:
                with open(init_py, "w") as f:
                    f.write("# patched-init: minimal stub\n")
                print("  Patched __init__.py")

    _patch_smdm_source_nb6(smdm_root)
    # ── End patch ────────────────────────────────────────────────────────

    if smdm_root not in sys.path:
        sys.path.insert(0, smdm_root)

    # Clear stale lit_gpt modules
    stale = [k for k in sys.modules if k == "lit_gpt" or k.startswith("lit_gpt.")]
    for k in stale: del sys.modules[k]

    from lit_gpt.diffmodel import Config, TransEncoder

    ckpt_path = hf_hub_download(
        repo_id=config["hf_repo"],
        filename=config["hf_checkpoint"],
    )
    smdm_config = Config.from_name("Diff_LLaMA_1028M")
    core = TransEncoder(smdm_config).to(DEVICE)
    core.load_state_dict(load_file(ckpt_path, device=DEVICE))
    core.half()
    core.eval()

    class SMDMWrap:
        def __init__(self, m):
            self.model = m
            self.lm_head = m.lm_head
            self.config = SimpleNamespace(mask_token_id=32000)
        def eval(self):
            self.model.eval()
            return self
        def __call__(self, input_ids=None, attention_mask=None, **kwargs):
            return SimpleNamespace(logits=self.model(input_ids))

    model = SMDMWrap(core)
    tokenizer = AutoTokenizer.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
        padding_side="right", use_fast=True,
    )
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    # FIX: pad_token_id must NOT equal mask_token_id!
    tokenizer.pad_token_id = 2  # EOS token; NOT mask_token_id=32000
elif config["type"] == "ar":
    tokenizer = AutoTokenizer.from_pretrained(config["repo"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config["repo"], torch_dtype=torch.float16, device_map={"": DEVICE}
    )
else:
    tokenizer = AutoTokenizer.from_pretrained(config["repo"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
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
    MASK_TOKEN_ID = config.get("mask_token_id")
    if MASK_TOKEN_ID is None:
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
    """MRE scoring for diffusion model.

    No fixed-length padding: batch size is 1, so we score the passage at its
    true length. (v1 padded to 512 with EOS tokens, which SMDM — having no
    attention mask — then attended over, adding noise and wasting compute.)
    """
    enc = tokenizer(text, max_length=MAX_LENGTH, truncation=True, return_tensors="pt")
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
            if len(eidx) == 0: continue
            n = max(1, int(len(eidx) * ratio))
            perm = torch.randperm(len(eidx), device=DEVICE)[:n]
            mpos = eidx[perm]

            m_ids = ids.clone(); m_ids[0, mpos] = MASK_TOKEN_ID
            try: out = model(input_ids=m_ids, attention_mask=attn)
            except TypeError: out = model(input_ids=m_ids)
            logits = out.logits if hasattr(out, 'logits') else (out[0] if isinstance(out, tuple) else out)

            lp = F.log_softmax(logits.float(), dim=-1)
            nll = -lp[0, mpos, :][torch.arange(len(mpos)), ids[0, mpos]].mean().item()
            draw_nlls.append(nll)

        mean_nll = np.mean(draw_nlls) if draw_nlls else np.nan
        results[f"mre_r{ratio:.2f}"] = mean_nll
        all_vals.append(mean_nll)

    results["mre_mean"] = np.nanmean(all_vals) if all_vals else np.nan
    return results


@torch.no_grad()
def score_classical(text):
    """Classical AR baselines."""
    enc = tokenizer(text, max_length=MAX_LENGTH, truncation=True, padding="max_length", return_tensors="pt")
    ids = enc["input_ids"].to(DEVICE); attn = enc["attention_mask"].to(DEVICE)
    out = model(input_ids=ids, attention_mask=attn)
    # FIX: upcast to float32 before softmax/entropy (fp16 entropy over a large
    # vocab → NaN; this is the cls_mean_entropy=nan bug from v1).
    logits = out.logits[:, :-1, :].float(); labels = ids[:, 1:]; mask = attn[:, 1:].float()
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
def score_fast_detectgpt(text):
    """Fast-DetectGPT — ANALYTIC sampling discrepancy (Bao et al., 2024).

    Closed-form white-box curvature: one forward pass per passage, no MC
    perturbation sampling. ~40x faster than v1 and reproduces published AUROC.
    """
    enc = tokenizer(text, max_length=MAX_LENGTH, truncation=True, return_tensors="pt")
    ids = enc["input_ids"].to(DEVICE); attn = enc["attention_mask"].to(DEVICE)
    if ids.shape[1] < 2:
        return {"fdgpt_curvature": np.nan, "fdgpt_original_ll": np.nan}

    logits = model(input_ids=ids, attention_mask=attn).logits[:, :-1, :].float()
    labels = ids[:, 1:]
    lprobs = F.log_softmax(logits, dim=-1)
    probs = lprobs.exp()
    ll = lprobs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    mean_ref = (probs * lprobs).sum(dim=-1)
    var_ref = (probs * lprobs.square()).sum(dim=-1) - mean_ref.square()
    discrepancy = (ll.sum(dim=-1) - mean_ref.sum(dim=-1)) / var_ref.sum(dim=-1).clamp_min(1e-8).sqrt()
    return {
        "fdgpt_curvature": discrepancy.mean().item(),
        "fdgpt_original_ll": ll.mean().item(),
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

SCORE_DIRECTIONS = {}
# FIX: auto-detect direction instead of hardcoding

score_cols = [c for c in scores_df.columns if c not in ["fdgpt_original_ll","fdgpt_perturb_mean_ll"]]
clean_data = df[df["attack"] == "none"]
attacks = [a for a in df["attack"].unique() if a != "none"]

for col in score_cols:
    cv = clean_data[col].values; valid_c = np.isfinite(cv)
    if valid_c.sum() < 10: continue
    try:
        auroc_pos = roc_auc_score(clean_data["label"].values[valid_c], cv[valid_c])
        auroc_neg = roc_auc_score(clean_data["label"].values[valid_c], -cv[valid_c])
        if auroc_pos >= auroc_neg:
            auroc_c = auroc_pos; use_neg = False
        else:
            auroc_c = auroc_neg; use_neg = True
    except: continue

    print(f"\n  {col} — Clean AUROC: {auroc_c:.4f} [{'negated' if use_neg else 'raw'}]")
    for attack in sorted(attacks):
        att_data = df[df["attack"] == attack]
        if len(att_data) < 30: continue
        av = att_data[col].values; valid_a = np.isfinite(av)
        if valid_a.sum() < 10: continue
        s_a = -av[valid_a] if use_neg else av[valid_a]
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
