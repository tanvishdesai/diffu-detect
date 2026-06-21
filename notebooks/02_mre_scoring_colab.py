"""
==========================================================================
  DiffuDetect — Notebook 2 (GOOGLE COLAB port): MRE Scoring
==========================================================================

Same MRE scoring as 02_mre_scoring.py, but self-contained for Google Colab:
  - pulls the preprocessed data from Kaggle (no Kaggle notebook needed)
  - hardened 4-bit loader for LLaDA-8B / Dream-7B with a disk-offload fallback
  - prints GPU / RAM / "quantization active" so an OOM is diagnosable, not silent

HOW TO RUN ON COLAB
  1. Runtime → Change runtime type → GPU.
       * T4 (16 GB, free)  → keep 4-bit (default). 8B fits in ~6 GB *if* 4-bit
         engages. This is the SAME card as Kaggle, so it only helps over Kaggle
         if your failure was CPU-RAM (use a High-RAM runtime) — watch the printout.
       * L4 (24 GB) / A100 (Colab Pro) → 4-bit is trivially safe; you may also
         set QUANTIZE_BITS=None to run fp16 and rule quantization out entirely.
  2. Put your Kaggle creds in Colab Secrets (🔑 left sidebar) as
        KAGGLE_USERNAME  and  KAGGLE_KEY
     (recommended). If you skip this, the notebook will prompt for them.
     ⚠ Rotate the key on Kaggle (Account → API → Expire Token) when done — a key
       that has been pasted into chat/notebooks should be treated as compromised.
  3. Run the whole file (it is plain top-to-bottom Python; you can paste it into
     one Colab cell, or split on the "# ─── Cell" markers).
"""

# ─── Cell 1: Install deps ────────────────────────────────────────────────────
# THE ROOT CAUSE of the LLaDA/Dream load failure was NOT VRAM — it was a missing
# / too-old bitsandbytes. transformers 5.x requires bitsandbytes>=0.46.1 for
# 4-bit; without it from_pretrained raises before touching the GPU. We install it
# (and accelerate) at runtime so this can't bite again. Set INSTALL_DEPS=False if
# your environment already has bitsandbytes>=0.46.1.
#   ⚠ transformers 5.0 ALSO breaks LLaDA/Dream's trust_remote_code modeling files
#     ("Could not import module 'PreTrainedModel'"). Their custom code targets
#     transformers 4.x, so we pin transformers<5. If the install DOWNGRADES
#     transformers (you'll see it below), you MUST do Runtime → Restart session
#     and run again — a 5.x already imported into the kernel won't go away
#     otherwise.
INSTALL_DEPS = True
if INSTALL_DEPS:
    import subprocess as _sp, sys as _sys
    _sp.run([_sys.executable, "-m", "pip", "install", "-q", "-U",
             "transformers>=4.46,<5", "bitsandbytes>=0.46.1", "accelerate>=0.33",
             "kaggle", "psutil", "pyarrow"], check=False)
    try:
        import transformers as _tf
        print(f"transformers pinned to {_tf.__version__} (need <5 for LLaDA/Dream "
              f"remote code). If this is still 5.x, RESTART THE SESSION now.")
    except Exception:
        print("Installed transformers<5 + bitsandbytes>=0.46.1 + accelerate. "
              "If transformers had to downgrade, RESTART THE SESSION before running on.")

import os
import sys
import time
import glob
import subprocess
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve

# ─── Cell 2: Kaggle credentials (Colab Secrets → getpass fallback) ───────────
# Never hardcode the key into this file. Prefer Colab Secrets; fall back to a
# runtime prompt. Both just populate the env vars the Kaggle CLI reads.

def _ensure_kaggle_creds():
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return
    # 1) Colab Secrets (recommended)
    try:
        from google.colab import userdata  # type: ignore
        u = userdata.get("KAGGLE_USERNAME")
        k = userdata.get("KAGGLE_KEY")
        if u and k:
            os.environ["KAGGLE_USERNAME"], os.environ["KAGGLE_KEY"] = u, k
            print("Kaggle creds loaded from Colab Secrets.")
            return
    except Exception:
        pass
    # 2) Interactive prompt (key hidden)
    from getpass import getpass
    os.environ["KAGGLE_USERNAME"] = input("Kaggle username: ").strip()
    os.environ["KAGGLE_KEY"] = getpass("Kaggle key (hidden): ").strip()

_ensure_kaggle_creds()

# Kaggle CLI also looks for ~/.kaggle/kaggle.json; write it from the env vars so
# the CLI works regardless of how creds were supplied.
def _write_kaggle_json():
    import json
    kdir = os.path.expanduser("~/.kaggle")
    os.makedirs(kdir, exist_ok=True)
    kpath = os.path.join(kdir, "kaggle.json")
    with open(kpath, "w") as f:
        json.dump({"username": os.environ["KAGGLE_USERNAME"],
                   "key": os.environ["KAGGLE_KEY"]}, f)
    os.chmod(kpath, 0o600)

_write_kaggle_json()

# ─── Cell 3: Configuration ───────────────────────────────────────────────────

# Kaggle dataset slugs to download. NB02 needs the preprocessed MAGE parquet.
# If your Kaggle 02 setup attached a SECOND dataset, add its slug here too.
KAGGLE_DATASETS = [
    "vasuaashadesai/diffudetect-data",
    # "vasuaashadesai/<second-dataset-slug>",   # ← add if you used two
]

# === CHANGE THESE FOR YOUR RUN ===
#   "llada-8b"  → 8B diffusion scorer, 4-bit (the one that OOM'd on Kaggle)
#   "dream-7b"  → 7B diffusion scorer, 4-bit
#   "smdm-1.1b" → small scorer (no quant) — already ran fine on Kaggle
#   "mdlm-110m" → tiny ablation scorer
MODEL_NAME = "llada-8b"

MODEL_PRESETS = {
    "smdm-1.1b": {
        "loader": "smdm",
        "hf_repo": "nieshen/SMDM",
        "hf_checkpoint": "mdm_safetensors/mdm-1028M-1600e18.safetensors",
        "smdm_config_name": "Diff_LLaMA_1028M",
        "tokenizer_repo": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
        "mask_token_id": 32000,
        "smdm_root": "/content/SMDM",
    },
    "llada-8b": {
        "loader": "transformers",
        "hf_repo": "GSAI-ML/LLaDA-8B-Instruct",
        "tokenizer_repo": "GSAI-ML/LLaDA-8B-Instruct",
        "model_class": "causal_lm",     # AutoModelForCausalLM
        "mask_token_id": 126336,
    },
    "dream-7b": {
        "loader": "transformers",
        "hf_repo": "Dream-org/Dream-v0-Instruct-7B",
        "tokenizer_repo": "Dream-org/Dream-v0-Instruct-7B",
        "model_class": "automodel",     # AutoModel
        "mask_token_id": 151666,
    },
    "mdlm-110m": {
        "loader": "transformers",
        "hf_repo": "kuleshov-group/mdlm-owt",
        "tokenizer_repo": "gpt2",
        "model_class": None,
        "mask_token_id": None,
    },
}
MODEL_CFG = MODEL_PRESETS[MODEL_NAME]
QUANTIZE_BITS = 4 if MODEL_NAME in ("llada-8b", "dream-7b") else None

MAX_SAMPLES = 2000
MAX_LENGTH = 512
MASK_RATIOS = [0.15, 0.30, 0.50]
NUM_MASK_DRAWS = 16
SEED = 42
DATASET = "mage"

WORK_DIR = "/content"
KAGGLE_DL_DIR = os.path.join(WORK_DIR, "kaggle_data")
RESULTS_DIR = os.path.join(WORK_DIR, "results")
os.makedirs(KAGGLE_DL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
np.random.seed(SEED)
torch.manual_seed(SEED)

# ─── Cell 4: Environment diagnostics ─────────────────────────────────────────
print(f"PyTorch: {torch.__version__}  CUDA available: {torch.cuda.is_available()}")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {p.name} ({p.total_memory/1024**3:.1f} GB)")
try:
    import psutil
    vm = psutil.virtual_memory()
    print(f"System RAM: {vm.total/1024**3:.1f} GB total, {vm.available/1024**3:.1f} GB free")
    if vm.total / 1024**3 < 20:
        print("  ⚠ Low-RAM runtime. If load OOMs on CPU RAM, switch to a High-RAM runtime.")
except ImportError:
    pass

# ─── Cell 5: Download Kaggle data ────────────────────────────────────────────

def _download_kaggle_datasets(slugs, dest):
    for slug in slugs:
        out = os.path.join(dest, slug.split("/")[-1])
        os.makedirs(out, exist_ok=True)
        print(f"Downloading kaggle dataset: {slug} → {out}")
        r = subprocess.run(
            ["kaggle", "datasets", "download", "-d", slug, "-p", out, "--unzip"],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            print(r.stdout); print(r.stderr)
            raise RuntimeError(
                f"Kaggle download failed for {slug}. Check the slug and that "
                f"KAGGLE_USERNAME/KAGGLE_KEY are valid (and the key isn't expired)."
            )
    print("Kaggle downloads complete.")

_download_kaggle_datasets(KAGGLE_DATASETS, KAGGLE_DL_DIR)

# Locate the data parquet wherever it unzipped to.
def _find_data_file(root, dataset):
    for name in (f"{dataset}_quick.parquet", f"{dataset}_full.parquet"):
        hits = glob.glob(os.path.join(root, "**", name), recursive=True)
        if hits:
            return hits[0]
    return None

data_file = _find_data_file(KAGGLE_DL_DIR, DATASET)
if data_file:
    df = pd.read_parquet(data_file)
    print(f"Loaded {len(df)} passages from {data_file}")
else:
    print("Parquet not found in Kaggle download — falling back to HuggingFace MAGE...")
    from datasets import load_dataset
    ds = load_dataset("yaful/MAGE")
    split = "test" if "test" in ds else list(ds.keys())[0]
    df = ds[split].to_pandas()
    df["label"] = 1 - df["label"].astype(int)  # MAGE: 0→machine,1→human → flip
    if "src" in df.columns:
        df["generator"] = df["src"]
        df["domain"] = df["src"].str.split("_").str[0]
    else:
        df["generator"] = df["label"].apply(lambda x: "machine" if x == 1 else "human")
        df["domain"] = "unknown"
    df["dataset"] = "mage"; df["attack"] = "none"
    df["id"] = [f"mage_{i}" for i in range(len(df))]

if MAX_SAMPLES and len(df) > MAX_SAMPLES:
    per_class = MAX_SAMPLES // 2
    df = pd.concat(
        [g.sample(n=min(per_class, len(g)), random_state=SEED)
         for _, g in df.groupby("label", sort=False)],
        ignore_index=True,
    )
    print(f"Subsampled to {len(df)} passages")

print(f"Labels: {df['label'].value_counts().to_dict()}")

# ─── Cell 6: Model loaders ───────────────────────────────────────────────────

from transformers import (
    AutoTokenizer, AutoModel, AutoModelForMaskedLM, AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from types import SimpleNamespace


def load_transformers_diffusion_model(cfg, device, quantize_bits=None):
    """Load a HF masked diffusion model (MDLM / LLaDA / Dream) with optional
    bitsandbytes 4/8-bit. Hardened for Colab: disk-offload fallback, reserved
    GPU headroom, and an explicit 'quantization active' check after load."""
    import gc

    repo = cfg["hf_repo"]
    tok_repo = cfg.get("tokenizer_repo") or repo
    n_gpus = torch.cuda.device_count()

    # LLaDA/Dream remote code needs transformers<5. A 5.x still loaded in the
    # kernel (because the install ran but the session wasn't restarted) is the
    # "Could not import module 'PreTrainedModel'" failure — catch it clearly.
    import transformers as _tf
    if int(_tf.__version__.split(".")[0]) >= 5:
        raise RuntimeError(
            f"transformers {_tf.__version__} is loaded, but LLaDA/Dream remote "
            f"code requires transformers<5. Re-run Cell 1, then Runtime → Restart "
            f"session and run again.")

    tokenizer = AutoTokenizer.from_pretrained(tok_repo, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {
        "pretrained_model_name_or_path": repo,
        "trust_remote_code": True,
        "torch_dtype": torch.float16,
        "low_cpu_mem_usage": True,
    }
    if quantize_bits in (4, 8):
        # Hard-check bitsandbytes version up front — this is the exact thing that
        # failed before ("requires bitsandbytes>=0.46.1"). Fail loud and early.
        try:
            import bitsandbytes as _bnb
            from packaging.version import parse as _vparse
            if _vparse(_bnb.__version__) < _vparse("0.46.1"):
                raise RuntimeError(
                    f"bitsandbytes {_bnb.__version__} < 0.46.1 (transformers 5.x "
                    f"needs >=0.46.1). Re-run Cell 1 with INSTALL_DEPS=True, then "
                    f"Runtime → Restart session and run again.")
        except ImportError:
            raise RuntimeError(
                "bitsandbytes is not installed. Re-run Cell 1 (INSTALL_DEPS=True), "
                "then Runtime → Restart session and run again.")
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=(quantize_bits == 4),
            load_in_8bit=(quantize_bits == 8),
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = "auto"
        offload_dir = os.path.join(WORK_DIR, "_offload")
        os.makedirs(offload_dir, exist_ok=True)
        kwargs["offload_folder"] = offload_dir
        max_mem = {}
        for i in range(n_gpus):
            total = torch.cuda.get_device_properties(i).total_memory
            max_mem[i] = f"{int((total - 2 * 1024**3) / 1024**3)}GiB"  # 2 GB headroom
        max_mem["cpu"] = "8GiB"
        kwargs["max_memory"] = max_mem
    else:
        kwargs["device_map"] = {"": device} if n_gpus <= 1 else "auto"

    def _patch_tied_weights():
        for _, mod in list(sys.modules.items()):
            if not mod:
                continue
            for attr in dir(mod):
                try:
                    obj = getattr(mod, attr)
                    if isinstance(obj, type) and hasattr(obj, "from_pretrained") \
                            and not hasattr(obj, "_tied_weights_keys"):
                        obj._tied_weights_keys = []
                except Exception:
                    pass

    pref = cfg.get("model_class")
    class_order = {
        "causal_lm": [AutoModelForCausalLM, AutoModelForMaskedLM, AutoModel],
        "automodel": [AutoModel, AutoModelForCausalLM, AutoModelForMaskedLM],
    }.get(pref, [AutoModelForMaskedLM, AutoModelForCausalLM, AutoModel])

    model = None
    for ModelClass in class_order:
        try:
            model = ModelClass.from_pretrained(**kwargs)
            print(f"Loaded with {ModelClass.__name__}")
            break
        except Exception as e:
            print(f"  {ModelClass.__name__} failed: {str(e)[:160]}")
            if "tied_weights" in str(e):
                _patch_tied_weights()
                try:
                    model = ModelClass.from_pretrained(**kwargs)
                    print(f"Loaded with {ModelClass.__name__} (after tied-weights patch)")
                    break
                except Exception as e2:
                    print(f"  retry failed: {str(e2)[:160]}")
            gc.collect(); torch.cuda.empty_cache()
    if model is None:
        raise RuntimeError(f"Could not load model {repo}")
    model.eval()

    # Did 4-bit actually engage? If False on a <24 GB GPU, the model loaded fp16
    # and the next forward pass is what will OOM — that is the Kaggle failure.
    if quantize_bits in (4, 8):
        try:
            is_q = any(type(m).__name__ in ("Linear4bit", "Linear8bitLt")
                       for m in model.modules())
            print(f"  Quantization active ({quantize_bits}-bit): {is_q}")
            if not is_q:
                print("  ⚠ Quantization did NOT engage — bitsandbytes may be too "
                      "old or incompatible with this custom model. Upgrade "
                      "bitsandbytes, or use an L4/A100 and set QUANTIZE_BITS=None.")
        except Exception:
            pass
    for i in range(torch.cuda.device_count()):
        a = torch.cuda.memory_allocated(i) / 1024**3
        t = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  GPU {i}: {a:.2f} / {t:.1f} GB allocated")
    return model, tokenizer


def load_smdm_model(cfg, device):
    """Minimal SMDM loader (clones + patches ML-GSAI/SMDM). Only needed if you
    set MODEL_NAME='smdm-1.1b' on Colab; the 8B path above is the usual reason
    to be here."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    smdm_root = cfg["smdm_root"]
    if not os.path.isdir(os.path.join(smdm_root, "lit_gpt")):
        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/ML-GSAI/SMDM.git", smdm_root], check=True)
    # NOTE: SMDM needs the source patches from 02_mre_scoring.py (_patch_smdm_source).
    # If you actually need SMDM on Colab, copy that function over; for the 8B
    # models (the point of this Colab port) it is not used.
    raise NotImplementedError(
        "SMDM on Colab: copy _patch_smdm_source + load_smdm_model from "
        "02_mre_scoring.py. This port targets llada-8b / dream-7b.")


print(f"\nLoading model: {MODEL_NAME} (quantize={QUANTIZE_BITS})...")
_t0 = time.time()
if MODEL_CFG["loader"] == "smdm":
    model, tokenizer = load_smdm_model(MODEL_CFG, DEVICE)
else:
    model, tokenizer = load_transformers_diffusion_model(MODEL_CFG, DEVICE, QUANTIZE_BITS)
print(f"Model loaded in {time.time()-_t0:.1f}s")


def _resolve_input_device(model):
    try:
        if hasattr(model, "hf_device_map") and model.hf_device_map:
            first = next(iter(model.hf_device_map.values()))
            if isinstance(first, int):
                return torch.device(f"cuda:{first}")
            if isinstance(first, str) and first not in ("cpu", "disk"):
                return torch.device(first)
    except Exception:
        pass
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device(DEVICE)

INPUT_DEVICE = _resolve_input_device(model)
print(f"Input device: {INPUT_DEVICE}")

# Mask token id: prefer the verified preset constant.
MASK_TOKEN_ID = MODEL_CFG.get("mask_token_id")
if MASK_TOKEN_ID is None and getattr(tokenizer, "mask_token_id", None) is not None:
    MASK_TOKEN_ID = tokenizer.mask_token_id
elif MASK_TOKEN_ID is None and hasattr(model.config, "mask_token_id"):
    MASK_TOKEN_ID = model.config.mask_token_id
elif MASK_TOKEN_ID is None:
    MASK_TOKEN_ID = (tokenizer.unk_token_id or (len(tokenizer) - 1))
print(f"Mask token ID: {MASK_TOKEN_ID}")

SPECIAL_IDS = set()
for attr in ['bos_token_id', 'eos_token_id', 'pad_token_id', 'cls_token_id', 'sep_token_id']:
    tid = getattr(tokenizer, attr, None)
    if tid is not None:
        SPECIAL_IDS.add(tid)

# ─── Cell 7: MRE scoring ─────────────────────────────────────────────────────

# Dream passes the raw int attention_mask into scaled_dot_product_attention,
# which rejects it. batch=1 no-padding ⇒ all-ones mask ⇒ dropping it is
# equivalent. Probe once, then skip the mask.
_PASS_ATTN = True

@torch.no_grad()
def compute_mre(text, mask_ratio, num_draws=16, max_length=512):
    """Mean NLL of true tokens at masked positions, averaged over num_draws
    random masks. No fixed-length padding; float32 log_softmax at masked
    positions; inputs sent to INPUT_DEVICE for device_map='auto' models."""
    encoding = tokenizer(text, max_length=max_length, truncation=True, return_tensors="pt")
    input_ids = encoding["input_ids"].to(INPUT_DEVICE)
    attention_mask = encoding["attention_mask"].to(INPUT_DEVICE)
    seq_len = input_ids.shape[1]
    draw_nlls = []

    for _ in range(num_draws):
        eligible = torch.ones(seq_len, dtype=torch.bool, device=INPUT_DEVICE)
        if tokenizer.pad_token_id is not None:
            eligible &= (input_ids[0] != tokenizer.pad_token_id)
        for sid in SPECIAL_IDS:
            eligible &= (input_ids[0] != sid)
        eligible_idx = eligible.nonzero(as_tuple=True)[0]
        if len(eligible_idx) == 0:
            continue
        n_mask = max(1, int(len(eligible_idx) * mask_ratio))
        perm = torch.randperm(len(eligible_idx), device=INPUT_DEVICE)[:n_mask]
        mask_positions = eligible_idx[perm]

        masked_input = input_ids.clone()
        masked_input[0, mask_positions] = MASK_TOKEN_ID
        global _PASS_ATTN
        if _PASS_ATTN:
            try:
                outputs = model(input_ids=masked_input, attention_mask=attention_mask)
            except (TypeError, RuntimeError):
                _PASS_ATTN = False
                outputs = model(input_ids=masked_input)
        else:
            outputs = model(input_ids=masked_input)
        if hasattr(outputs, 'logits') and outputs.logits is not None:
            logits = outputs.logits
        elif hasattr(outputs, 'last_hidden_state'):
            logits = outputs.last_hidden_state
        elif isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs

        # Dream (AutoModel) can return hidden states, not logits — project via head.
        vocab = getattr(getattr(model, "config", None), "vocab_size", None)
        if vocab is not None and logits.shape[-1] != vocab:
            head = getattr(model, "lm_head", None)
            if head is None and hasattr(model, "get_output_embeddings"):
                head = model.get_output_embeddings()
            if head is not None:
                logits = head(logits.to(next(head.parameters()).device))

        # Device-safe (device_map='auto' shards across GPUs → logits may be on a
        # different device than mask_positions). Index on logits.device.
        dev = logits.device
        mp = mask_positions.to(dev)
        log_probs = F.log_softmax(logits[0, mp, :].float(), dim=-1)
        true_token_ids = input_ids[0, mask_positions].to(dev)
        token_nlls = -log_probs[torch.arange(len(mp), device=dev), true_token_ids]
        draw_nlls.append(token_nlls.mean().item())

    return float(np.mean(draw_nlls)) if draw_nlls else np.nan

# ─── Cell 8: Run scoring ─────────────────────────────────────────────────────

print(f"\nScoring {len(df)} passages × {len(MASK_RATIOS)} ratios × {NUM_MASK_DRAWS} draws")
all_scores = {f"mre_r{r:.2f}": [] for r in MASK_RATIOS}
all_scores["mre_mean"] = []
errors = []
scoring_start = time.time()

for idx, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="MRE")):
    text = str(row["text"])
    try:
        ratio_scores = {}
        for ratio in MASK_RATIOS:
            ratio_scores[f"mre_r{ratio:.2f}"] = compute_mre(
                text, mask_ratio=ratio, num_draws=NUM_MASK_DRAWS, max_length=MAX_LENGTH)
        ratio_scores["mre_mean"] = np.mean(list(ratio_scores.values()))
        for k, v in ratio_scores.items():
            all_scores[k].append(v)
    except Exception as e:
        errors.append((idx, str(e)))
        for k in all_scores:
            all_scores[k].append(np.nan)
    if (idx + 1) % 100 == 0:
        el = time.time() - scoring_start
        print(f"  [{idx+1}/{len(df)}] {(idx+1)/el:.1f} passages/s")

print(f"Scoring done: {time.time()-scoring_start:.0f}s, errors={len(errors)}/{len(df)}")
if errors:
    print(f"  First error: {errors[0][1]}")
    if len(errors) == len(df):
        print("  ⚠ ALL passages errored → forward/NLL path failing (see message above).")

for k, v in all_scores.items():
    df[k] = v
score_cols = list(all_scores.keys())

out_path = os.path.join(RESULTS_DIR, f"scores_{DATASET}_mre_{MODEL_NAME}.parquet")
meta_cols = ["id", "text", "label", "generator", "domain", "dataset", "attack"]
save_cols = [c for c in meta_cols if c in df.columns] + score_cols
df[save_cols].to_parquet(out_path, index=False)
print(f"Saved → {out_path}")
print("Download it from the Colab Files pane, then upload as a Kaggle Dataset "
      "for Notebook 5 (evaluation).")

# ─── Cell 9: GO/NO-GO #1 (within-testbed primary) ────────────────────────────

labels = df["label"].values
if "domain" not in df.columns or df["domain"].isna().all():
    df["domain"] = df["generator"].astype(str).str.split("_").str[0]

print("\n" + "=" * 60)
print("GO/NO-GO #1: MRE within-testbed AUROC")
print("=" * 60)

def _within_domain_auroc(col, min_per_class=20):
    s = df[col].values.astype(float)
    m = np.isfinite(s)
    if m.sum() < 10:
        return np.nan, 0
    flip = roc_auc_score(labels[m], s[m]) < 0.5      # fix orientation ONCE
    s_or = -s if flip else s
    aurocs = []
    for _, g in df.groupby("domain"):
        idx = g.index.values
        y, sv = labels[idx], s_or[idx]
        v = np.isfinite(sv); y, sv = y[v], sv[v]
        if (y == 0).sum() < min_per_class or (y == 1).sum() < min_per_class:
            continue
        try:
            aurocs.append(roc_auc_score(y, sv))
        except ValueError:
            pass
    return (float(np.mean(aurocs)) if aurocs else np.nan), len(aurocs)

best_col, best_wt, best_ntb = None, -1, 0
for col in score_cols:
    a, ntb = _within_domain_auroc(col)
    pooled = max(roc_auc_score(labels[np.isfinite(df[col].values)],
                               df[col].values[np.isfinite(df[col].values)]),
                 roc_auc_score(labels[np.isfinite(df[col].values)],
                               -df[col].values[np.isfinite(df[col].values)]))
    print(f"  {col}: within-testbed={a:.4f} ({ntb} domains)  pooled={pooled:.4f}")
    if np.isfinite(a) and a > best_wt:
        best_wt, best_col, best_ntb = a, col, ntb

print(f"\nBest within-testbed (primary): {best_col} = {best_wt:.4f} over {best_ntb} domains")
if best_wt >= 0.85:
    print("🟢 GO: within-testbed ≥ 0.85.")
elif best_wt >= 0.70:
    print("🟡 MARGINAL 0.70–0.85: real signal; DC/DTD or combiner may push it over.")
else:
    print("🔴 NO-GO < 0.70 even within-testbed.")

print("\nNOTEBOOK 2 (COLAB) COMPLETE")
