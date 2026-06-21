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
# For SMDM-1.1B (default):
# !pip install -q torch transformers datasets accelerate safetensors xformers \
#     scikit-learn pandas pyarrow tqdm huggingface_hub sentencepiece protobuf
# !MAX_JOBS=4 pip install -q flash-attn --no-build-isolation
# !git clone --depth 1 https://github.com/ML-GSAI/SMDM.git /tmp/SMDM
#
# For LLaDA-8B / Dream-7B (4-bit) — needs bitsandbytes + accelerate, NO SMDM clone:
# !pip install -q -U "transformers>=4.46,<5" datasets accelerate "bitsandbytes>=0.46.1" \
#     scikit-learn pandas pyarrow tqdm huggingface_hub sentencepiece protobuf
#   ⚠ TWO pins matter (both seen as load failures, NOT VRAM/OOM):
#     1. bitsandbytes>=0.46.1 — older versions are rejected for 4-bit.
#     2. transformers<5 — transformers 5.0 breaks LLaDA/Dream trust_remote_code
#        ("Could not import module 'PreTrainedModel'"). Their custom modeling
#        targets transformers 4.x. Pin <5 (same applies to NB03 for DTD).
#     After installing on Kaggle, RESTART the kernel if transformers had to
#     downgrade, then run. You MUST actually run this pip line (not commented).
#   (Set MODEL_NAME="llada-8b" or "dream-7b" below; QUANTIZE_BITS auto-set to 4.
#    These load via trust_remote_code — no GGUF/llama.cpp: we need raw per-token
#    logits at masked positions, which GGUF/llama.cpp can't provide for these
#    diffusion architectures. bitsandbytes on-the-fly 4-bit is the right path.)
#
# For MDLM-110M quick test (no extra deps): skip the flash-attn / git clone lines.

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
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# ─── Cell 2: Configuration ───────────────────────────────────────────────────

# === CHANGE THESE FOR YOUR RUN ===
#   "smdm-1.1b"  → small diffusion scorer (no quantization, fast, T4 x1)
#   "llada-8b"   → 8B diffusion scorer, 4-bit (set QUANTIZE_BITS=4). Stronger MRE.
#   "dream-7b"   → 7B diffusion scorer, 4-bit. Second large model for robustness.
#   "mdlm-110m"  → tiny ablation scorer
MODEL_NAME = "smdm-1.1b"

# Model presets.
#  - SMDM weights live at nieshen/SMDM (safetensors) and need ML-GSAI/SMDM code.
#  - LLaDA loads as AutoModelForCausalLM (trust_remote_code), mask_token_id=126336.
#  - Dream loads as AutoModel (trust_remote_code), mask_token_id=151666.
#    (Both expose mask_token_id in model.config too — we read it at runtime and
#     only fall back to these constants if the config lookup fails.)
MODEL_PRESETS = {
    "smdm-1.1b": {
        "loader": "smdm",
        "hf_repo": "nieshen/SMDM",
        "hf_checkpoint": "mdm_safetensors/mdm-1028M-1600e18.safetensors",
        "smdm_config_name": "Diff_LLaMA_1028M",
        "tokenizer_repo": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
        "mask_token_id": 32000,
        "smdm_root": "/tmp/SMDM",
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
        "mask_token_id": None,
    },
}
MODEL_CFG = MODEL_PRESETS[MODEL_NAME]
# 4-bit for the 7-8B diffusion models (fits a single 16 GB T4); None for SMDM.
QUANTIZE_BITS = 4 if MODEL_NAME in ("llada-8b", "dream-7b") else None
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
    ds = load_dataset("yaful/MAGE")
    split = "test" if "test" in ds else list(ds.keys())[0]
    df = ds[split].to_pandas()

    # ──────────────────────────────────────────────────────────────────────
    # FIX: MAGE label convention is label=0→machine, label=1→human.
    # We standardize to label=0→human, label=1→machine for all our code.
    # ──────────────────────────────────────────────────────────────────────
    df["label"] = 1 - df["label"].astype(int)
    print("MAGE labels flipped: now label=0→human, label=1→machine")

    # FIX: MAGE uses 'src' column (not 'source_model' or 'category')
    if "src" in df.columns:
        df["generator"] = df["src"]
        # Extract domain from src (format: domain_human or domain_machine_...)
        df["domain"] = df["src"].str.split("_").str[0]
    else:
        for col_src, col_dst in [("source_model", "generator"), ("category", "domain")]:
            if col_src in df.columns:
                df = df.rename(columns={col_src: col_dst})
    if "generator" not in df.columns:
        df["generator"] = df["label"].apply(lambda x: "machine" if x == 1 else "human")
    if "domain" not in df.columns:
        df["domain"] = "unknown"
    df["dataset"] = "mage"
    df["attack"] = "none"
    df["id"] = [f"mage_{i}" for i in range(len(df))]

# Subsample if needed (balanced by label)
if MAX_SAMPLES and len(df) > MAX_SAMPLES:
    per_class = MAX_SAMPLES // 2
    df = pd.concat(
        [
            g.sample(n=min(per_class, len(g)), random_state=SEED)
            for _, g in df.groupby("label", sort=False)
        ],
        ignore_index=True,
    )
    print(f"Subsampled to {len(df)} passages")

print(f"Labels: {df['label'].value_counts().to_dict()}")
if "generator" in df.columns:
    print(f"Generators: {df['generator'].value_counts().to_dict()}")

# ─── Cell 4: Load diffusion model ────────────────────────────────────────────

from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
from types import SimpleNamespace


class SMDMModelWrapper:
    """HuggingFace-like wrapper for SMDM TransEncoder."""

    def __init__(self, model, mask_token_id=32000):
        self.model = model
        self.lm_head = model.lm_head
        self.config = SimpleNamespace(mask_token_id=mask_token_id)

    def eval(self):
        self.model.eval()
        return self

    def __call__(self, input_ids=None, attention_mask=None, **kwargs):
        return SimpleNamespace(logits=self.model(input_ids))


def _patch_smdm_source(smdm_root):
    """Patch SMDM source to remove hard dependencies on flash_attn and xformers."""
    import re

    # --- Patch lit_gpt/model.py: replace flash_attn import + usage ---
    model_py = os.path.join(smdm_root, "lit_gpt", "model.py")
    if os.path.exists(model_py):
        with open(model_py, "r") as f:
            src = f.read()
        if "flash_attn" in src:
            # Remove flash_attn import
            src = src.replace(
                "from flash_attn import flash_attn_func",
                "# flash_attn patched out — using PyTorch SDPA fallback",
            )
            # Replace flash_attn_func(...) calls with F.scaled_dot_product_attention
            # flash_attn_func(q, k, v, dropout_p=..., causal=...) → SDPA equivalent
            src = re.sub(
                r'flash_attn_func\s*\(([^)]*)\)',
                r'F.scaled_dot_product_attention(\1)',
                src,
            )
            # Ensure torch.nn.functional is imported
            if "import torch.nn.functional as F" not in src:
                src = "import torch.nn.functional as F\n" + src
            # Patch xformers SwiGLU import in model.py
            src = src.replace(
                "from xformers.ops import SwiGLU",
                "# xformers patched out\n"
                "class SwiGLU(torch.nn.Module):\n"
                "    def __init__(self, in_features, hidden_features, out_features, bias=True):\n"
                "        super().__init__()\n"
                "        self.w1 = torch.nn.Linear(in_features, hidden_features, bias=bias)\n"
                "        self.w2 = torch.nn.Linear(in_features, hidden_features, bias=bias)\n"
                "        self.w3 = torch.nn.Linear(hidden_features, out_features, bias=bias)\n"
                "        self.act = torch.nn.SiLU()\n"
                "    def forward(self, x):\n"
                "        return self.w3(self.act(self.w1(x)) * self.w2(x))\n",
            )
            with open(model_py, "w") as f:
                f.write(src)
            print("  Patched model.py (flash_attn + xformers → PyTorch native)")

    # --- Patch lit_gpt/diffmodel.py: remove ALL non-standard deps ---
    diff_py = os.path.join(smdm_root, "lit_gpt", "diffmodel.py")
    if os.path.exists(diff_py):
        with open(diff_py, "r") as f:
            src = f.read()
        changed = False

        # 1) flash_attn top-level import
        if "from flash_attn import flash_attn_func" in src:
            src = src.replace(
                "from flash_attn import flash_attn_func",
                "# flash_attn patched out — using PyTorch SDPA fallback",
            )
            changed = True

        # 2) xformers SwiGLU
        if "from xformers.ops import SwiGLU" in src:
            src = src.replace(
                "from xformers.ops import SwiGLU",
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
                "        return self.w3(self.act(self.w1(x)) * self.w2(x))\n",
            )
            changed = True

        # 3) fused_rotary_embedding → pure PyTorch apply_rotary_emb_func
        if "from .fused_rotary_embedding import apply_rotary_emb_func" in src:
            rotary_impl = (
                "# fused_rotary_embedding patched out — pure PyTorch fallback\n"
                "def apply_rotary_emb_func(x, cos, sin, interleaved=False, inplace=False):\n"
                "    '''Pure PyTorch rotary embedding. x: (B, T, nh, hs), cos/sin: (T, rotary_dim//2)'''\n"
                "    rot_dim = cos.shape[-1] * 2\n"
                "    x_rot = x[..., :rot_dim]\n"
                "    x_pass = x[..., rot_dim:]\n"
                "    x1 = x_rot[..., : rot_dim // 2]\n"
                "    x2 = x_rot[..., rot_dim // 2 :]\n"
                "    cos = cos[:x.shape[1]].unsqueeze(0).unsqueeze(2)  # (1, T, 1, rot_dim//2)\n"
                "    sin = sin[:x.shape[1]].unsqueeze(0).unsqueeze(2)\n"
                "    o1 = x1 * cos - x2 * sin\n"
                "    o2 = x2 * cos + x1 * sin\n"
                "    out_rot = torch.cat([o1, o2], dim=-1)\n"
                "    return torch.cat([out_rot, x_pass], dim=-1).to(x.dtype)\n"
            )
            src = src.replace(
                "from .fused_rotary_embedding import apply_rotary_emb_func",
                rotary_impl,
            )
            changed = True

        # 4) lightning_utilities → stub RequirementCache
        if "from lightning_utilities.core.imports import RequirementCache" in src:
            src = src.replace(
                "from lightning_utilities.core.imports import RequirementCache",
                "# lightning_utilities patched out\n"
                "class RequirementCache:\n"
                "    def __init__(self, *a, **kw): pass\n"
                "    def __bool__(self): return False\n",
            )
            changed = True

        # 5) Ensure torch.nn.functional import
        if changed and "import torch.nn.functional as F" not in src:
            src = "import torch.nn.functional as F\n" + src

        if changed:
            with open(diff_py, "w") as f:
                f.write(src)
            print("  Patched diffmodel.py (flash_attn/xformers/rotary/lightning → PyTorch native)")

    # --- Patch lit_gpt/config.py: rewrite to be fully self-contained ---
    # Original config.py imports:
    #   - lit_gpt.model  → circular import with model.py
    #   - lit_gpt.utils  → pulls in lightning.fabric → torchvision → crash
    # We only need Config dataclass + find_multiple, so rewrite entirely.
    config_py = os.path.join(smdm_root, "lit_gpt", "config.py")
    if os.path.exists(config_py):
        with open(config_py, "r") as f:
            src = f.read()
        if "# config-fully-patched" not in src:
            # Read the original to extract the configs dict and Config class body
            # Then write a self-contained version
            new_src = (
                "# config-fully-patched: self-contained, no lit_gpt.model or lit_gpt.utils\n"
                "from dataclasses import dataclass\n"
                "from typing import Any, Literal, Optional, Type\n"
                "\n"
                "import torch\n"
                "from typing_extensions import Self\n"
                "\n"
                "\n"
                "def find_multiple(n: int, k: int) -> int:\n"
                "    if n % k == 0:\n"
                "        return n\n"
                "    return n + k - (n % k)\n"
                "\n"
            )
            # Keep everything from the @dataclass line onwards, but strip the
            # lit_gpt.model / lit_gpt.utils imports that were already removed
            # above.  Also need to patch any `lit_gpt.model.XXX` references.
            marker = "@dataclass"
            idx = src.find(marker)
            if idx != -1:
                body = src[idx:]

                # --- Replace norm_class property entirely ---
                # Original imports from lit_gpt.rmsnorm which needs
                # dropout_layer_norm CUDA kernel.  Use torch.nn.RMSNorm
                # (available in PyTorch 2.4+) for all RMSNorm variants.
                import re
                body = re.sub(
                    r'@property\s+def norm_class\(self\).*?(?=\n    @|\nconfigs)',
                    '@property\n'
                    '    def norm_class(self) -> Type:\n'
                    '        # Patched: use torch.nn.RMSNorm instead of lit_gpt.rmsnorm\n'
                    '        if "RMSNorm" in self._norm_class:\n'
                    '            if hasattr(torch.nn, "RMSNorm"):\n'
                    '                return torch.nn.RMSNorm\n'
                    '            # Fallback for PyTorch < 2.4\n'
                    '            class _RMSNorm(torch.nn.Module):\n'
                    '                def __init__(self, d, eps=1e-5):\n'
                    '                    super().__init__()\n'
                    '                    self.eps = eps\n'
                    '                    self.weight = torch.nn.Parameter(torch.ones(d))\n'
                    '                def forward(self, x):\n'
                    '                    norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)\n'
                    '                    return x * norm * self.weight\n'
                    '            return _RMSNorm\n'
                    '        return getattr(torch.nn, self._norm_class)\n\n',
                    body,
                    flags=re.DOTALL,
                )

                # --- Replace mlp_class property ---
                # Original does getattr(lit_gpt.model, self._mlp_class) which
                # triggers importing model.py with all its deps.
                # Instead, import from diffmodel which we already patched.
                body = re.sub(
                    r'@property\s+def mlp_class\(self\).*?(?=\n    @|\nconfigs)',
                    '@property\n'
                    '    def mlp_class(self) -> Type:\n'
                    '        # Patched: resolve MLP class from diffmodel (already patched)\n'
                    '        import lit_gpt.diffmodel as _dm\n'
                    '        return getattr(_dm, self._mlp_class)\n\n',
                    body,
                    flags=re.DOTALL,
                )

                # Also patch any remaining lit_gpt.model references
                body = body.replace(
                    "lit_gpt.model.",
                    "__import__('lit_gpt.model', fromlist=['model']).model.",
                )
                new_src += body
            else:
                # Fallback: keep original but strip the problematic imports
                new_src += src.replace("import lit_gpt.model\n", "")
                new_src = new_src.replace(
                    "from lit_gpt.utils import find_multiple\n", ""
                )

            with open(config_py, "w") as f:
                f.write(new_src)
            print("  Patched config.py (self-contained, no lightning/torchvision deps)")

    # --- Rewrite lit_gpt/__init__.py to avoid circular import chain ---
    # Original __init__.py imports model→config→model (circular).
    # We only need diffmodel, so replace with a minimal stub.
    init_py = os.path.join(smdm_root, "lit_gpt", "__init__.py")
    if os.path.exists(init_py):
        with open(init_py, "r") as f:
            src = f.read()
        if "# patched-init" not in src:
            with open(init_py, "w") as f:
                f.write(
                    "# patched-init: minimal stub to avoid circular imports\n"
                    "# Only diffmodel.Config and diffmodel.TransEncoder are needed.\n"
                )
            print("  Patched __init__.py (replaced with minimal stub)")


def load_smdm_model(cfg, device):
    """Load SMDM-1.1B from nieshen/SMDM safetensors + ML-GSAI/SMDM code."""
    import subprocess
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    smdm_root = cfg["smdm_root"]
    if not os.path.isdir(os.path.join(smdm_root, "lit_gpt")):
        print(f"Cloning ML-GSAI/SMDM → {smdm_root}")
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/ML-GSAI/SMDM.git", smdm_root],
            check=True,
        )

    # Patch source to remove flash_attn / xformers hard dependencies
    _patch_smdm_source(smdm_root)

    if smdm_root not in sys.path:
        sys.path.insert(0, smdm_root)

    # Clear any stale/partially-initialized lit_gpt modules from prior attempts
    stale = [k for k in sys.modules if k == "lit_gpt" or k.startswith("lit_gpt.")]
    for k in stale:
        del sys.modules[k]

    from lit_gpt.diffmodel import Config, TransEncoder

    ckpt_path = hf_hub_download(
        repo_id=cfg["hf_repo"],
        filename=cfg["hf_checkpoint"],
    )
    config = Config.from_name(cfg["smdm_config_name"])
    model = TransEncoder(config).to(device)
    model.load_state_dict(load_file(ckpt_path, device=device))
    model.half()
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["tokenizer_repo"], padding_side="right", use_fast=True
    )
    # FIX: pad_token_id MUST be different from mask_token_id!
    # Using EOS (id=2) as pad token instead of mask (id=32000).
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    tokenizer.pad_token_id = 2  # EOS token; NOT mask_token_id=32000

    return SMDMModelWrapper(model, cfg["mask_token_id"]), tokenizer


def load_transformers_diffusion_model(cfg, device, quantize_bits=None):
    """Load a HuggingFace masked diffusion model (MDLM / LLaDA / Dream),
    with optional 4/8-bit bitsandbytes quantization for the 7-8B models.

    LLaDA → AutoModelForCausalLM, Dream → AutoModel; both need trust_remote_code.
    On a single T4, 4-bit puts an 8B model at ~6 GB. On Kaggle's T4 x2 you can
    instead run fp16 (device_map='auto' shards it across both 16 GB GPUs).
    """
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    import gc

    repo = cfg["hf_repo"]
    tok_repo = cfg.get("tokenizer_repo") or repo
    n_gpus = torch.cuda.device_count()

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
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=(quantize_bits == 4),
            load_in_8bit=(quantize_bits == 8),
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = "auto"   # shards across all visible GPUs
        # Disk-offload fallback so a load that doesn't fit can stage through disk
        # instead of OOM-ing. Also reserve ~2 GB GPU headroom for activations and
        # cap CPU so accelerate doesn't try to keep fp16 shadow weights in RAM.
        offload_dir = "/kaggle/working/_offload"
        os.makedirs(offload_dir, exist_ok=True)
        kwargs["offload_folder"] = offload_dir
        max_mem = {}
        for i in range(n_gpus):
            total = torch.cuda.get_device_properties(i).total_memory
            max_mem[i] = f"{int((total - 2 * 1024**3) / 1024**3)}GiB"
        max_mem["cpu"] = "8GiB"
        kwargs["max_memory"] = max_mem
    else:
        kwargs["device_map"] = {"": device} if n_gpus <= 1 else "auto"

    # LLaDA/Dream custom modeling can miss `_tied_weights_keys` on newer
    # transformers; patch defensively before loading.
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

    # Class preference: explicit cfg["model_class"], else try all.
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

    # Surface whether 4-bit actually engaged. If this prints False on a single
    # T4, the custom modeling silently loaded fp16 (~16 GB) and that is the OOM
    # — switch to GPU T4 x2, or report this back.
    if quantize_bits in (4, 8):
        try:
            is_4bit = any(type(m).__name__ in ("Linear4bit", "Linear8bitLt")
                          for m in model.modules())
            print(f"  Quantization active ({quantize_bits}-bit): {is_4bit}")
        except Exception:
            pass
    for i in range(torch.cuda.device_count()):
        a = torch.cuda.memory_allocated(i) / 1024**3
        t = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  GPU {i}: {a:.2f} / {t:.1f} GB allocated")
    return model, tokenizer


print(f"\nLoading model: {MODEL_NAME}...")
start_time = time.time()

if MODEL_CFG["loader"] == "smdm":
    model, tokenizer = load_smdm_model(MODEL_CFG, DEVICE)
else:
    model, tokenizer = load_transformers_diffusion_model(MODEL_CFG, DEVICE, QUANTIZE_BITS)
load_time = time.time() - start_time
print(f"Model loaded in {load_time:.1f}s")
print(f"GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")


def _resolve_input_device(model):
    """Where to send inputs. For device_map='auto' (quantized / multi-GPU)
    models, use the device of the first sharded module; else the model device.
    Falls back to DEVICE for the SMDM wrapper (no .parameters())."""
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

# Determine mask token ID
MASK_TOKEN_ID = MODEL_CFG.get("mask_token_id")
if MASK_TOKEN_ID is None and hasattr(tokenizer, 'mask_token_id') and tokenizer.mask_token_id is not None:
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

# Some diffusion models (Dream) pass the raw int attention_mask straight into
# scaled_dot_product_attention, which rejects it ("attn_mask.dtype: long int").
# We score batch=1 with no padding, so the mask is all-ones (full attention) and
# dropping it is exactly equivalent. Probe once on the first forward, then skip.
_PASS_ATTN = True

@torch.no_grad()
def compute_mre(text, mask_ratio, num_draws=16, max_length=512):
    """
    Compute Masked Reconstruction Error for a single text.

    Returns the mean NLL of true tokens at masked positions,
    averaged over num_draws random mask samples.

    Notes vs v1:
      - No fixed-length padding (batch=1): score the passage at its true length.
        Padding to 512 with EOS made SMDM (no attention mask) attend over a wall
        of pad tokens, and wasted ~Nx compute on the 8B models.
      - log_softmax in float32 for numerical stability.
      - Inputs sent to INPUT_DEVICE so device_map='auto' (quantized) models work.
    """
    encoding = tokenizer(
        text, max_length=max_length, truncation=True, return_tensors="pt"
    )
    input_ids = encoding["input_ids"].to(INPUT_DEVICE)
    attention_mask = encoding["attention_mask"].to(INPUT_DEVICE)

    batch_size, seq_len = input_ids.shape
    draw_nlls = []

    for _ in range(num_draws):
        # Create random mask
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

        # Apply mask
        masked_input = input_ids.clone()
        masked_input[0, mask_positions] = MASK_TOKEN_ID

        # Forward pass. Probe attention_mask once; if the model rejects it
        # (Dream's SDPA dtype bug, or a signature mismatch), drop it permanently.
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

        # Some models (e.g. Dream loaded as AutoModel) return hidden states, not
        # logits — last dim is hidden_size, not vocab. Project through the LM head.
        vocab = getattr(getattr(model, "config", None), "vocab_size", None)
        if vocab is not None and logits.shape[-1] != vocab:
            head = getattr(model, "lm_head", None)
            if head is None and hasattr(model, "get_output_embeddings"):
                head = model.get_output_embeddings()
            if head is not None:
                logits = head(logits.to(next(head.parameters()).device))

        # Device-safe: with device_map='auto' the model shards across GPUs, so
        # logits can land on a different device than mask_positions (this is why
        # Dream on T4 x2 errored on every passage). Do all indexing on logits.device.
        dev = logits.device
        mp = mask_positions.to(dev)
        log_probs = F.log_softmax(logits[0, mp, :].float(), dim=-1)
        true_token_ids = input_ids[0, mask_positions].to(dev)
        token_nlls = -log_probs[torch.arange(len(mp), device=dev), true_token_ids]

        draw_nlls.append(token_nlls.mean().item())

    return float(np.mean(draw_nlls)) if draw_nlls else np.nan

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
    print(f"Errors: {len(errors)}/{len(df)}")
    print(f"  First error: {errors[0][1]}")
    if len(errors) == len(df):
        print("  ⚠ ALL passages errored → the forward/NLL path is failing, not the data."
              " The message above is the real cause.")

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

    # FIX: Auto-detect the correct score direction instead of assuming.
    # Try both directions, use whichever gives AUROC > 0.5.
    try:
        auroc_raw = roc_auc_score(labels[valid], scores[valid])
        auroc_neg = roc_auc_score(labels[valid], -scores[valid])
        if auroc_raw >= auroc_neg:
            auroc = auroc_raw
            best_sign = scores[valid]
            direction = "higher=machine"
        else:
            auroc = auroc_neg
            best_sign = -scores[valid]
            direction = "lower=machine"
    except ValueError:
        auroc = 0.5
        best_sign = scores[valid]
        direction = "unknown"

    # TPR at low FPR
    fpr, tpr, _ = roc_curve(labels[valid], best_sign)
    idx_1 = np.searchsorted(fpr, 0.01, side="right") - 1
    idx_5 = np.searchsorted(fpr, 0.05, side="right") - 1
    tpr_1 = tpr[max(0, idx_1)]
    tpr_5 = tpr[max(0, idx_5)]

    # Also print mean scores per class for sanity checking
    mean_human = np.nanmean(scores[labels == 0])
    mean_machine = np.nanmean(scores[labels == 1])
    print(f"  {score_col}: AUROC={auroc:.4f}  TPR@1%={tpr_1:.4f}  TPR@5%={tpr_5:.4f}  [{direction}]  mean(H)={mean_human:.3f} mean(M)={mean_machine:.3f}")

# Best score (auto-detect direction) — POOLED (reference only; floored by domain mixing)
def _best_auroc(col):
    v = df[col].values; m = ~np.isnan(v)
    if m.sum() < 10: return 0
    return max(roc_auc_score(labels[m], v[m]), roc_auc_score(labels[m], -v[m]))

best_col = max(score_cols, key=_best_auroc)
best_auroc_pooled = _best_auroc(best_col)
print(f"\nBest POOLED (reference): {best_col} = {best_auroc_pooled:.4f}")

# PRIMARY gate = mean WITHIN-TESTBED (per-domain) AUROC. Pooled AUROC mixes
# domains with different score scales and is known to floor to ~0.60 even when
# per-domain/per-generator separation is excellent — do NOT gate on it.
if "domain" not in df.columns or df["domain"].isna().all():
    df["domain"] = df["generator"].astype(str).str.split("_").str[0]

def _within_domain_auroc(col, min_per_class=20):
    aurocs = []
    s = df[col].values.astype(float)
    # fix orientation once on pooled data, then no per-domain flipping
    m = np.isfinite(s)
    if m.sum() < 10:               # all-NaN column (e.g. scoring failed) → skip
        return np.nan, 0
    flip = roc_auc_score(labels[m], s[m]) < 0.5
    s_or = -s if flip else s
    for dom, g in df.groupby("domain"):
        idx = g.index.values
        y = labels[idx]; sv = s_or[idx]; v = np.isfinite(sv)
        y, sv = y[v], sv[v]
        if (y == 0).sum() < min_per_class or (y == 1).sum() < min_per_class:
            continue
        try: aurocs.append(roc_auc_score(y, sv))
        except ValueError: pass
    return float(np.mean(aurocs)) if aurocs else np.nan, len(aurocs)

best_wt_col, best_wt, best_wt_ntb = None, -1, 0
for col in score_cols:
    a, ntb = _within_domain_auroc(col)
    if np.isfinite(a) and a > best_wt:
        best_wt, best_wt_col, best_wt_ntb = a, col, ntb

print(f"Best WITHIN-TESTBED (primary): {best_wt_col} = {best_wt:.4f} over {best_wt_ntb} domains")

if best_wt >= 0.85:
    print("🟢 GO: within-testbed AUROC ≥ 0.85. Core premise validated. Proceed to Phase 2.")
elif best_wt >= 0.70:
    print("🟡 MARGINAL: 0.70-0.85 within-testbed. Real signal; a larger diffusion")
    print("   scorer (llada-8b / dream-7b) or DC/DTD may push it over 0.85.")
else:
    print("🔴 NO-GO: < 0.70 even within-testbed. Core premise is weak. Consider pivoting.")

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
            auroc = max(roc_auc_score(gen_labels[valid], gen_scores[valid]),
                       roc_auc_score(gen_labels[valid], -gen_scores[valid]))
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
    # Auto-detect direction
    auroc_pos = roc_auc_score(labels[valid], scores[valid])
    auroc_neg = roc_auc_score(labels[valid], -scores[valid])
    if auroc_pos >= auroc_neg:
        fpr, tpr, _ = roc_curve(labels[valid], scores[valid])
        auroc = auroc_pos
    else:
        fpr, tpr, _ = roc_curve(labels[valid], -scores[valid])
        auroc = auroc_neg
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
