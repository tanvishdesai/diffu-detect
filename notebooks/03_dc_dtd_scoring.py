"""
==========================================================================
  DiffuDetect — Kaggle Notebook 3: DC + DTD Scoring (Phase 2)
==========================================================================

PURPOSE:
  - Run Diffusion Curvature (DC) scoring with SMDM-1.1B
  - Run Denoising-Trajectory Dynamics (DTD) with LLaDA-8B (4-bit)
  - Build combined feature table
  - Compare per-statistic AUROCs → M2 milestone

KAGGLE SETTINGS:
  - GPU: T4 x1 (required)
  - Internet: ON
  - Accelerator: GPU T4
  - Persistence: Save outputs

NOTE: DC is SLOW (50 perturbations × 16 draws per passage).
      Use MAX_SAMPLES=500 for initial validation, scale up later.
"""

# Uncomment this line when running on Kaggle:
# !pip install -q torch transformers datasets accelerate "bitsandbytes>=0.46.1" \
#     scikit-learn pandas pyarrow tqdm huggingface_hub sentencepiece protobuf psutil

import os, sys, time, json, gc
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

# ─── Config ──────────────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)

# ─── GPU diagnostics ─────────────────────────────────────────────────────────
NUM_GPUS = torch.cuda.device_count()
print(f"Number of GPUs: {NUM_GPUS}")
for i in range(NUM_GPUS):
    name = torch.cuda.get_device_name(i)
    mem = torch.cuda.get_device_properties(i).total_memory / 1024**3  # FIX: total_memory (was total_mem → AttributeError crash)
    print(f"  GPU {i}: {name} ({mem:.1f} GB)")
try:
    import psutil
    print(f"System RAM: {psutil.virtual_memory().total / 1024**3:.1f} GB "
          f"(available: {psutil.virtual_memory().available / 1024**3:.1f} GB)")
except ImportError:
    print("(psutil not installed — skipping RAM diagnostics)")

# Choose which to run (DC is slow, DTD needs iterative model)
RUN_DC = False          # DC gave AUROC=0.506 (random); skip unless investigating
RUN_DTD = True

# DC config — SMDM weights are at nieshen/SMDM (not nieshen/SMDM-1.1b)
DC_MODEL_CFG = {
    "loader": "smdm",
    "hf_repo": "nieshen/SMDM",
    "hf_checkpoint": "mdm_safetensors/mdm-1028M-1600e18.safetensors",
    "smdm_config_name": "Diff_LLaMA_1028M",
    "tokenizer_repo": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    "mask_token_id": 32000,
    "smdm_root": "/tmp/SMDM",
}
DC_MODEL_NAME = "smdm-1.1b"
DC_MASK_RATIO = 0.30
DC_NUM_PERTURBATIONS = 30       # Reduced from 50 for speed
DC_NUM_MASK_DRAWS = 8           # Reduced from 16 for speed

# DTD config — LLaDA primary, Dream-7B as fallback
DTD_MODEL_CANDIDATES = [
    ("GSAI-ML/LLaDA-8B-Instruct", "llada-8b"),
    ("Dream-org/Dream-v0-Instruct-7B", "dream-7b"),
]
# Verified mask-token ids (also present in each model.config). We prefer these
# constants so a tokenizer that lacks an explicit mask_token can't silently
# fall back to unk/last-vocab (which would make every DTD feature meaningless).
KNOWN_MASK_IDS = {
    "GSAI-ML/LLaDA-8B-Instruct": 126336,
    "Dream-org/Dream-v0-Instruct-7B": 151666,
}
DTD_QUANTIZE = 4
DTD_NUM_STEPS = 16              # Reduced from 32 to save memory (halves forward passes)
DTD_NUM_DRAWS = 2               # Reduced from 4 to save memory
DTD_INITIAL_MASK_RATIO = 0.90

MAX_SAMPLES = 300               # Small — saves RAM for model loading
MAX_LENGTH = 256                # Reduced from 512 — halves activation memory per forward pass

DATA_DIR = "/kaggle/input/diffudetect-data/data"
RESULTS_DIR = "/kaggle/working/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─── Load data ───────────────────────────────────────────────────────────────

data_file = os.path.join(DATA_DIR, "mage_quick.parquet")
if not os.path.exists(data_file):
    # Use streaming to avoid loading the entire MAGE dataset into RAM
    from datasets import load_dataset
    print("Loading MAGE via streaming (memory-safe)...")
    ds = load_dataset("yaful/MAGE", streaming=True)
    split = "test" if "test" in ds else list(ds.keys())[0]
    rows = []
    for i, example in enumerate(ds[split]):
        if len(rows) >= MAX_SAMPLES * 3:  # Grab enough to subsample
            break
        label_flipped = 1 - int(example["label"])  # MAGE labels are inverted
        src = example.get("src", "unknown")
        rows.append({
            "id": f"mage_{i}",
            "text": example["text"],
            "label": label_flipped,
            "generator": src,
            "domain": src.split("_")[0] if src else "unknown",
            "dataset": "mage",
            "attack": "none",
        })
    df = pd.DataFrame(rows)
    del rows; gc.collect()  # Free the list immediately
    print(f"Streamed {len(df)} rows")
else:
    df = pd.read_parquet(data_file)

if MAX_SAMPLES and len(df) > MAX_SAMPLES:
    df = df.groupby("label", group_keys=False).apply(
        lambda x: x.sample(n=min(MAX_SAMPLES//2, len(x)), random_state=SEED)
    ).reset_index(drop=True)

print(f"Data: {len(df)} passages, labels={df['label'].value_counts().to_dict()}")

# Extract texts as a lightweight list and drop the heavy text column from the DataFrame
# This frees significant RAM before model loading
_texts = df["text"].tolist()
df = df.drop(columns=["text"])
gc.collect()
print(f"Extracted {len(_texts)} texts, freed DataFrame text column")

# ─── Helper functions ────────────────────────────────────────────────────────

def _patch_smdm_source(smdm_root):
    """Patch SMDM source to remove hard dependencies on flash_attn, xformers, etc."""
    import re

    # --- Patch lit_gpt/model.py: replace flash_attn import + usage ---
    model_py = os.path.join(smdm_root, "lit_gpt", "model.py")
    if os.path.exists(model_py):
        with open(model_py, "r") as f:
            src = f.read()
        if "flash_attn" in src:
            src = src.replace(
                "from flash_attn import flash_attn_func",
                "# flash_attn patched out — using PyTorch SDPA fallback",
            )
            src = re.sub(
                r'flash_attn_func\s*\(([^)]*)\)',
                r'F.scaled_dot_product_attention(\1)',
                src,
            )
            if "import torch.nn.functional as F" not in src:
                src = "import torch.nn.functional as F\n" + src
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

        if "from flash_attn import flash_attn_func" in src:
            src = src.replace(
                "from flash_attn import flash_attn_func",
                "# flash_attn patched out — using PyTorch SDPA fallback",
            )
            changed = True

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

        if "from lightning_utilities.core.imports import RequirementCache" in src:
            src = src.replace(
                "from lightning_utilities.core.imports import RequirementCache",
                "# lightning_utilities patched out\n"
                "class RequirementCache:\n"
                "    def __init__(self, *a, **kw): pass\n"
                "    def __bool__(self): return False\n",
            )
            changed = True

        if changed and "import torch.nn.functional as F" not in src:
            src = "import torch.nn.functional as F\n" + src

        if changed:
            with open(diff_py, "w") as f:
                f.write(src)
            print("  Patched diffmodel.py (flash_attn/xformers/rotary/lightning → PyTorch native)")

    # --- Patch lit_gpt/config.py: rewrite to be fully self-contained ---
    config_py = os.path.join(smdm_root, "lit_gpt", "config.py")
    if os.path.exists(config_py):
        with open(config_py, "r") as f:
            src = f.read()
        if "# config-fully-patched" not in src:
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
            marker = "@dataclass"
            idx = src.find(marker)
            if idx != -1:
                body = src[idx:]
                body = re.sub(
                    r'@property\s+def norm_class\(self\).*?(?=\n    @|\nconfigs)',
                    '@property\n'
                    '    def norm_class(self) -> Type:\n'
                    '        if "RMSNorm" in self._norm_class:\n'
                    '            if hasattr(torch.nn, "RMSNorm"):\n'
                    '                return torch.nn.RMSNorm\n'
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
                body = re.sub(
                    r'@property\s+def mlp_class\(self\).*?(?=\n    @|\nconfigs)',
                    '@property\n'
                    '    def mlp_class(self) -> Type:\n'
                    '        import lit_gpt.diffmodel as _dm\n'
                    '        return getattr(_dm, self._mlp_class)\n\n',
                    body,
                    flags=re.DOTALL,
                )
                body = body.replace(
                    "lit_gpt.model.",
                    "__import__('lit_gpt.model', fromlist=['model']).model.",
                )
                new_src += body
            else:
                new_src += src.replace("import lit_gpt.model\n", "")
                new_src = new_src.replace("from lit_gpt.utils import find_multiple\n", "")

            with open(config_py, "w") as f:
                f.write(new_src)
            print("  Patched config.py (self-contained, no lightning/torchvision deps)")

    # --- Rewrite lit_gpt/__init__.py ---
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


def _load_smdm_model(cfg):
    """Load SMDM-1.1B from nieshen/SMDM safetensors."""
    import subprocess
    from types import SimpleNamespace
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    class _Wrap:
        def __init__(self, model, mask_token_id):
            self.model = model
            self.lm_head = model.lm_head
            self.config = SimpleNamespace(mask_token_id=mask_token_id)
        def eval(self):
            self.model.eval()
            return self
        def __call__(self, input_ids=None, attention_mask=None, **kwargs):
            return SimpleNamespace(logits=self.model(input_ids))

    smdm_root = cfg["smdm_root"]
    if not os.path.isdir(os.path.join(smdm_root, "lit_gpt")):
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/ML-GSAI/SMDM.git", smdm_root],
            check=True,
        )

    # Patch source to remove flash_attn / xformers / rotary hard dependencies
    _patch_smdm_source(smdm_root)

    if smdm_root not in sys.path:
        sys.path.insert(0, smdm_root)

    # Clear any stale/partially-initialized lit_gpt modules from prior attempts
    stale = [k for k in sys.modules if k == "lit_gpt" or k.startswith("lit_gpt.")]
    for k in stale:
        del sys.modules[k]

    from lit_gpt.diffmodel import Config, TransEncoder

    ckpt_path = hf_hub_download(repo_id=cfg["hf_repo"], filename=cfg["hf_checkpoint"])
    config = Config.from_name(cfg["smdm_config_name"])
    model = TransEncoder(config).to(DEVICE)
    model.load_state_dict(load_file(ckpt_path, device=DEVICE))
    model.half()
    model.eval()

    tok = AutoTokenizer.from_pretrained(cfg["tokenizer_repo"], padding_side="right", use_fast=True)
    tok.add_special_tokens({"pad_token": "<|pad|>"})
    # FIX: pad_token_id must NOT equal mask_token_id!
    tok.pad_token_id = 2  # EOS token; NOT mask_token_id=32000
    return _Wrap(model, cfg["mask_token_id"]), tok


def _patch_llada_model_class(repo):
    """Monkey-patch LLaDA/Dream model class to fix transformers compatibility.

    Newer transformers versions expect `_tied_weights_keys` (list) and
    `all_tied_weights_keys` (property) on every PreTrainedModel subclass.
    LLaDA's custom code inherits from an older base and is missing these.
    """
    import importlib
    try:
        # Import the remote modeling module so the custom class is registered
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(repo, trust_remote_code=True)
        # The auto-downloaded module is cached; find the model class
        model_cls_name = getattr(cfg, "architectures", [None])
        if model_cls_name:
            model_cls_name = model_cls_name[0]
            # Find the class in loaded modules
            for mod_name, mod in list(sys.modules.items()):
                if mod and hasattr(mod, model_cls_name):
                    cls = getattr(mod, model_cls_name)
                    if not hasattr(cls, '_tied_weights_keys'):
                        cls._tied_weights_keys = []
                        print(f"  Patched {model_cls_name}._tied_weights_keys")
                    break
    except Exception as e:
        print(f"  Warning: could not pre-patch model class: {e}")


def load_diffusion_model(repo_or_cfg, quantize_bits=None):
    """Load a diffusion model with optional quantization."""
    if isinstance(repo_or_cfg, dict) and repo_or_cfg.get("loader") == "smdm":
        return _load_smdm_model(repo_or_cfg)

    repo = repo_or_cfg
    from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM, AutoModelForCausalLM, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    # Pre-patch model class for LLaDA/Dream compatibility
    _patch_llada_model_class(repo)

    # Free CPU RAM before loading large model
    gc.collect()
    torch.cuda.empty_cache()

    kwargs = {
        "pretrained_model_name_or_path": repo,
        "trust_remote_code": True,
        "torch_dtype": torch.float16,
        "low_cpu_mem_usage": True,  # Don't double-allocate weights in CPU RAM
    }
    if quantize_bits in (4, 8):
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=(quantize_bits==4), load_in_8bit=(quantize_bits==8),
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        # Spread across ALL available GPUs
        if NUM_GPUS > 1:
            # Give each GPU a fair share of VRAM (leave 2GB headroom per GPU for activations)
            max_mem = {}
            for i in range(NUM_GPUS):
                total = torch.cuda.get_device_properties(i).total_memory  # FIX: total_memory
                max_mem[i] = f"{int((total - 2 * 1024**3) / 1024**3)}GiB"
            max_mem["cpu"] = "6GiB"  # Kaggle has ~13GB RAM; cap CPU to leave room for OS/Python
            kwargs["device_map"] = "auto"
            kwargs["max_memory"] = max_mem
            print(f"  Multi-GPU loading: max_memory={max_mem}")
        else:
            kwargs["device_map"] = "auto"
        # Use disk offloading — stages weights through disk to avoid RAM spikes
        offload_dir = "/kaggle/working/_offload"
        os.makedirs(offload_dir, exist_ok=True)
        kwargs["offload_folder"] = offload_dir
    else:
        kwargs["device_map"] = {"": DEVICE}

    mdl = None
    # Try AutoModelForCausalLM FIRST — LLaDA registers as causal LM
    for Cls in [AutoModelForCausalLM, AutoModelForMaskedLM, AutoModel]:
        try:
            mdl = Cls.from_pretrained(**kwargs)
            print(f"Loaded with {Cls.__name__}")
            break
        except Exception as e:
            print(f"  {Cls.__name__} failed: {e}")
            # If it's the tied_weights issue, try patching more aggressively
            if "tied_weights" in str(e) or "all_tied_weights_keys" in str(e):
                try:
                    # Patch ALL PreTrainedModel subclasses in loaded modules
                    for mod_name, mod in list(sys.modules.items()):
                        if mod:
                            for attr_name in dir(mod):
                                try:
                                    obj = getattr(mod, attr_name)
                                    if (isinstance(obj, type) and
                                        hasattr(obj, 'from_pretrained') and
                                        not hasattr(obj, '_tied_weights_keys')):
                                        obj._tied_weights_keys = []
                                except: pass
                    mdl = Cls.from_pretrained(**kwargs)
                    print(f"Loaded with {Cls.__name__} (after aggressive patch)")
                    break
                except Exception as e2:
                    print(f"  {Cls.__name__} failed again after patch: {e2}")
            # Reclaim RAM from the failed attempt
            gc.collect()
            torch.cuda.empty_cache()
            continue
    if mdl is None: raise RuntimeError(f"Cannot load {repo}")
    mdl.eval()

    # Print memory usage after loading
    for i in range(NUM_GPUS):
        alloc = torch.cuda.memory_allocated(i) / 1024**3
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3  # FIX: total_memory (was total_mem → AttributeError crash after load)
        print(f"  GPU {i}: {alloc:.2f} / {total:.1f} GB used")

    return mdl, tok

def get_mask_token_id(model, tokenizer):
    for check in [
        lambda: tokenizer.mask_token_id if hasattr(tokenizer, 'mask_token_id') and tokenizer.mask_token_id else None,
        lambda: model.config.mask_token_id if hasattr(model.config, 'mask_token_id') else None,
        lambda: tokenizer.convert_tokens_to_ids("[MASK]") if "[MASK]" in tokenizer.get_vocab() else None,
        lambda: tokenizer.convert_tokens_to_ids("<mask>") if "<mask>" in tokenizer.get_vocab() else None,
        lambda: tokenizer.unk_token_id,
    ]:
        v = check()
        if v is not None: return v
    return len(tokenizer) - 1

def get_special_ids(tokenizer):
    ids = set()
    for attr in ['bos_token_id','eos_token_id','pad_token_id','cls_token_id','sep_token_id']:
        v = getattr(tokenizer, attr, None)
        if v is not None: ids.add(v)
    return ids

def make_mask(input_ids, mask_ratio, special_ids, pad_id):
    seq_len = input_ids.shape[1]
    eligible = torch.ones(seq_len, dtype=torch.bool, device=input_ids.device)
    if pad_id is not None: eligible &= (input_ids[0] != pad_id)
    for sid in special_ids: eligible &= (input_ids[0] != sid)
    eidx = eligible.nonzero(as_tuple=True)[0]
    n = max(1, int(len(eidx) * mask_ratio))
    perm = torch.randperm(len(eidx), device=input_ids.device)[:n]
    return eidx[perm]

def get_model_input_device(model):
    """Resolve the device where inputs should be sent for a multi-GPU model."""
    # For device_map='auto' models, find the first parameter's device
    try:
        if hasattr(model, 'hf_device_map'):
            # Model is split across devices; send inputs to the device of the first module
            first_device = next(iter(model.hf_device_map.values()))
            if isinstance(first_device, int):
                return torch.device(f"cuda:{first_device}")
            elif isinstance(first_device, str) and first_device != "cpu":
                return torch.device(first_device)
    except StopIteration:
        pass
    # Fallback: find first parameter's device
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device(DEVICE)


def forward_get_logits(model, input_ids, attention_mask=None):
    try: out = model(input_ids=input_ids, attention_mask=attention_mask)
    except TypeError: out = model(input_ids=input_ids)
    if hasattr(out, 'logits'): return out.logits
    if isinstance(out, tuple): return out[0]
    return out

# =========================================================================
#  PART A: DIFFUSION CURVATURE (DC)
# =========================================================================

if RUN_DC:
    print("\n" + "=" * 60)
    print("DIFFUSION CURVATURE (DC) SCORING")
    print("=" * 60)

    model, tokenizer = load_diffusion_model(DC_MODEL_CFG)
    mask_id = get_mask_token_id(model, tokenizer)
    special_ids = get_special_ids(tokenizer)
    print(f"Mask token: {mask_id}, special: {special_ids}")

    @torch.no_grad()
    def compute_mre_for_ids(ids, attn, mask_ratio, n_draws):
        nlls = []
        for _ in range(n_draws):
            mpos = make_mask(ids, mask_ratio, special_ids, tokenizer.pad_token_id)
            m_ids = ids.clone(); m_ids[0, mpos] = mask_id
            logits = forward_get_logits(model, m_ids, attn)
            lp = F.log_softmax(logits, dim=-1)
            nll = -lp[0, mpos, :][torch.arange(len(mpos)), ids[0, mpos]].mean().item()
            nlls.append(nll)
        return np.mean(nlls)

    @torch.no_grad()
    def generate_perturbation(ids, attn, mask_ratio):
        mpos = make_mask(ids, mask_ratio, special_ids, tokenizer.pad_token_id)
        m_ids = ids.clone(); m_ids[0, mpos] = mask_id
        logits = forward_get_logits(model, m_ids, attn)
        probs = F.softmax(logits, dim=-1)
        perturbed = ids.clone()
        for p in mpos:
            perturbed[0, p] = torch.multinomial(probs[0, p], 1).item()
        return perturbed

    @torch.no_grad()
    def compute_dc(text):
        enc = tokenizer(text, max_length=MAX_LENGTH, truncation=True, padding="max_length", return_tensors="pt")
        ids = enc["input_ids"].to(DEVICE); attn = enc["attention_mask"].to(DEVICE)

        orig_mre = compute_mre_for_ids(ids, attn, DC_MASK_RATIO, DC_NUM_MASK_DRAWS)
        p_mres = []
        for _ in range(DC_NUM_PERTURBATIONS):
            p_ids = generate_perturbation(ids, attn, DC_MASK_RATIO)
            p_mre = compute_mre_for_ids(p_ids, attn, DC_MASK_RATIO, max(2, DC_NUM_MASK_DRAWS//4))
            p_mres.append(p_mre)

        p_mean = np.mean(p_mres); p_std = np.std(p_mres) + 1e-8
        return {
            "dc_curvature": orig_mre - p_mean,
            "dc_original_mre": orig_mre,
            "dc_perturb_mean_mre": p_mean,
            "dc_normalized": (orig_mre - p_mean) / p_std,
        }

    dc_results = []
    for idx in tqdm(range(len(df)), desc="DC scoring"):
        try:
            dc_results.append(compute_dc(str(_texts[idx])))
        except Exception as e:
            dc_results.append({"dc_curvature": np.nan, "dc_original_mre": np.nan,
                             "dc_perturb_mean_mre": np.nan, "dc_normalized": np.nan})

    dc_df = pd.DataFrame(dc_results)
    for c in dc_df.columns: df[c] = dc_df[c].values

    # Save DC scores (re-attach text column for parquet output)
    meta = ["id","label","generator","domain","dataset","attack"]
    save_df = df.copy()
    save_df["text"] = _texts
    save_cols = [c for c in (["id","text","label","generator","domain","dataset","attack"]) if c in save_df.columns] + list(dc_df.columns)
    save_df[save_cols].to_parquet(os.path.join(RESULTS_DIR, f"scores_mage_dc_{DC_MODEL_NAME}.parquet"), index=False)
    del save_df; gc.collect()

    # DC AUROC
    valid = ~np.isnan(df["dc_normalized"].values)
    if valid.sum() > 10:
        # FIX: auto-detect score direction
        auroc_pos = roc_auc_score(df["label"].values[valid], df["dc_normalized"].values[valid])
        auroc_neg = roc_auc_score(df["label"].values[valid], -df["dc_normalized"].values[valid])
        auroc = max(auroc_pos, auroc_neg)
        direction = "higher=machine" if auroc_pos >= auroc_neg else "lower=machine"
        print(f"\nDC Normalized AUROC: {auroc:.4f} [{direction}]")

    del model; torch.cuda.empty_cache()

# =========================================================================
#  PART B: DENOISING-TRAJECTORY DYNAMICS (DTD)
# =========================================================================

if RUN_DTD:
    print("\n" + "=" * 60)
    print("DENOISING-TRAJECTORY DYNAMICS (DTD) SCORING")
    print("=" * 60)

    # Try each candidate model until one loads successfully
    model = None
    tokenizer = None
    DTD_MODEL_NAME = None
    _loaded_repo = None
    for _repo, _name in DTD_MODEL_CANDIDATES:
        try:
            print(f"\nTrying DTD model: {_repo}...")
            model, tokenizer = load_diffusion_model(_repo, DTD_QUANTIZE)
            DTD_MODEL_NAME = _name
            _loaded_repo = _repo
            print(f"Successfully loaded {_name}")
            break
        except Exception as e:
            print(f"Failed to load {_name}: {e}")
            # Aggressively reclaim RAM from the failed load attempt
            # (partial model weights + tokenizer can leak several GB)
            gc.collect()
            torch.cuda.empty_cache()
            continue

    if model is None:
        print("\n" + "!" * 60)
        print("ERROR: Could not load ANY DTD model. Skipping DTD scoring.")
        print("!" * 60)
        RUN_DTD = False
    else:
        # Prefer the verified constant; fall back to config/tokenizer lookup.
        mask_id = KNOWN_MASK_IDS.get(_loaded_repo) or get_mask_token_id(model, tokenizer)
        special_ids = get_special_ids(tokenizer)
        print(f"Mask token: {mask_id}")

if RUN_DTD:  # Second guard: only runs if model loaded successfully above
    # Resolve input device once (for multi-GPU models)
    _dtd_input_device = get_model_input_device(model)
    print(f"DTD input device: {_dtd_input_device}")

    @torch.no_grad()
    def compute_dtd(text):
        # No fixed-length padding: batch=1, so score at true length. On an 8B
        # model this is the difference between forwarding ~120 real tokens vs a
        # padded 256 every denoising step.
        enc = tokenizer(text, max_length=MAX_LENGTH, truncation=True, return_tensors="pt")
        ids = enc["input_ids"].to(_dtd_input_device)
        attn = enc["attention_mask"].to(_dtd_input_device)
        seq_len = ids.shape[1]

        all_features = []
        for _ in range(DTD_NUM_DRAWS):
            # Create initial heavy mask
            mpos = make_mask(ids, DTD_INITIAL_MASK_RATIO, special_ids, tokenizer.pad_token_id)
            cur = ids.clone(); cur[0, mpos] = mask_id
            masked_set = set(mpos.cpu().tolist())
            orig_masked = set(masked_set)

            step_entropies = []
            commit_times = {p: DTD_NUM_STEPS for p in orig_masked}
            committed = set()
            prev_preds = {}
            flips = {p: 0 for p in orig_masked}

            schedule = np.linspace(DTD_INITIAL_MASK_RATIO, 0.0, DTD_NUM_STEPS + 1)[:-1]

            for step_idx, target_ratio in enumerate(schedule):
                if not masked_set: break
                logits = forward_get_logits(model, cur, attn)
                # Move logits to CPU-side to free GPU VRAM for next forward pass
                # Only need softmax at masked positions + top-1 for the full seq
                logits_cpu = logits.float().cpu()
                del logits  # Free GPU memory immediately

                probs = F.softmax(logits_cpu, dim=-1)
                entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)
                del logits_cpu  # Free after computing probs/entropy

                # Step entropy over masked positions
                mpos_list = list(masked_set)
                if mpos_list:
                    step_ent = entropy[0, mpos_list].mean().item()
                    step_entropies.append(step_ent)

                top1_probs, top1_ids = probs[0].max(dim=-1)

                # Track flips and commits
                for p in list(masked_set):
                    pred = top1_ids[p].item()
                    if step_idx > 0 and p in prev_preds and pred != prev_preds[p]:
                        flips[p] += 1
                    prev_preds[p] = pred
                    if top1_probs[p].item() > 0.9 and p not in committed:
                        commit_times[p] = step_idx
                        committed.add(p)

                # Unmask most confident tokens
                n_target = max(0, int(target_ratio * len(orig_masked)))
                n_unmask = max(0, len(masked_set) - n_target)
                if n_unmask > 0 and mpos_list:
                    ents = [(entropy[0, p].item(), p) for p in mpos_list]
                    ents.sort()
                    for _, p in ents[:n_unmask]:
                        cur[0, p] = top1_ids[p].to(_dtd_input_device)
                        masked_set.discard(p)

                del probs, entropy, top1_probs, top1_ids  # Free memory

            # Extract features
            feat = {}
            if len(step_entropies) >= 2:
                feat["dtd_entropy_auc"] = float(np.trapz(step_entropies))
            else:
                feat["dtd_entropy_auc"] = step_entropies[0] if step_entropies else 0.0

            ct_vals = [commit_times[p] / max(len(schedule), 1) for p in orig_masked]
            feat["dtd_mean_commit_time"] = float(np.mean(ct_vals)) if ct_vals else 1.0

            if len(step_entropies) >= 3:
                feat["dtd_trajectory_curvature"] = float(np.mean(np.abs(np.diff(step_entropies, n=2))))
            else:
                feat["dtd_trajectory_curvature"] = 0.0

            flip_vals = [flips[p] for p in orig_masked]
            feat["dtd_mean_flips"] = float(np.mean(flip_vals)) if flip_vals else 0.0
            feat["dtd_final_entropy"] = step_entropies[-1] if step_entropies else 0.0
            feat["dtd_entropy_drop"] = (step_entropies[0] - step_entropies[-1]) if len(step_entropies) >= 2 else 0.0

            all_features.append(feat)

        # Average over draws
        avg = {}
        for k in all_features[0]:
            avg[k] = float(np.mean([f[k] for f in all_features]))
        return avg

    dtd_results = []
    for idx in tqdm(range(len(df)), desc="DTD scoring"):
        try:
            dtd_results.append(compute_dtd(str(_texts[idx])))
        except Exception as e:
            dtd_results.append({k: np.nan for k in ["dtd_entropy_auc","dtd_mean_commit_time",
                               "dtd_trajectory_curvature","dtd_mean_flips","dtd_final_entropy","dtd_entropy_drop"]})
        # Periodic GPU memory cleanup
        if (idx + 1) % 10 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    dtd_df = pd.DataFrame(dtd_results)
    for c in dtd_df.columns: df[c] = dtd_df[c].values

    # Save DTD scores (re-attach text column for parquet output)
    save_df = df.copy()
    save_df["text"] = _texts
    meta_all = ["id","text","label","generator","domain","dataset","attack"]
    save_cols = [c for c in meta_all if c in save_df.columns] + list(dtd_df.columns)
    dtd_out_name = DTD_MODEL_NAME or "unknown"
    save_df[save_cols].to_parquet(os.path.join(RESULTS_DIR, f"scores_mage_dtd_{dtd_out_name}.parquet"), index=False)
    del save_df; gc.collect()

    # DTD AUROCs
    for col in dtd_df.columns:
        valid = ~np.isnan(df[col].values)
        if valid.sum() < 10: continue
        try:
            # FIX: auto-detect score direction
            auroc = max(roc_auc_score(df["label"].values[valid], df[col].values[valid]),
                       roc_auc_score(df["label"].values[valid], -df[col].values[valid]))
            print(f"  {col}: AUROC={auroc:.4f}")
        except: pass

    del model; torch.cuda.empty_cache()

# ─── Summary ─────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("NOTEBOOK 3 COMPLETE — DC + DTD Scoring Done")
print("=" * 60)
print(f"Results saved to: {RESULTS_DIR}/")
print("Next: Run Notebook 4 (baselines) then Notebook 5 (evaluation)")
