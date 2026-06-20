"""
DiffuDetect — Global Configuration

Central config for all models, datasets, scoring hyperparameters, and paths.
All notebooks and scripts import from here so there is a single source of truth.
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path


# ─── Paths ────────────────────────────────────────────────────────────────────

# On Kaggle, results go to /kaggle/working then get saved.
# On local/Colab, point RESULTS_DIR to a Google Drive mount.
RESULTS_DIR = os.environ.get("DIFFUDETECT_RESULTS", "./results")
CACHE_DIR = os.environ.get("HF_HOME", None)  # HuggingFace cache


# ─── Model Registry ──────────────────────────────────────────────────────────

@dataclass
class ModelSpec:
    name: str
    hf_repo: str
    model_type: str          # "diffusion" | "ar" | "binoculars_obs" | "binoculars_perf" | "supervised"
    quantize_bits: Optional[int] = None   # None = full precision, 4 = 4-bit, 8 = 8-bit
    max_length: int = 512
    supports_iterative: bool = False      # True for LLaDA, Dream (DTD features)
    loader: str = "transformers"          # "transformers" | "smdm"
    hf_checkpoint: Optional[str] = None   # safetensors path inside hf_repo (SMDM)
    tokenizer_repo: Optional[str] = None
    smdm_config_name: Optional[str] = None
    mask_token_id: Optional[int] = None
    model_class: Optional[str] = None     # "causal_lm" | "automodel" | "masked_lm"


MODELS: Dict[str, ModelSpec] = {
    # ── Diffusion scorers ──
    "smdm-1.1b": ModelSpec(
        name="smdm-1.1b",
        hf_repo="nieshen/SMDM",
        model_type="diffusion",
        quantize_bits=None,
        max_length=512,
        supports_iterative=False,
        loader="smdm",
        hf_checkpoint="mdm_safetensors/mdm-1028M-1600e18.safetensors",
        tokenizer_repo="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
        smdm_config_name="Diff_LLaMA_1028M",
        mask_token_id=32000,
    ),
    "mdlm-110m": ModelSpec(
        name="mdlm-110m",
        hf_repo="kuleshov-group/mdlm-owt",
        model_type="diffusion",
        quantize_bits=None,
        max_length=512,
        supports_iterative=False,
    ),
    "llada-8b": ModelSpec(
        name="llada-8b",
        hf_repo="GSAI-ML/LLaDA-8B-Instruct",
        model_type="diffusion",
        quantize_bits=4,                 # ~6 GB on a single T4; fp16 on T4 x2
        max_length=512,
        supports_iterative=True,
        loader="transformers",
        tokenizer_repo="GSAI-ML/LLaDA-8B-Instruct",
        model_class="causal_lm",         # AutoModelForCausalLM (trust_remote_code)
        mask_token_id=126336,            # verified; also in model.config
    ),
    "dream-7b": ModelSpec(
        name="dream-7b",
        hf_repo="Dream-org/Dream-v0-Instruct-7B",
        model_type="diffusion",
        quantize_bits=4,
        max_length=512,
        supports_iterative=True,
        loader="transformers",
        tokenizer_repo="Dream-org/Dream-v0-Instruct-7B",
        model_class="automodel",         # AutoModel (trust_remote_code)
        mask_token_id=151666,            # verified; also in model.config
    ),
    # ── AR baselines ──
    "gpt-neo-2.7b": ModelSpec(
        name="gpt-neo-2.7b",
        hf_repo="EleutherAI/gpt-neo-2.7B",
        model_type="ar",
        quantize_bits=None,
        max_length=512,
    ),
    "gpt-j-6b": ModelSpec(
        name="gpt-j-6b",
        hf_repo="EleutherAI/gpt-j-6b",
        model_type="ar",
        quantize_bits=4,
        max_length=512,
    ),
    # ── Binoculars pair ──
    "falcon-7b": ModelSpec(
        name="falcon-7b",
        hf_repo="tiiuae/falcon-7b",
        model_type="binoculars_perf",
        quantize_bits=4,
        max_length=512,
    ),
    "falcon-7b-instruct": ModelSpec(
        name="falcon-7b-instruct",
        hf_repo="tiiuae/falcon-7b-instruct",
        model_type="binoculars_obs",
        quantize_bits=4,
        max_length=512,
    ),
}


# ─── Dataset Registry ────────────────────────────────────────────────────────

@dataclass
class DatasetSpec:
    name: str
    hf_repo: str
    subset: Optional[str] = None
    split: str = "test"
    text_col: str = "text"
    label_col: str = "label"
    max_samples: Optional[int] = None     # None = use all; set for speed during dev


DATASETS: Dict[str, DatasetSpec] = {
    "mage": DatasetSpec(
        name="mage",
        hf_repo="yaful/MAGE",
        text_col="text",
        label_col="label",
        max_samples=None,
    ),
    "raid": DatasetSpec(
        name="raid",
        hf_repo="liamdugan/raid",
        text_col="generation",
        label_col="model",        # will be binarized: "human" vs everything else
        max_samples=None,
    ),
    "m4gt": DatasetSpec(
        name="m4gt",
        hf_repo="SemEval2024/M4GT-Bench",
        subset="subtask2",
        text_col="text",
        label_col="label",
        max_samples=None,
    ),
}


# ─── Scoring Hyperparameters ─────────────────────────────────────────────────

@dataclass
class MREConfig:
    """Masked Reconstruction Error config."""
    mask_ratios: List[float] = field(default_factory=lambda: [0.15, 0.30, 0.50])
    num_mask_draws: int = 16          # K in the planning doc
    mask_token_id: Optional[int] = None  # set per-model at runtime


@dataclass
class DCConfig:
    """Diffusion Curvature config."""
    mask_ratio: float = 0.30
    num_perturbations: int = 50       # number of perturbation samples
    num_mask_draws: int = 16


@dataclass
class DTDConfig:
    """Denoising-Trajectory Dynamics config."""
    num_denoising_steps: int = 64     # iterative denoising schedule
    num_mask_draws: int = 8           # fewer draws since each is multi-step
    initial_mask_ratio: float = 0.90  # start heavily masked


@dataclass
class ScoringConfig:
    mre: MREConfig = field(default_factory=MREConfig)
    dc: DCConfig = field(default_factory=DCConfig)
    dtd: DTDConfig = field(default_factory=DTDConfig)
    batch_size: int = 4
    max_length: int = 512
    device: str = "cuda"
    seed: int = 42


# ─── Results Schema ──────────────────────────────────────────────────────────

RESULTS_COLUMNS = [
    "id",             # unique passage identifier
    "text",           # the passage text (first 200 chars for debugging)
    "label",          # 0=human, 1=machine
    "generator",      # which LLM generated (or "human")
    "domain",         # domain/source
    "dataset",        # which dataset
    "attack",         # paraphrase/synonym/none — for robustness analysis
    "method",         # scoring method name
    "score",          # the scalar detection score
    "model_used",     # which scorer model produced this score
]


def get_results_path(dataset: str, method: str, model: str) -> str:
    """Canonical path for a results parquet shard."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return os.path.join(RESULTS_DIR, f"scores_{dataset}_{method}_{model}.parquet")


def get_device():
    """Return the best available device."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
