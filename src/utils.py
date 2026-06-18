"""
DiffuDetect — Shared Utilities

Model loading (with quantization), tokenization helpers, seeding, Parquet I/O,
and general-purpose functions shared across scorers and baselines.
"""

import os
import random
import time
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


# ─── Reproducibility ─────────────────────────────────────────────────────────

def seed_everything(seed: int = 42):
    """Set seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ─── Model Loading ───────────────────────────────────────────────────────────

def load_model_and_tokenizer(
    hf_repo: str,
    quantize_bits: Optional[int] = None,
    device: str = "cuda",
    cache_dir: Optional[str] = None,
    trust_remote_code: bool = True,
    max_length: int = 512,
) -> Tuple[Any, Any]:
    """
    Load a HuggingFace model + tokenizer with optional 4/8-bit quantization.

    Returns (model, tokenizer).
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    print(f"[utils] Loading model: {hf_repo} (bits={quantize_bits})")
    start = time.time()

    tokenizer = AutoTokenizer.from_pretrained(
        hf_repo,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {
        "pretrained_model_name_or_path": hf_repo,
        "cache_dir": cache_dir,
        "trust_remote_code": trust_remote_code,
        "torch_dtype": torch.float16,
    }

    if quantize_bits in (4, 8):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=(quantize_bits == 4),
            load_in_8bit=(quantize_bits == 8),
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        load_kwargs["quantization_config"] = bnb_config
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["device_map"] = {"": device}

    model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
    model.eval()

    elapsed = time.time() - start
    print(f"[utils] Model loaded in {elapsed:.1f}s")
    return model, tokenizer


def load_masked_diffusion_model(
    hf_repo: str,
    quantize_bits: Optional[int] = None,
    device: str = "cuda",
    cache_dir: Optional[str] = None,
    trust_remote_code: bool = True,
) -> Tuple[Any, Any]:
    """
    Load a masked diffusion language model.

    Handles SMDM, MDLM, LLaDA, Dream architectures with trust_remote_code.
    These models typically use AutoModelForMaskedLM or custom architectures.
    """
    from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig

    print(f"[utils] Loading diffusion model: {hf_repo} (bits={quantize_bits})")
    start = time.time()

    tokenizer = AutoTokenizer.from_pretrained(
        hf_repo,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {
        "pretrained_model_name_or_path": hf_repo,
        "cache_dir": cache_dir,
        "trust_remote_code": trust_remote_code,
        "torch_dtype": torch.float16,
    }

    if quantize_bits in (4, 8):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=(quantize_bits == 4),
            load_in_8bit=(quantize_bits == 8),
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        load_kwargs["quantization_config"] = bnb_config
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["device_map"] = {"": device}

    # Try AutoModelForMaskedLM first, fall back to AutoModel
    try:
        from transformers import AutoModelForMaskedLM
        model = AutoModelForMaskedLM.from_pretrained(**load_kwargs)
    except Exception:
        model = AutoModel.from_pretrained(**load_kwargs)

    model.eval()
    elapsed = time.time() - start
    print(f"[utils] Diffusion model loaded in {elapsed:.1f}s")
    return model, tokenizer


# ─── Tokenization ────────────────────────────────────────────────────────────

def tokenize_texts(
    texts: List[str],
    tokenizer: Any,
    max_length: int = 512,
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    """Tokenize a list of texts with padding & truncation."""
    encodings = tokenizer(
        texts,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    return {k: v.to(device) for k, v in encodings.items()}


def tokenize_single(
    text: str,
    tokenizer: Any,
    max_length: int = 512,
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    """Tokenize a single text."""
    return tokenize_texts([text], tokenizer, max_length, device)


# ─── Masking ──────────────────────────────────────────────────────────────────

def create_random_mask(
    input_ids: torch.Tensor,
    mask_ratio: float,
    special_token_ids: Optional[set] = None,
    pad_token_id: Optional[int] = None,
) -> torch.Tensor:
    """
    Create a random boolean mask (True = masked) for a batch of token IDs.

    Does not mask special tokens (BOS, EOS, PAD, etc.).

    Args:
        input_ids: shape (batch, seq_len)
        mask_ratio: fraction of non-special tokens to mask
        special_token_ids: set of token IDs to never mask
        pad_token_id: pad token ID (also never masked)

    Returns:
        Boolean tensor of shape (batch, seq_len), True = position is masked.
    """
    batch_size, seq_len = input_ids.shape
    mask = torch.zeros_like(input_ids, dtype=torch.bool)

    for i in range(batch_size):
        # Find positions eligible for masking (non-special, non-pad)
        eligible = torch.ones(seq_len, dtype=torch.bool)

        if pad_token_id is not None:
            eligible &= (input_ids[i] != pad_token_id)

        if special_token_ids:
            for sid in special_token_ids:
                eligible &= (input_ids[i] != sid)

        eligible_indices = eligible.nonzero(as_tuple=True)[0]
        n_eligible = len(eligible_indices)
        n_mask = max(1, int(n_eligible * mask_ratio))

        perm = torch.randperm(n_eligible, device=input_ids.device)[:n_mask]
        mask[i, eligible_indices[perm]] = True

    return mask


# ─── Results I/O ─────────────────────────────────────────────────────────────

def save_scores_parquet(
    records: List[Dict[str, Any]],
    output_path: str,
):
    """Save a list of score records to Parquet."""
    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_parquet(output_path, index=False, engine="pyarrow")
    print(f"[utils] Saved {len(df)} records → {output_path}")


def load_scores_parquet(path: str) -> pd.DataFrame:
    """Load scores from a single Parquet file."""
    return pd.read_parquet(path, engine="pyarrow")


def load_all_scores(results_dir: str) -> pd.DataFrame:
    """Load and concatenate all Parquet score files from a directory."""
    import glob
    files = glob.glob(os.path.join(results_dir, "scores_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No score files found in {results_dir}")
    dfs = [pd.read_parquet(f, engine="pyarrow") for f in files]
    combined = pd.concat(dfs, ignore_index=True)
    print(f"[utils] Loaded {len(combined)} total records from {len(files)} files")
    return combined


# ─── Passage Processing ──────────────────────────────────────────────────────

def truncate_text(text: str, max_chars: int = 200) -> str:
    """Truncate text for storage/display."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def chunk_list(lst: list, chunk_size: int) -> list:
    """Split a list into chunks of chunk_size."""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


# ─── Logging / Progress ──────────────────────────────────────────────────────

class Timer:
    """Simple context-manager timer."""
    def __init__(self, name: str = ""):
        self.name = name
        self.elapsed = 0.0

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start
        if self.name:
            print(f"[Timer] {self.name}: {self.elapsed:.2f}s")


def get_gpu_memory_mb() -> float:
    """Return current GPU memory usage in MB (0 if no GPU)."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0.0
