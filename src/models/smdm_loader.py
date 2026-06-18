"""
Load SMDM-1.1B from nieshen/SMDM (raw safetensors + ML-GSAI/SMDM code).

The HuggingFace repo nieshen/SMDM hosts checkpoint files, not a transformers
model card. Weights must be loaded with the official TransEncoder architecture.
"""

from __future__ import annotations

import os
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, Optional, Tuple


SMDM_DEFAULTS = {
    "hf_repo": "nieshen/SMDM",
    "hf_checkpoint": "mdm_safetensors/mdm-1028M-1600e18.safetensors",
    "config_name": "Diff_LLaMA_1028M",
    "tokenizer_repo": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    "mask_token_id": 32000,
}


class SMDMModelWrapper:
    """Minimal HuggingFace-like wrapper around SMDM TransEncoder."""

    def __init__(self, model: Any, mask_token_id: int = 32000):
        self.model = model
        self.lm_head = model.lm_head
        self.config = SimpleNamespace(mask_token_id=mask_token_id)

    def eval(self):
        self.model.eval()
        return self

    def to(self, device):
        self.model.to(device)
        return self

    def __call__(self, input_ids=None, attention_mask=None, **kwargs):
        logits = self.model(input_ids)
        return SimpleNamespace(logits=logits)


def ensure_smdm_code(smdm_root: Optional[str] = None) -> str:
    """Clone ML-GSAI/SMDM if needed and add it to sys.path."""
    smdm_root = smdm_root or os.environ.get("SMDM_ROOT", "/tmp/SMDM")
    lit_gpt_dir = os.path.join(smdm_root, "lit_gpt")

    if not os.path.isdir(lit_gpt_dir):
        print(f"[smdm] Cloning ML-GSAI/SMDM → {smdm_root}")
        os.makedirs(os.path.dirname(smdm_root) or ".", exist_ok=True)
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "https://github.com/ML-GSAI/SMDM.git",
                smdm_root,
            ],
            check=True,
        )

    if smdm_root not in sys.path:
        sys.path.insert(0, smdm_root)

    return smdm_root


def load_smdm_model(
    hf_repo: str = SMDM_DEFAULTS["hf_repo"],
    hf_checkpoint: str = SMDM_DEFAULTS["hf_checkpoint"],
    config_name: str = SMDM_DEFAULTS["config_name"],
    tokenizer_repo: str = SMDM_DEFAULTS["tokenizer_repo"],
    mask_token_id: int = SMDM_DEFAULTS["mask_token_id"],
    device: str = "cuda",
    cache_dir: Optional[str] = None,
    smdm_root: Optional[str] = None,
    dtype=None,
) -> Tuple[SMDMModelWrapper, Any, int]:
    """
    Load SMDM-1.1B checkpoint and TinyLlama tokenizer.

    Returns (model_wrapper, tokenizer, mask_token_id).
    """
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    if dtype is None:
        dtype = torch.float16

    ensure_smdm_code(smdm_root)
    from lit_gpt.diffmodel import Config, TransEncoder

    print(f"[smdm] Downloading checkpoint {hf_repo}/{hf_checkpoint}")
    ckpt_path = hf_hub_download(
        repo_id=hf_repo,
        filename=hf_checkpoint,
        cache_dir=cache_dir,
    )

    print(f"[smdm] Building {config_name}")
    config = Config.from_name(config_name)
    model = TransEncoder(config).to(device)
    model.load_state_dict(load_file(ckpt_path, device=device))
    model.to(dtype=dtype)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_repo,
        cache_dir=cache_dir,
        padding_side="right",
        use_fast=True,
    )
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    tokenizer.pad_token_id = mask_token_id

    wrapper = SMDMModelWrapper(model, mask_token_id=mask_token_id)
    print("[smdm] Model loaded")
    return wrapper, tokenizer, mask_token_id
