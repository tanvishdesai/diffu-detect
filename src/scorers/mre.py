"""
DiffuDetect — Masked Reconstruction Error (MRE) Scorer

The core detection statistic:
  1. Take a passage x, tokenize it.
  2. Randomly mask a fraction r of tokens.
  3. Feed the masked input to a frozen diffusion LM.
  4. Compute the mean NLL of the TRUE tokens at masked positions.
  5. Average over K random mask draws for stability.

Hypothesis: AI text → lower MRE (easier to reconstruct).
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm

from ..config import MREConfig, get_device
from ..utils import (
    create_random_mask,
    tokenize_single,
    Timer,
)


class MREScorer:
    """
    Masked Reconstruction Error scorer.

    Given a frozen diffusion LM, scores passages by measuring how well
    the model reconstructs randomly masked tokens. Lower MRE = more
    predictable = more likely machine-generated.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: Optional[MREConfig] = None,
        device: Optional[str] = None,
        mask_token_id: Optional[int] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or MREConfig()
        self.device = device or get_device()

        # Determine mask token ID
        if mask_token_id is not None:
            self.mask_token_id = mask_token_id
        elif hasattr(tokenizer, 'mask_token_id') and tokenizer.mask_token_id is not None:
            self.mask_token_id = tokenizer.mask_token_id
        else:
            # For models without a dedicated mask token, use a sentinel
            # Many diffusion LMs use the last vocab token or a special token
            self.mask_token_id = self._find_mask_token_id()

        # Gather special token IDs to exclude from masking
        self.special_token_ids = set()
        for attr in ['bos_token_id', 'eos_token_id', 'pad_token_id', 'cls_token_id', 'sep_token_id']:
            tid = getattr(tokenizer, attr, None)
            if tid is not None:
                self.special_token_ids.add(tid)

        self.pad_token_id = tokenizer.pad_token_id

    def _find_mask_token_id(self) -> int:
        """Find or create a mask token ID for models without one."""
        # Check if [MASK] is in the vocabulary
        if "[MASK]" in self.tokenizer.get_vocab():
            return self.tokenizer.convert_tokens_to_ids("[MASK]")
        if "<mask>" in self.tokenizer.get_vocab():
            return self.tokenizer.convert_tokens_to_ids("<mask>")

        # For models that use a specific token for masking (e.g., SMDM)
        # Check model config
        if hasattr(self.model, 'config'):
            if hasattr(self.model.config, 'mask_token_id'):
                return self.model.config.mask_token_id

        # Fallback: use UNK token or the last token in vocab
        if self.tokenizer.unk_token_id is not None:
            return self.tokenizer.unk_token_id

        # Absolute fallback
        vocab_size = len(self.tokenizer)
        print(f"[MRE] WARNING: No mask token found. Using vocab_size-1 = {vocab_size - 1}")
        return vocab_size - 1

    @torch.no_grad()
    def _score_single_mask_draw(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        mask_ratio: float,
    ) -> float:
        """
        Score a single passage with one random mask draw.

        Args:
            input_ids: (1, seq_len) original token IDs
            attention_mask: (1, seq_len) attention mask
            mask_ratio: fraction of tokens to mask

        Returns:
            Mean NLL of true tokens at masked positions.
        """
        # Create random mask
        bool_mask = create_random_mask(
            input_ids,
            mask_ratio=mask_ratio,
            special_token_ids=self.special_token_ids,
            pad_token_id=self.pad_token_id,
        )

        # Store ground-truth tokens at masked positions
        true_tokens = input_ids.clone()

        # Apply mask: replace masked positions with mask token
        masked_input = input_ids.clone()
        masked_input[bool_mask] = self.mask_token_id

        # Forward pass through the diffusion model
        try:
            outputs = self.model(
                input_ids=masked_input,
                attention_mask=attention_mask,
            )
        except TypeError:
            # Some models don't accept attention_mask
            outputs = self.model(input_ids=masked_input)

        # Get logits — handle different output formats
        if hasattr(outputs, 'logits'):
            logits = outputs.logits  # (1, seq_len, vocab_size)
        elif hasattr(outputs, 'last_hidden_state'):
            # Some models return hidden states; need a head
            # Try to get logits from the model's lm_head
            if hasattr(self.model, 'lm_head'):
                logits = self.model.lm_head(outputs.last_hidden_state)
            elif hasattr(self.model, 'cls'):
                logits = self.model.cls(outputs.last_hidden_state)
            else:
                raise ValueError("Model output has no logits and no identifiable LM head")
        elif isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs

        # Compute NLL at masked positions only (float32 for stability)
        # logits: (1, seq_len, vocab_size)
        log_probs = F.log_softmax(logits.float(), dim=-1)  # (1, seq_len, vocab_size)

        # Gather log-probs of the true tokens
        true_log_probs = log_probs.gather(
            dim=-1,
            index=true_tokens.unsqueeze(-1),
        ).squeeze(-1)  # (1, seq_len)

        # NLL at masked positions only
        masked_nlls = -true_log_probs[bool_mask]  # (n_masked,)

        if len(masked_nlls) == 0:
            return float("nan")

        return masked_nlls.mean().item()

    def score_text(
        self,
        text: str,
        mask_ratios: Optional[List[float]] = None,
        num_draws: Optional[int] = None,
        max_length: int = 512,
    ) -> Dict[str, float]:
        """
        Score a single text passage.

        Returns a dict of MRE scores:
          - mre_r{ratio}: mean MRE at each mask ratio
          - mre_mean: average across all mask ratios (the recommended single score)
        """
        mask_ratios = mask_ratios or self.config.mask_ratios
        num_draws = num_draws or self.config.num_mask_draws

        # Tokenize (no fixed-length padding: batch=1, score at true length)
        encoding = tokenize_single(text, self.tokenizer, max_length, self.device,
                                   pad_to_max=False)
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        results = {}
        all_scores = []

        for ratio in mask_ratios:
            draw_scores = []
            for _ in range(num_draws):
                nll = self._score_single_mask_draw(input_ids, attention_mask, ratio)
                draw_scores.append(nll)

            mean_nll = np.mean(draw_scores)
            results[f"mre_r{ratio:.2f}"] = mean_nll
            all_scores.append(mean_nll)

        # The single summary score: mean across mask ratios
        results["mre_mean"] = np.mean(all_scores)

        return results

    def score_batch(
        self,
        texts: List[str],
        mask_ratios: Optional[List[float]] = None,
        num_draws: Optional[int] = None,
        max_length: int = 512,
        show_progress: bool = True,
    ) -> List[Dict[str, float]]:
        """
        Score a batch of text passages.

        Returns a list of score dicts (one per text).
        """
        mask_ratios = mask_ratios or self.config.mask_ratios
        num_draws = num_draws or self.config.num_mask_draws

        results = []
        iterator = tqdm(texts, desc="MRE scoring", disable=not show_progress)

        for text in iterator:
            scores = self.score_text(text, mask_ratios, num_draws, max_length)
            results.append(scores)

            # Update progress bar with latest score
            if show_progress:
                iterator.set_postfix(mre=f"{scores['mre_mean']:.4f}")

        return results

    def score_dataset(
        self,
        df,
        text_col: str = "text",
        mask_ratios: Optional[List[float]] = None,
        num_draws: Optional[int] = None,
        max_length: int = 512,
    ) -> List[Dict[str, float]]:
        """Score all texts in a DataFrame."""
        texts = df[text_col].tolist()
        return self.score_batch(texts, mask_ratios, num_draws, max_length)
