"""
DiffuDetect — Binoculars Baseline

Re-implementation of Binoculars (Hans et al., 2024).

Uses two LLMs as an observer/performer pair:
  1. Observer model computes cross-entropy of the text.
  2. Performer model (typically the instruction-tuned version) does the same.
  3. Binoculars score = cross-entropy(observer) / cross-entropy(performer)

Intuition: Human text has similar perplexity under both models.
Machine text is much easier for the performer (closer to the generation
distribution) than for the observer → lower ratio.

Default pair: Falcon-7B (performer) + Falcon-7B-Instruct (observer).
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Any, Dict, List, Optional
from tqdm import tqdm

from ..config import get_device
from ..utils import tokenize_single


class BinocularsScorer:
    """
    Binoculars baseline scorer.

    Uses an observer/performer model pair to compute a cross-entropy ratio.
    """

    def __init__(
        self,
        observer_model: Any,
        observer_tokenizer: Any,
        performer_model: Any,
        performer_tokenizer: Any,
        device: Optional[str] = None,
    ):
        self.observer_model = observer_model
        self.observer_tokenizer = observer_tokenizer
        self.performer_model = performer_model
        self.performer_tokenizer = performer_tokenizer
        self.device = device or get_device()

        # Use the performer's tokenizer for shared tokenization
        # (both models in a pair share the same tokenizer)
        self.tokenizer = performer_tokenizer

    @torch.no_grad()
    def _compute_cross_entropy(
        self,
        model: Any,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> float:
        """
        Compute the mean per-token cross-entropy of the text under a model.
        """
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = outputs.logits  # (batch, seq_len, vocab)

        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :]     # (batch, seq_len-1, vocab)
        shift_labels = input_ids[:, 1:]       # (batch, seq_len-1)
        mask = attention_mask[:, 1:].float()  # (batch, seq_len-1)

        # Cross-entropy per token
        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_ce = -log_probs.gather(
            dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)  # (batch, seq_len-1)

        # Mean over non-padding tokens
        mean_ce = (token_ce * mask).sum() / mask.sum()
        return mean_ce.item()

    @torch.no_grad()
    def _compute_perplexity(
        self,
        model: Any,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> float:
        """Compute perplexity = exp(mean cross-entropy)."""
        ce = self._compute_cross_entropy(model, input_ids, attention_mask)
        return np.exp(ce)

    def score_text(
        self,
        text: str,
        max_length: int = 512,
    ) -> Dict[str, float]:
        """
        Score a single text using Binoculars.

        Returns:
          - bino_score: CE(observer) / CE(performer) — the main score
          - bino_observer_ce: cross-entropy under observer
          - bino_performer_ce: cross-entropy under performer
          - bino_observer_ppl: perplexity under observer
          - bino_performer_ppl: perplexity under performer
          - bino_ce_diff: CE(observer) - CE(performer)
        """
        # Tokenize using shared tokenizer
        encoding = tokenize_single(text, self.tokenizer, max_length, self.device)
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        # Compute cross-entropies
        observer_ce = self._compute_cross_entropy(
            self.observer_model, input_ids, attention_mask
        )
        performer_ce = self._compute_cross_entropy(
            self.performer_model, input_ids, attention_mask
        )

        # Binoculars score
        bino_score = observer_ce / (performer_ce + 1e-8)

        return {
            "bino_score": bino_score,
            "bino_observer_ce": observer_ce,
            "bino_performer_ce": performer_ce,
            "bino_observer_ppl": np.exp(observer_ce),
            "bino_performer_ppl": np.exp(performer_ce),
            "bino_ce_diff": observer_ce - performer_ce,
        }

    def score_batch(
        self,
        texts: List[str],
        max_length: int = 512,
        show_progress: bool = True,
    ) -> List[Dict[str, float]]:
        """Score a batch of texts."""
        results = []
        iterator = tqdm(texts, desc="Binoculars scoring", disable=not show_progress)

        for text in iterator:
            scores = self.score_text(text, max_length)
            results.append(scores)
            if show_progress:
                iterator.set_postfix(bino=f"{scores['bino_score']:.4f}")

        return results

    def score_dataset(
        self,
        df,
        text_col: str = "text",
        max_length: int = 512,
    ) -> List[Dict[str, float]]:
        """Score all texts in a DataFrame."""
        texts = df[text_col].tolist()
        return self.score_batch(texts, max_length)
