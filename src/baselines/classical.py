"""
DiffuDetect — Classical Baselines

GLTR-style baselines using simple statistics from an AR language model:
  1. Log-likelihood: mean log p(x_i | x_{<i})
  2. Rank: mean rank of each token in the model's prediction
  3. Log-rank: mean log(rank) — more robust to outliers
  4. Entropy: mean entropy of the model's prediction at each position

These are cheap, fast baselines. Higher log-likelihood and lower entropy/rank
suggest machine-generated text.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Any, Dict, List, Optional
from tqdm import tqdm

from ..config import get_device
from ..utils import tokenize_single


class ClassicalScorer:
    """
    Classical (GLTR-style) baseline scorer.

    Computes log-likelihood, rank, log-rank, and entropy from an AR model.
    Fast and lightweight — no perturbations needed.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        device: Optional[str] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or get_device()

    @torch.no_grad()
    def score_text(
        self,
        text: str,
        max_length: int = 512,
    ) -> Dict[str, float]:
        """
        Score a single text with all classical statistics.

        Returns:
          - cls_log_likelihood: mean log p(x_i | x_{<i})
          - cls_mean_rank: mean rank of actual tokens
          - cls_mean_log_rank: mean log(rank)
          - cls_mean_entropy: mean prediction entropy
          - cls_perplexity: exp(-mean log likelihood)
        """
        encoding = tokenize_single(text, self.tokenizer, max_length, self.device)
        input_ids = encoding["input_ids"]      # (1, seq_len)
        attention_mask = encoding["attention_mask"]

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = outputs.logits  # (1, seq_len, vocab)

        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :]      # (1, seq_len-1, vocab)
        shift_labels = input_ids[:, 1:]        # (1, seq_len-1)
        mask = attention_mask[:, 1:]           # (1, seq_len-1)

        # Get the number of real (non-padding) tokens
        n_tokens = mask.sum().item()
        if n_tokens == 0:
            return self._empty_scores()

        # Probabilities
        probs = F.softmax(shift_logits, dim=-1)          # (1, seq_len-1, vocab)
        log_probs = F.log_softmax(shift_logits, dim=-1)  # (1, seq_len-1, vocab)

        # 1. Log-likelihood
        token_log_probs = log_probs.gather(
            dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)  # (1, seq_len-1)
        mean_ll = (token_log_probs * mask.float()).sum() / n_tokens

        # 2. Ranks
        # Sort probabilities in descending order, find rank of actual token
        sorted_probs, sorted_indices = probs.sort(dim=-1, descending=True)

        # Compute rank for each position
        ranks = torch.zeros_like(shift_labels, dtype=torch.float)
        for pos in range(shift_labels.shape[1]):
            if mask[0, pos] == 0:
                continue
            actual_token = shift_labels[0, pos].item()
            token_ranks = (sorted_indices[0, pos] == actual_token).nonzero(as_tuple=True)[0]
            if len(token_ranks) > 0:
                ranks[0, pos] = token_ranks[0].float() + 1  # 1-indexed
            else:
                ranks[0, pos] = probs.shape[-1]  # worst case

        mean_rank = (ranks * mask.float()).sum() / n_tokens
        mean_log_rank = ((ranks + 1).log() * mask.float()).sum() / n_tokens

        # 3. Entropy
        entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)  # (1, seq_len-1)
        mean_entropy = (entropy * mask.float()).sum() / n_tokens

        # 4. Perplexity
        perplexity = torch.exp(-mean_ll)

        return {
            "cls_log_likelihood": mean_ll.item(),
            "cls_mean_rank": mean_rank.item(),
            "cls_mean_log_rank": mean_log_rank.item(),
            "cls_mean_entropy": mean_entropy.item(),
            "cls_perplexity": perplexity.item(),
        }

    def _empty_scores(self) -> Dict[str, float]:
        """Return empty scores for degenerate cases."""
        return {
            "cls_log_likelihood": 0.0,
            "cls_mean_rank": 0.0,
            "cls_mean_log_rank": 0.0,
            "cls_mean_entropy": 0.0,
            "cls_perplexity": 0.0,
        }

    def score_batch(
        self,
        texts: List[str],
        max_length: int = 512,
        show_progress: bool = True,
    ) -> List[Dict[str, float]]:
        """Score a batch of texts."""
        results = []
        iterator = tqdm(texts, desc="Classical scoring", disable=not show_progress)

        for text in iterator:
            scores = self.score_text(text, max_length)
            results.append(scores)
            if show_progress:
                iterator.set_postfix(
                    ll=f"{scores['cls_log_likelihood']:.4f}",
                    ent=f"{scores['cls_mean_entropy']:.2f}"
                )

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
