"""
DiffuDetect — Fast-DetectGPT Baseline (analytic)

Re-implementation of Fast-DetectGPT (Bao et al., 2024) using the ANALYTIC
sampling discrepancy, which is the whole point of "Fast"-DetectGPT.

v1 used a Monte-Carlo approximation: for every passage it sampled ~100 perturbed
sequences and ran a full forward pass on each. That was ~40-100x too slow
(13.96 s/passage) AND high-variance, and it floored to near-chance AUROC. The
conditional curvature has a closed form, so we compute it directly.

White-box discrepancy:

    d(x) = ( Σ_t logp(x_t | x_<t) − Σ_t μ_t ) / sqrt( Σ_t σ²_t )

with, at each position t (sampling model == scoring model):

    μ_t  = Σ_v p(v) logp(v)            (expected conditional log-prob)
    σ²_t = Σ_v p(v) logp(v)²  − μ_t²   (its variance)

One forward pass per passage. Higher discrepancy ⇒ more machine-generated.

A reference (black-box) model can be supplied for the sampling distribution; the
scoring model still provides logp. This matches the official two-model variant.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Any, Dict, List, Optional
from tqdm import tqdm

from ..config import get_device
from ..utils import tokenize_single


class FastDetectGPTScorer:
    """Fast-DetectGPT baseline scorer (analytic closed form)."""

    def __init__(
        self,
        scoring_model: Any,
        scoring_tokenizer: Any,
        source_model: Any = None,      # reference/sampling model; None ⇒ white-box
        source_tokenizer: Any = None,
        num_perturbations: int = 0,    # kept for API compatibility; unused (analytic)
        device: Optional[str] = None,
    ):
        self.scoring_model = scoring_model
        self.scoring_tokenizer = scoring_tokenizer
        self.source_model = source_model or scoring_model
        self.source_tokenizer = source_tokenizer or scoring_tokenizer
        self.device = device or get_device()

    @torch.no_grad()
    def score_text(
        self,
        text: str,
        num_perturbations: Optional[int] = None,   # ignored (analytic)
        max_length: int = 512,
    ) -> Dict[str, float]:
        """
        Score a single text using the analytic Fast-DetectGPT discrepancy.

        Returns:
          - fdgpt_curvature: the analytic sampling discrepancy (main score)
          - fdgpt_original_ll: mean conditional log-likelihood of the passage
        """
        # No fixed-length padding: score the passage at its true length.
        enc = tokenize_single(text, self.scoring_tokenizer, max_length, self.device,
                              pad_to_max=False)
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        if input_ids.shape[1] < 2:
            return {"fdgpt_curvature": float("nan"), "fdgpt_original_ll": float("nan")}

        labels = input_ids[:, 1:]                                    # (1, T)

        # Scoring-model log-probs (float32 for numerical stability).
        score_logits = self.scoring_model(
            input_ids=input_ids, attention_mask=attention_mask
        ).logits[:, :-1, :].float()
        lprobs_score = F.log_softmax(score_logits, dim=-1)
        ll = lprobs_score.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)  # (1,T)

        # Reference (sampling) distribution. White-box ⇒ same model/logits.
        if self.source_model is self.scoring_model:
            probs_ref = lprobs_score.exp()
        else:
            ref_logits = self.source_model(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits[:, :-1, :].float()
            probs_ref = F.softmax(ref_logits, dim=-1)

        mean_ref = (probs_ref * lprobs_score).sum(dim=-1)                       # (1,T)
        var_ref = (probs_ref * lprobs_score.square()).sum(dim=-1) - mean_ref.square()
        discrepancy = (
            (ll.sum(dim=-1) - mean_ref.sum(dim=-1))
            / var_ref.sum(dim=-1).clamp_min(1e-8).sqrt()
        )

        return {
            "fdgpt_curvature": discrepancy.mean().item(),
            "fdgpt_original_ll": ll.mean().item(),
        }

    def score_batch(
        self,
        texts: List[str],
        num_perturbations: Optional[int] = None,
        max_length: int = 512,
        show_progress: bool = True,
    ) -> List[Dict[str, float]]:
        """Score a batch of texts."""
        results = []
        iterator = tqdm(texts, desc="Fast-DetectGPT scoring", disable=not show_progress)
        for text in iterator:
            scores = self.score_text(text, num_perturbations, max_length)
            results.append(scores)
            if show_progress:
                iterator.set_postfix(curv=f"{scores['fdgpt_curvature']:.4f}")
        return results

    def score_dataset(
        self,
        df,
        text_col: str = "text",
        num_perturbations: Optional[int] = None,
        max_length: int = 512,
    ) -> List[Dict[str, float]]:
        """Score all texts in a DataFrame."""
        texts = df[text_col].tolist()
        return self.score_batch(texts, num_perturbations, max_length)
