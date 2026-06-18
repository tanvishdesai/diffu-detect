"""
DiffuDetect — Fast-DetectGPT Baseline

Re-implementation of Fast-DetectGPT (Bao et al., 2024).

White-box setting: Uses the same model for both scoring and perturbation.
Black-box setting: Uses a separate source model for perturbation.

Core idea: Sample-then-perturb conditional probability curvature.
  1. For each position i, compute log p(x_i | x_{<i}) under the scoring model.
  2. Sample perturbations by replacing x_i with samples from p(· | x_{<i}).
  3. Curvature = (log p(x) - E[log p(x̃)]) / std(log p(x̃))
     where x̃ are perturbations.

Higher curvature → more likely machine-generated.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Any, Dict, List, Optional
from tqdm import tqdm

from ..config import get_device
from ..utils import tokenize_single


class FastDetectGPTScorer:
    """
    Fast-DetectGPT baseline scorer.

    Uses AR models to compute conditional probability curvature.
    The primary baseline to beat on paraphrase robustness.
    """

    def __init__(
        self,
        scoring_model: Any,
        scoring_tokenizer: Any,
        source_model: Any = None,      # if None, white-box (same as scoring)
        source_tokenizer: Any = None,
        num_perturbations: int = 100,
        device: Optional[str] = None,
    ):
        self.scoring_model = scoring_model
        self.scoring_tokenizer = scoring_tokenizer
        self.source_model = source_model or scoring_model
        self.source_tokenizer = source_tokenizer or scoring_tokenizer
        self.num_perturbations = num_perturbations
        self.device = device or get_device()

    @torch.no_grad()
    def _get_log_probs(
        self,
        model: Any,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get per-token conditional log probabilities from an AR model.

        Returns log p(x_i | x_{<i}) for each position i.
        Shape: (batch, seq_len)
        """
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = outputs.logits  # (batch, seq_len, vocab)

        # Shift: logits at position i predict token at position i+1
        shift_logits = logits[:, :-1, :]  # (batch, seq_len-1, vocab)
        shift_labels = input_ids[:, 1:]    # (batch, seq_len-1)

        log_probs = F.log_softmax(shift_logits, dim=-1)  # (batch, seq_len-1, vocab)

        # Gather log-prob of actual next tokens
        token_log_probs = log_probs.gather(
            dim=-1,
            index=shift_labels.unsqueeze(-1),
        ).squeeze(-1)  # (batch, seq_len-1)

        return token_log_probs

    @torch.no_grad()
    def _get_sampling_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get the sampling distribution from the source model.

        Returns probabilities at each position for perturbation sampling.
        Shape: (batch, seq_len, vocab)
        """
        outputs = self.source_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = outputs.logits
        probs = F.softmax(logits[:, :-1, :], dim=-1)  # (batch, seq_len-1, vocab)
        return probs

    @torch.no_grad()
    def _sample_perturbation(
        self,
        input_ids: torch.Tensor,
        sampling_probs: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Create a perturbed version by sampling from the source model's
        conditional distribution at each position.
        """
        batch_size, seq_len = input_ids.shape
        perturbed = input_ids.clone()

        # Sample from the conditional distribution at each position
        # sampling_probs is (batch, seq_len-1, vocab)
        for pos in range(sampling_probs.shape[1]):
            if attention_mask[0, pos + 1] == 0:  # skip padding
                continue
            sampled = torch.multinomial(sampling_probs[:, pos, :], 1)  # (batch, 1)
            perturbed[:, pos + 1] = sampled.squeeze(-1)

        return perturbed

    def score_text(
        self,
        text: str,
        num_perturbations: Optional[int] = None,
        max_length: int = 512,
    ) -> Dict[str, float]:
        """
        Score a single text using Fast-DetectGPT.

        Returns:
          - fdgpt_curvature: the conditional probability curvature (main score)
          - fdgpt_original_ll: log-likelihood of original text
          - fdgpt_perturb_mean_ll: mean log-likelihood of perturbations
          - fdgpt_perturb_std_ll: std of perturbation log-likelihoods
        """
        num_perturbations = num_perturbations or self.num_perturbations

        # Tokenize
        encoding = tokenize_single(text, self.scoring_tokenizer, max_length, self.device)
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        # 1. Original log-probs under scoring model
        original_lp = self._get_log_probs(
            self.scoring_model, input_ids, attention_mask
        )
        # Mask out padding
        mask = attention_mask[:, 1:].float()  # aligned with shifted log-probs
        original_mean_ll = (original_lp * mask).sum() / mask.sum()

        # 2. Get sampling distribution from source model
        sampling_probs = self._get_sampling_probs(input_ids, attention_mask)

        # 3. Generate perturbations and score them
        perturb_lls = []
        for _ in range(num_perturbations):
            perturbed = self._sample_perturbation(
                input_ids, sampling_probs, attention_mask
            )
            p_lp = self._get_log_probs(
                self.scoring_model, perturbed, attention_mask
            )
            p_mean_ll = (p_lp * mask).sum() / mask.sum()
            perturb_lls.append(p_mean_ll.item())

        perturb_mean = np.mean(perturb_lls)
        perturb_std = np.std(perturb_lls) + 1e-8

        # 4. Curvature
        curvature = (original_mean_ll.item() - perturb_mean) / perturb_std

        return {
            "fdgpt_curvature": curvature,
            "fdgpt_original_ll": original_mean_ll.item(),
            "fdgpt_perturb_mean_ll": perturb_mean,
            "fdgpt_perturb_std_ll": perturb_std,
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
