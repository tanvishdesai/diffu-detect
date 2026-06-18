"""
DiffuDetect — DetectGPT Baseline

Re-implementation of DetectGPT (Mitchell et al., 2023).

Unlike Fast-DetectGPT which uses model-based perturbation, DetectGPT
uses T5/mask-filling to perturb text and then computes curvature:
  1. Take passage x, compute log p(x) under a scoring LM.
  2. Perturb x by masking spans and filling them with T5 → x̃.
  3. Curvature = (log p(x) - log p(x̃)) / std(log p(x̃_i))

This is the weaker OOD baseline (slower, less effective than Fast-DetectGPT).
We include it for completeness.

Note: For Kaggle, we can use a small T5 model for perturbation.
"""

import torch
import torch.nn.functional as F
import numpy as np
import re
from typing import Any, Dict, List, Optional
from tqdm import tqdm

from ..config import get_device
from ..utils import tokenize_single


class DetectGPTScorer:
    """
    DetectGPT baseline scorer.

    Uses T5 mask-filling for perturbation + AR model for scoring.
    """

    def __init__(
        self,
        scoring_model: Any,
        scoring_tokenizer: Any,
        perturbation_model: Any = None,     # T5 model for mask-filling
        perturbation_tokenizer: Any = None,
        num_perturbations: int = 25,
        span_length: int = 2,               # average span length for masking
        mask_pct: float = 0.3,              # fraction of tokens to mask
        device: Optional[str] = None,
    ):
        self.scoring_model = scoring_model
        self.scoring_tokenizer = scoring_tokenizer
        self.perturbation_model = perturbation_model
        self.perturbation_tokenizer = perturbation_tokenizer
        self.num_perturbations = num_perturbations
        self.span_length = span_length
        self.mask_pct = mask_pct
        self.device = device or get_device()

    @torch.no_grad()
    def _compute_ll(
        self,
        text: str,
        max_length: int = 512,
    ) -> float:
        """Compute mean log-likelihood of text under the scoring model."""
        encoding = tokenize_single(text, self.scoring_tokenizer, max_length, self.device)
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        outputs = self.scoring_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = outputs.logits

        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        mask = attention_mask[:, 1:].float()

        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(
            dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        mean_ll = (token_log_probs * mask).sum() / mask.sum()
        return mean_ll.item()

    def _mask_and_fill_t5(self, text: str) -> str:
        """
        Perturb text using T5 mask-filling.

        If T5 is not loaded, fall back to word-level shuffling as a proxy.
        """
        if self.perturbation_model is not None and self.perturbation_tokenizer is not None:
            return self._t5_mask_fill(text)
        else:
            return self._word_shuffle_perturbation(text)

    def _t5_mask_fill(self, text: str) -> str:
        """Use T5 to fill masked spans."""
        words = text.split()
        n_masks = max(1, int(len(words) * self.mask_pct / self.span_length))

        # Create masked text with sentinel tokens
        mask_positions = set()
        for i in range(n_masks):
            pos = np.random.randint(0, max(1, len(words) - self.span_length))
            for j in range(self.span_length):
                if pos + j < len(words):
                    mask_positions.add(pos + j)

        masked_words = []
        sentinel_idx = 0
        prev_masked = False
        for i, word in enumerate(words):
            if i in mask_positions:
                if not prev_masked:
                    masked_words.append(f"<extra_id_{sentinel_idx}>")
                    sentinel_idx += 1
                prev_masked = True
            else:
                masked_words.append(word)
                prev_masked = False

        masked_text = " ".join(masked_words)

        # Generate fills with T5
        inputs = self.perturbation_tokenizer(
            masked_text, return_tensors="pt", max_length=512, truncation=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.perturbation_model.generate(
                **inputs, max_new_tokens=150, do_sample=True,
                temperature=1.0, top_p=0.96,
            )

        decoded = self.perturbation_tokenizer.decode(outputs[0], skip_special_tokens=False)

        # Parse the fills
        fills = {}
        for match in re.finditer(r"<extra_id_(\d+)>\s*(.*?)(?=<extra_id_|\Z|</s>)", decoded):
            idx = int(match.group(1))
            fill = match.group(2).strip()
            fills[idx] = fill

        # Reconstruct the text
        result_words = []
        sentinel_idx = 0
        prev_masked = False
        for i, word in enumerate(words):
            if i in mask_positions:
                if not prev_masked:
                    if sentinel_idx in fills:
                        result_words.append(fills[sentinel_idx])
                    sentinel_idx += 1
                prev_masked = True
            else:
                result_words.append(word)
                prev_masked = False

        return " ".join(result_words)

    def _word_shuffle_perturbation(self, text: str) -> str:
        """
        Fallback perturbation: word-level local shuffling.

        Not as good as T5, but avoids loading another model.
        """
        words = text.split()
        n_swap = max(1, int(len(words) * self.mask_pct))

        perturbed = words.copy()
        for _ in range(n_swap):
            i = np.random.randint(0, max(1, len(perturbed) - 1))
            j = min(i + np.random.randint(1, 4), len(perturbed) - 1)
            perturbed[i], perturbed[j] = perturbed[j], perturbed[i]

        return " ".join(perturbed)

    def score_text(
        self,
        text: str,
        num_perturbations: Optional[int] = None,
        max_length: int = 512,
    ) -> Dict[str, float]:
        """
        Score text using DetectGPT.

        Returns:
          - dgpt_curvature: the curvature score (main metric)
          - dgpt_original_ll: log-likelihood of original
          - dgpt_perturb_mean_ll: mean log-likelihood of perturbations
          - dgpt_perturb_std_ll: std
        """
        num_perturbations = num_perturbations or self.num_perturbations

        # Original log-likelihood
        original_ll = self._compute_ll(text, max_length)

        # Perturbation log-likelihoods
        perturb_lls = []
        for _ in range(num_perturbations):
            perturbed_text = self._mask_and_fill_t5(text)
            p_ll = self._compute_ll(perturbed_text, max_length)
            perturb_lls.append(p_ll)

        perturb_mean = np.mean(perturb_lls)
        perturb_std = np.std(perturb_lls) + 1e-8

        curvature = (original_ll - perturb_mean) / perturb_std

        return {
            "dgpt_curvature": curvature,
            "dgpt_original_ll": original_ll,
            "dgpt_perturb_mean_ll": perturb_mean,
            "dgpt_perturb_std_ll": perturb_std,
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
        iterator = tqdm(texts, desc="DetectGPT scoring", disable=not show_progress)

        for text in iterator:
            scores = self.score_text(text, num_perturbations, max_length)
            results.append(scores)
            if show_progress:
                iterator.set_postfix(curv=f"{scores['dgpt_curvature']:.4f}")

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
