"""
DiffuDetect — Diffusion Curvature (DC) Scorer

Analogue of Fast-DetectGPT's conditional probability curvature, adapted
for masked diffusion language models.

Algorithm:
  1. Mask a fraction r of tokens in the original passage x.
  2. Let the diffusion LM reconstruct → get reconstructed sequence x'.
  3. Repeat N times to get N perturbed versions {x'_1, ..., x'_N}.
  4. For each x'_i, compute the MRE (masked reconstruction error).
  5. Curvature = MRE(original x) - mean(MRE(x'_i))
     i.e., how much the original sits above the perturbation landscape.

Hypothesis: AI text sits at sharper local maxima → higher curvature.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Any, Dict, List, Optional
from tqdm import tqdm

from ..config import DCConfig, get_device
from ..utils import (
    create_random_mask,
    tokenize_single,
)


class DCScorer:
    """
    Diffusion Curvature scorer.

    Measures the local curvature of the reconstruction-likelihood surface
    around the original text. AI text is hypothesized to sit at sharper
    local maxima (higher curvature).
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: Optional[DCConfig] = None,
        device: Optional[str] = None,
        mask_token_id: Optional[int] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or DCConfig()
        self.device = device or get_device()

        # Determine mask token ID (same logic as MRE)
        if mask_token_id is not None:
            self.mask_token_id = mask_token_id
        elif hasattr(tokenizer, 'mask_token_id') and tokenizer.mask_token_id is not None:
            self.mask_token_id = tokenizer.mask_token_id
        else:
            self.mask_token_id = self._find_mask_token_id()

        self.special_token_ids = set()
        for attr in ['bos_token_id', 'eos_token_id', 'pad_token_id', 'cls_token_id', 'sep_token_id']:
            tid = getattr(tokenizer, attr, None)
            if tid is not None:
                self.special_token_ids.add(tid)

        self.pad_token_id = tokenizer.pad_token_id

    def _find_mask_token_id(self) -> int:
        """Find or create a mask token ID."""
        if "[MASK]" in self.tokenizer.get_vocab():
            return self.tokenizer.convert_tokens_to_ids("[MASK]")
        if "<mask>" in self.tokenizer.get_vocab():
            return self.tokenizer.convert_tokens_to_ids("<mask>")
        if hasattr(self.model, 'config') and hasattr(self.model.config, 'mask_token_id'):
            return self.model.config.mask_token_id
        if self.tokenizer.unk_token_id is not None:
            return self.tokenizer.unk_token_id
        return len(self.tokenizer) - 1

    @torch.no_grad()
    def _compute_mre_for_ids(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        mask_ratio: float,
        num_draws: int = 8,
    ) -> float:
        """Compute MRE score for given input_ids (used for both original and perturbed)."""
        nlls = []
        for _ in range(num_draws):
            bool_mask = create_random_mask(
                input_ids,
                mask_ratio=mask_ratio,
                special_token_ids=self.special_token_ids,
                pad_token_id=self.pad_token_id,
            )

            true_tokens = input_ids.clone()
            masked_input = input_ids.clone()
            masked_input[bool_mask] = self.mask_token_id

            try:
                outputs = self.model(
                    input_ids=masked_input,
                    attention_mask=attention_mask,
                )
            except TypeError:
                outputs = self.model(input_ids=masked_input)

            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            elif hasattr(outputs, 'last_hidden_state'):
                if hasattr(self.model, 'lm_head'):
                    logits = self.model.lm_head(outputs.last_hidden_state)
                else:
                    logits = outputs.last_hidden_state
            elif isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs

            log_probs = F.log_softmax(logits.float(), dim=-1)
            true_log_probs = log_probs.gather(
                dim=-1, index=true_tokens.unsqueeze(-1)
            ).squeeze(-1)

            masked_nlls = -true_log_probs[bool_mask]
            if len(masked_nlls) > 0:
                nlls.append(masked_nlls.mean().item())

        return np.mean(nlls) if nlls else 0.0

    @torch.no_grad()
    def _generate_perturbation(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        mask_ratio: float,
    ) -> torch.Tensor:
        """
        Generate a single perturbed version of the text.

        Mask some tokens → let the model reconstruct → take the argmax predictions.
        """
        bool_mask = create_random_mask(
            input_ids,
            mask_ratio=mask_ratio,
            special_token_ids=self.special_token_ids,
            pad_token_id=self.pad_token_id,
        )

        masked_input = input_ids.clone()
        masked_input[bool_mask] = self.mask_token_id

        try:
            outputs = self.model(
                input_ids=masked_input,
                attention_mask=attention_mask,
            )
        except TypeError:
            outputs = self.model(input_ids=masked_input)

        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        elif isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs

        # Sample from the model's distribution at masked positions
        # Using argmax for deterministic perturbations (or temperature sampling)
        probs = F.softmax(logits, dim=-1)

        # Sample from the distribution for diversity
        perturbed = input_ids.clone()
        for idx in bool_mask.nonzero(as_tuple=False):
            batch_idx, pos_idx = idx[0], idx[1]
            token_probs = probs[batch_idx, pos_idx]
            sampled_token = torch.multinomial(token_probs, 1).item()
            perturbed[batch_idx, pos_idx] = sampled_token

        return perturbed

    def score_text(
        self,
        text: str,
        mask_ratio: Optional[float] = None,
        num_perturbations: Optional[int] = None,
        num_mask_draws: Optional[int] = None,
        max_length: int = 512,
    ) -> Dict[str, float]:
        """
        Score a single text passage using Diffusion Curvature.

        Returns:
          - dc_curvature: MRE(original) - mean(MRE(perturbations))
          - dc_original_mre: MRE of the original text
          - dc_perturb_mean_mre: mean MRE of perturbations
          - dc_perturb_std_mre: std of perturbation MREs
          - dc_normalized: curvature / std(perturbation MREs) (z-score)
        """
        mask_ratio = mask_ratio or self.config.mask_ratio
        num_perturbations = num_perturbations or self.config.num_perturbations
        num_mask_draws = num_mask_draws or self.config.num_mask_draws

        # Tokenize
        encoding = tokenize_single(text, self.tokenizer, max_length, self.device)
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        # 1. Compute MRE for the original text
        original_mre = self._compute_mre_for_ids(
            input_ids, attention_mask, mask_ratio, num_mask_draws
        )

        # 2. Generate perturbations and compute their MREs
        perturb_mres = []
        for _ in range(num_perturbations):
            perturbed_ids = self._generate_perturbation(
                input_ids, attention_mask, mask_ratio
            )
            p_mre = self._compute_mre_for_ids(
                perturbed_ids, attention_mask, mask_ratio,
                num_draws=max(4, num_mask_draws // 2),  # fewer draws for perturbations
            )
            perturb_mres.append(p_mre)

        perturb_mean = np.mean(perturb_mres)
        perturb_std = np.std(perturb_mres) + 1e-8  # avoid div by zero

        # 3. Curvature = how much the original differs from perturbation landscape
        curvature = original_mre - perturb_mean
        normalized = curvature / perturb_std

        return {
            "dc_curvature": curvature,
            "dc_original_mre": original_mre,
            "dc_perturb_mean_mre": perturb_mean,
            "dc_perturb_std_mre": perturb_std,
            "dc_normalized": normalized,
        }

    def score_batch(
        self,
        texts: List[str],
        mask_ratio: Optional[float] = None,
        num_perturbations: Optional[int] = None,
        num_mask_draws: Optional[int] = None,
        max_length: int = 512,
        show_progress: bool = True,
    ) -> List[Dict[str, float]]:
        """Score a batch of texts."""
        results = []
        iterator = tqdm(texts, desc="DC scoring", disable=not show_progress)

        for text in iterator:
            scores = self.score_text(
                text, mask_ratio, num_perturbations, num_mask_draws, max_length
            )
            results.append(scores)
            if show_progress:
                iterator.set_postfix(dc=f"{scores['dc_normalized']:.4f}")

        return results

    def score_dataset(
        self,
        df,
        text_col: str = "text",
        mask_ratio: Optional[float] = None,
        num_perturbations: Optional[int] = None,
        num_mask_draws: Optional[int] = None,
        max_length: int = 512,
    ) -> List[Dict[str, float]]:
        """Score all texts in a DataFrame."""
        texts = df[text_col].tolist()
        return self.score_batch(
            texts, mask_ratio, num_perturbations, num_mask_draws, max_length
        )
