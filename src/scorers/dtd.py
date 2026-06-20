"""
DiffuDetect — Denoising-Trajectory Dynamics (DTD) Scorer

For iterative diffusion models (LLaDA, Dream) that denoise step-by-step,
we track the TRAJECTORY of denoising, not just the final reconstruction.

Features extracted from the trajectory:
  1. Entropy-AUC: Area under the per-step entropy curve.
     AI text → lower entropy-AUC (model is more certain throughout).
  2. Mean Commit Time: Average denoising step at which each token is
     "committed" (probability of the final token exceeds threshold).
     AI text → commits earlier.
  3. Trajectory Curvature: Second derivative of the entropy curve.
     AI text → smoother trajectory (lower curvature).
  4. Remasking Dynamics: How often tokens flip during denoising.
     AI text → fewer flips (more stable trajectory).

This module is designed for models that support iterative unmasking,
like LLaDA-8B-Instruct and Dream-7B.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm

from ..config import DTDConfig, get_device
from ..utils import create_random_mask, tokenize_single


class DTDScorer:
    """
    Denoising-Trajectory Dynamics scorer.

    Requires an iterative diffusion model that can denoise step-by-step.
    Extracts trajectory-level features from the denoising process.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: Optional[DTDConfig] = None,
        device: Optional[str] = None,
        mask_token_id: Optional[int] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or DTDConfig()
        self.device = device or get_device()

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
        self.commit_threshold = 0.9  # token is "committed" when prob exceeds this

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

    def _create_denoising_schedule(
        self,
        num_steps: int,
        initial_mask_ratio: float,
    ) -> List[float]:
        """
        Create a linear unmasking schedule.

        Goes from initial_mask_ratio → 0.0 over num_steps.
        Each step unmasks a fraction of the remaining masked tokens.
        """
        ratios = np.linspace(initial_mask_ratio, 0.0, num_steps + 1)[:-1]
        return ratios.tolist()

    @torch.no_grad()
    def _iterative_denoise(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        num_steps: int,
        initial_mask_ratio: float,
    ) -> Dict[str, Any]:
        """
        Run iterative denoising and record the trajectory.

        At each step:
          1. Forward pass to get logits.
          2. Record per-token entropy and top-1 probability.
          3. Unmask the most confident tokens (lowest entropy).
          4. Repeat until all tokens are unmasked.

        Returns trajectory data: per-step entropies, commit times, token flips.
        """
        batch_size, seq_len = input_ids.shape

        # Create initial mask (mask initial_mask_ratio of tokens)
        currently_masked = create_random_mask(
            input_ids,
            mask_ratio=initial_mask_ratio,
            special_token_ids=self.special_token_ids,
            pad_token_id=self.pad_token_id,
        )

        # Current state: original tokens where unmasked, mask token elsewhere
        current_ids = input_ids.clone()
        current_ids[currently_masked] = self.mask_token_id

        # Track which positions were originally masked
        original_masked = currently_masked.clone()

        # Trajectory tracking
        n_originally_masked = currently_masked.sum().item()
        step_entropies = []           # mean entropy at each step
        per_token_entropies = []      # (step, position) entropy
        commit_times = torch.full(
            (batch_size, seq_len), num_steps, dtype=torch.float, device=self.device
        )
        committed = torch.zeros_like(currently_masked)  # which tokens have been committed
        token_flips = torch.zeros(
            batch_size, seq_len, dtype=torch.long, device=self.device
        )
        prev_predictions = torch.zeros_like(input_ids)

        schedule = self._create_denoising_schedule(num_steps, initial_mask_ratio)

        for step_idx, target_mask_ratio in enumerate(schedule):
            if not currently_masked.any():
                break

            # Forward pass
            try:
                outputs = self.model(
                    input_ids=current_ids,
                    attention_mask=attention_mask,
                )
            except TypeError:
                outputs = self.model(input_ids=current_ids)

            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            elif isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs

            probs = F.softmax(logits.float(), dim=-1)  # (batch, seq_len, vocab); float32 for stable entropy

            # Per-token entropy at masked positions
            token_entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)  # (batch, seq_len)

            # Record mean entropy over currently masked positions
            if currently_masked.any():
                mean_ent = token_entropy[currently_masked].mean().item()
                step_entropies.append(mean_ent)
                per_token_entropies.append(
                    token_entropy.detach().cpu().numpy()
                )

            # Top-1 predictions
            top1_probs, top1_tokens = probs.max(dim=-1)  # (batch, seq_len)

            # Track token flips (at masked positions only)
            if step_idx > 0:
                flipped = (top1_tokens != prev_predictions) & currently_masked & original_masked
                token_flips += flipped.long()
            prev_predictions = top1_tokens.clone()

            # Track commit times: if top-1 prob exceeds threshold and not yet committed
            newly_committed = (top1_probs > self.commit_threshold) & currently_masked & ~committed
            commit_times[newly_committed] = step_idx
            committed |= newly_committed

            # Determine how many tokens to unmask at this step
            n_currently_masked = currently_masked.sum().item()
            if n_currently_masked == 0:
                break

            n_target_masked = max(0, int(target_mask_ratio * n_originally_masked))
            n_to_unmask = max(0, n_currently_masked - n_target_masked)

            if n_to_unmask > 0:
                # Unmask the most confident tokens (lowest entropy)
                for b in range(batch_size):
                    masked_positions = currently_masked[b].nonzero(as_tuple=True)[0]
                    if len(masked_positions) == 0:
                        continue

                    # Get entropies at masked positions
                    pos_entropies = token_entropy[b, masked_positions]

                    # Select positions with lowest entropy to unmask
                    n_unmask_b = min(n_to_unmask, len(masked_positions))
                    _, unmask_indices = pos_entropies.topk(
                        n_unmask_b, largest=False
                    )
                    unmask_positions = masked_positions[unmask_indices]

                    # Unmask: replace with model's prediction
                    current_ids[b, unmask_positions] = top1_tokens[b, unmask_positions]
                    currently_masked[b, unmask_positions] = False

        # Compute trajectory features
        trajectory = {
            "step_entropies": step_entropies,
            "per_token_entropies": per_token_entropies,
            "commit_times": commit_times,
            "token_flips": token_flips,
            "original_masked": original_masked,
            "n_steps": len(step_entropies),
        }

        return trajectory

    def _extract_features(
        self,
        trajectory: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Extract scalar features from a denoising trajectory.

        Features:
          - dtd_entropy_auc: area under the entropy curve
          - dtd_mean_commit_time: average step at which tokens commit
          - dtd_trajectory_curvature: second derivative of entropy curve
          - dtd_mean_flips: average number of token flips during denoising
          - dtd_final_entropy: entropy at the last denoising step
          - dtd_entropy_drop: initial entropy - final entropy
        """
        step_entropies = trajectory["step_entropies"]
        commit_times = trajectory["commit_times"]
        token_flips = trajectory["token_flips"]
        original_masked = trajectory["original_masked"]
        n_steps = trajectory["n_steps"]

        features = {}

        # 1. Entropy-AUC (trapezoidal integration)
        if len(step_entropies) >= 2:
            features["dtd_entropy_auc"] = float(np.trapz(step_entropies))
        else:
            features["dtd_entropy_auc"] = step_entropies[0] if step_entropies else 0.0

        # 2. Mean commit time (normalized by total steps)
        masked_commit = commit_times[original_masked]
        if len(masked_commit) > 0:
            features["dtd_mean_commit_time"] = masked_commit.float().mean().item() / max(n_steps, 1)
        else:
            features["dtd_mean_commit_time"] = 1.0

        # 3. Trajectory curvature (mean absolute second derivative)
        if len(step_entropies) >= 3:
            ent_array = np.array(step_entropies)
            second_derivative = np.diff(ent_array, n=2)
            features["dtd_trajectory_curvature"] = float(np.mean(np.abs(second_derivative)))
        else:
            features["dtd_trajectory_curvature"] = 0.0

        # 4. Mean token flips at originally masked positions
        masked_flips = token_flips[original_masked]
        if len(masked_flips) > 0:
            features["dtd_mean_flips"] = masked_flips.float().mean().item()
        else:
            features["dtd_mean_flips"] = 0.0

        # 5. Final entropy
        features["dtd_final_entropy"] = step_entropies[-1] if step_entropies else 0.0

        # 6. Entropy drop (initial - final)
        if len(step_entropies) >= 2:
            features["dtd_entropy_drop"] = step_entropies[0] - step_entropies[-1]
        else:
            features["dtd_entropy_drop"] = 0.0

        # 7. Commit time variance (how spread out are commit times?)
        if len(masked_commit) > 1:
            features["dtd_commit_time_std"] = masked_commit.float().std().item() / max(n_steps, 1)
        else:
            features["dtd_commit_time_std"] = 0.0

        return features

    def score_text(
        self,
        text: str,
        num_steps: Optional[int] = None,
        num_draws: Optional[int] = None,
        initial_mask_ratio: Optional[float] = None,
        max_length: int = 512,
    ) -> Dict[str, float]:
        """
        Score a single text passage using DTD features.

        Runs multiple denoising trajectories and averages the features.
        """
        num_steps = num_steps or self.config.num_denoising_steps
        num_draws = num_draws or self.config.num_mask_draws
        initial_mask_ratio = initial_mask_ratio or self.config.initial_mask_ratio

        encoding = tokenize_single(text, self.tokenizer, max_length, self.device)
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        all_features = []
        for _ in range(num_draws):
            trajectory = self._iterative_denoise(
                input_ids, attention_mask, num_steps, initial_mask_ratio
            )
            features = self._extract_features(trajectory)
            all_features.append(features)

        # Average across draws
        averaged = {}
        for key in all_features[0]:
            values = [f[key] for f in all_features]
            averaged[key] = float(np.mean(values))

        return averaged

    def score_batch(
        self,
        texts: List[str],
        num_steps: Optional[int] = None,
        num_draws: Optional[int] = None,
        initial_mask_ratio: Optional[float] = None,
        max_length: int = 512,
        show_progress: bool = True,
    ) -> List[Dict[str, float]]:
        """Score a batch of texts."""
        results = []
        iterator = tqdm(texts, desc="DTD scoring", disable=not show_progress)

        for text in iterator:
            scores = self.score_text(
                text, num_steps, num_draws, initial_mask_ratio, max_length
            )
            results.append(scores)
            if show_progress:
                iterator.set_postfix(
                    eauc=f"{scores['dtd_entropy_auc']:.2f}",
                    ct=f"{scores['dtd_mean_commit_time']:.3f}",
                )

        return results

    def score_dataset(
        self,
        df,
        text_col: str = "text",
        num_steps: Optional[int] = None,
        num_draws: Optional[int] = None,
        initial_mask_ratio: Optional[float] = None,
        max_length: int = 512,
    ) -> List[Dict[str, float]]:
        """Score all texts in a DataFrame."""
        texts = df[text_col].tolist()
        return self.score_batch(
            texts, num_steps, num_draws, initial_mask_ratio, max_length
        )
