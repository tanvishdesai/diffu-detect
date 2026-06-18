# PLANNING.md — DiffuDetect

**Diffusion Language Models as Zero-Shot Detectors of AI-Generated Text**

> Living design document. This is the single source of truth for the project's vision, architecture, constraints, and definitions of success. Update it when a decision changes; track day-to-day work in `TASK.md`.

---

## 1. One-paragraph vision

Autoregressive (AR) language models write left-to-right; masked **diffusion** language models (dLLMs) instead reconstruct blanked-out tokens using bidirectional context. The thesis of DiffuDetect is that **machine-generated text is systematically easier for a diffusion LM to reconstruct than human text**, because AI text is itself high-probability under language models and lacks the idiosyncratic, locally-surprising structure of human writing. We turn a frozen, pretrained diffusion LM into a **training-free detector** by reading out its native denoising behaviour — masked-reconstruction error, a diffusion analogue of probability-curvature, and denoising-trajectory dynamics — as the detection statistic. The central bet, and the wedge that makes this publishable rather than redundant with Fast-DetectGPT, is **robustness to paraphrase / light editing**: AR-curvature detectors collapse under paraphrase, and a bidirectional reconstruction signal may degrade more gracefully.

## 2. Why this is novel (and the honest risk)

- **Cross-field transfer.** Imports the reconstruction-error OOD-detection paradigm from image forensics (DIRE/DDPM) and the perturbation-curvature paradigm from AR detection (DetectGPT / Fast-DetectGPT) **into the diffusion-LM setting for human-vs-AI authorship detection.** As of the last sweep this exact reframing is unclaimed; the nearest neighbour (AuthentiGPT) uses a black-box AR model to denoise, not a diffusion LM, and is not training-free.
- **The risk, stated plainly.** The diffusion reconstruction signal may be *highly correlated* with AR probability-curvature — both ultimately measure "how predictable is this text." If so, DiffuDetect **matches** Fast-DetectGPT on clean text but does not **beat** it, which is a reject at AAAI. **The paper lives or dies on the paraphrase-robustness experiment (Phase 3).** Everything before Phase 3 exists to earn the right to run Phase 3.
- **Minimum viable contribution.** A rigorous benchmark showing diffusion-LM scoring is (a) competitive on clean text and (b) clearly more robust under paraphrase/adversarial edits, plus an analysis of *which* signal carries the discriminative power and *why*.

## 3. Target venue & timeline

- **Primary:** AAAI-27 Main Technical Track. Abstract due **2026-07-20**, full paper **2026-07-27** (AoE; confirm on official CFP). 7 content pages + unlimited references.
- **Fallback:** AAAI-28 cycle, or a strong workshop (e.g., an NLP/security workshop) if the robustness win is real but the clean-text numbers are merely competitive.
- This is the most schedule-safe of the three projects because it requires **zero training**.

## 4. Constraints & resource profile

- **Compute reality:** inference-rich, training-GPU-poor. One ~16 GB T4 per Kaggle notebook; 6–7 accounts for parallelism; Colab; large Google Drive; $100/month/model frontier inference (mostly irrelevant here — DiffuDetect uses open scorers, not API LLMs).
- **No training, no fine-tuning.** Pure inference. No RunPod required.
- **Parallelism strategy:** one (model × dataset × statistic) cell per notebook/account; write all scores to Drive as Parquet; aggregate/plot in a single lightweight notebook.
- **The real cost is wall-clock inference time:** diffusion scoring is multi-pass (e.g., 16–32 denoising/mask samples per passage), so throughput, not VRAM, is the binding constraint. Cache every score; never recompute.

## 5. Models (all open, all HuggingFace)

| Role | Model | HF repo (verify exact string) | Fits T4? | Notes |
|---|---|---|---|---|
| Primary diffusion scorer (small) | SMDM-1.1B | `nieshen/SMDM` family (170M–1.1B) | Yes, unquantized | Native masked diffusion; main workhorse |
| Tiny ablation scorer | MDLM ~110M | `kuleshov-group/mdlm-owt` | Yes | Cheap signal ablations |
| Large diffusion scorer | LLaDA-8B-Instruct | `GSAI-ML/LLaDA-8B-Instruct` | Yes @ 4-bit (~6 GB) | Iterative denoising → trajectory features |
| Large diffusion scorer | Dream-7B | `Dream-org/Dream-v0-Instruct-7B` | Yes @ 4-bit | Second trajectory model for robustness of claims |
| AR baseline scorer | GPT-Neo-2.7B / GPT-J-6B | `EleutherAI/*` | Yes (J @ 4-bit) | For DetectGPT / Fast-DetectGPT |
| Binoculars pair | Falcon-7B + Falcon-7B-Instruct | `tiiuae/*` | Tight; 4-bit both | Two 7B models; quantize aggressively or swap to a smaller observer/performer pair |
| Supervised reference | MAGE Longformer | `yaful/...` | Yes | "Trained detector" upper-reference, not a baseline we must beat |

## 6. Datasets (all small, all Kaggle-hostable)

| Dataset | Source | Size | Role |
|---|---|---|---|
| **MAGE** | `yaful/MAGE` (HF) | 554 MB, 436k rows | Core: in-distribution + cross-domain/cross-model testbeds + 2 wild OOD sets (incl. a **paraphrase** set) |
| **RAID** | `liamdugan/raid` (HF) | medium | **Key robustness testbed** — many adversarial attacks incl. paraphrase, synonym, whitespace |
| **M4GT-Bench** | HF | medium | Multi-generator / multilingual generalization |
| HC3 (optional) | HF | small | Extra human-vs-ChatGPT slice |

All download via `datasets.load_dataset(...)` directly inside Kaggle (internet on) or upload once as a Kaggle Dataset and attach to every notebook. None require special access.

## 7. Detection statistics (the scientific core)

Define each as a scalar score `s(x)` per passage `x`; lower ⇒ more likely AI (sign per statistic).

1. **Masked Reconstruction Error (MRE).** Mask a fraction `r` of tokens (sweep `r ∈ {0.15, 0.3, 0.5}`), have the dLLM predict them, compute mean NLL of the *true* tokens at masked positions; average over `K` random mask draws. Hypothesis: AI text → lower MRE.
2. **Diffusion Curvature (DC).** Analogue of Fast-DetectGPT's conditional curvature: re-mask & reconstruct to sample alternative continuations, measure the local curvature / gap of the reconstruction-likelihood surface around `x`. Hypothesis: AI text sits at sharper local maxima.
3. **Denoising-Trajectory Dynamics (DTD).** For iterative models (LLaDA, Dream): track per-step entropy trajectory, per-token commit time (which denoising step finalizes each token), and remasking dynamics. Features: entropy-AUC, mean commit time, trajectory curvature. Hypothesis: AI text commits earlier / with lower entropy.

**Combiner:** report each statistic standalone (zero-shot AUROC) **and** a simple logistic regression over the 3 statistics fit on a small calibration split (still essentially training-free — a 3-parameter head).

## 8. Baselines (must be strong, not strawmen)

- **Fast-DetectGPT** (white-box and black-box settings) — the primary baseline to beat on robustness.
- **DetectGPT** — the weak-OOD reference.
- **Binoculars** — strong zero-shot, untuned.
- **Log-likelihood / rank / entropy (GLTR-style)** — cheap classical baselines.
- **Supervised Longformer (MAGE)** — upper-reference for "what a trained detector gets," shown for context, not a target.

## 9. Pipeline / execution flow

```
[passage x]
   → tokenize
   → for each statistic S in {MRE, DC, DTD}:
        run frozen dLLM inference (K samples / mask ratios)
        → scalar score s_S(x)
   → write {id, label, generator, domain, s_MRE, s_DC, s_DTD} to Parquet on Drive
   → (aggregation notebook) compute AUROC / TPR@FPR per (statistic, dataset, generator)
   → (robustness notebook) clean vs paraphrased ΔAUROC, per method
   → tables + figures
```

Parallelize the middle step across accounts by (model × dataset shard). All downstream analysis is a single CPU-light notebook reading Parquet.

## 10. Metrics

- **Primary:** AUROC (per dataset, per generator).
- **Operating points:** TPR@1%FPR and TPR@5%FPR (deployment-relevant; detectors are graded at low FPR).
- **Calibrated:** Accuracy / F1 at a fixed threshold.
- **Robustness (THE metric):** ΔAUROC = AUROC(clean) − AUROC(paraphrased), per method. Smaller drop = win.
- **Generalization:** calibrate on generator set A, evaluate on unseen set B (cross-generator AUROC).
- **Cost (report honestly):** scoring latency per passage and tokens/sec for each method — diffusion is multi-pass and slower; we own this.

## 11. What a "good result" looks like (publishable bar)

- **Clean text:** DiffuDetect AUROC within ~1–2 points of Fast-DetectGPT across MAGE testbeds (competitive — not necessarily winning).
- **Under paraphrase (decisive):** baselines drop sharply (e.g., Fast-DetectGPT −15 to −25 AUROC; TPR@1%FPR collapses), while DiffuDetect drops materially less (e.g., < −10). A clear, consistent robustness gap across RAID attacks and the MAGE paraphrase set.
- **Cross-generator:** competitive-or-better transfer to unseen generators.
- **Analysis:** an ablation isolating which statistic (MRE/DC/DTD) drives the robustness advantage, with an interpretable explanation.
- **Stretch:** the robustness gap holds for both SMDM-1.1B and a 7–8B diffusion model (shows it's a property of diffusion scoring, not one checkpoint).

## 12. Key risks & mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Diffusion signal ≈ AR-curvature (no differentiation) | Medium-high | Lead with robustness, not clean-text; kill early at Phase-1/Phase-3 gates |
| Multi-pass inference too slow for full datasets | Medium | Subsample to fixed N per testbed; cache; parallelize across accounts |
| 4-bit quantization degrades trajectory signal | Medium | Validate signal on unquantized SMDM-1.1B first; treat 7–8B as confirmation |
| Field moves / gets scooped before July | Medium | Re-run literature sweep the week before submission |
| Paraphrase robustness gap is real but small | Medium | Still a workshop paper; report effect size honestly |

## 13. Tech stack

- Python, PyTorch, HuggingFace `transformers` + `datasets`, `bitsandbytes` (4-bit), `peft` (not needed unless adding head), `scikit-learn` (AUROC/logistic), `pandas`/`pyarrow` (Parquet), `matplotlib` (figures).
- Reference implementations to port: Fast-DetectGPT repo (Bao et al.), Binoculars repo (Hans et al.), DetectGPT repo.
- Repo layout: `src/scorers/` (one file per statistic), `src/baselines/`, `src/eval/`, `notebooks/` (Kaggle), `results/` (Parquet on Drive), `paper/`.

## 14. Definition of done

A 7-page AAAI paper with: (1) the method, (2) clean-text competitiveness table, (3) the robustness table/figure that is the headline, (4) cross-generator generalization, (5) the signal ablation, (6) released code + cached scores. Go/no-go gates in `TASK.md` decide whether we reach this or pivot to Proposal 5.
