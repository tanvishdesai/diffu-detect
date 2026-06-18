# TASK.md — DiffuDetect

> Working task tracker. `[ ]` todo, `[~]` in progress, `[x]` done, `[!]` blocked. Add anything discovered mid-process to the "Discovered mid-process" section at the bottom. Pair with `PLANNING.md`.

**Current phase:** Phase 0 — setup & baseline reproduction
**Headline bet:** robustness-to-paraphrase win over Fast-DetectGPT (decided at Phase 3)
**Hard deadline:** AAAI-27 paper 2026-07-27 (fallback AAAI-28)

---

## Milestones (decision gates)

- **M0 — Baselines reproduced.** Fast-DetectGPT + Binoculars running on a MAGE slice, numbers in the right ballpark vs published.
- **M1 — First signal.** MRE statistic on SMDM-1.1B yields above-chance AUROC on clean MAGE.
  - **GO/NO-GO #1:** Is any single statistic AUROC ≥ ~0.85 on clean in-distribution MAGE? If no statistic clears chance-plus, the core premise is weak → reconsider or pivot.
- **M2 — Competitive on clean.** Best DiffuDetect config within ~2 AUROC points of Fast-DetectGPT on MAGE clean testbeds.
- **M3 — ROBUSTNESS VERDICT (the project's pivot point).** On RAID + MAGE-paraphrase: DiffuDetect ΔAUROC clearly smaller than baselines'.
  - **GO/NO-GO #2 (decisive):** Is DiffuDetect's paraphrase ΔAUROC at least ~8–10 points better than Fast-DetectGPT's? **YES → this is the paper, push to AAAI-27. NO → demote to workshop or pivot effort to Proposal 5.**
- **M4 — Generalization + scale.** Cross-generator transfer done; robustness gap confirmed on a 7–8B diffusion model.
- **M5 — Paper.** Tables, figures, ablation, code release, draft complete.

---

## Phase 0 — Setup & baselines
- [ ] Stand up Kaggle env template (transformers, datasets, bitsandbytes, sklearn, pyarrow) + Drive mount pattern.
- [ ] `load_dataset("yaful/MAGE")`; carve the 6 testbeds + the 2 wild OOD sets (incl. paraphrase) per the MAGE script.
- [ ] Download RAID; map its attack taxonomy (paraphrase, synonym, whitespace, etc.) to evaluation slices.
- [ ] Port Fast-DetectGPT (white-box + black-box) into a Kaggle notebook; reproduce a sanity AUROC on a MAGE slice.
- [ ] Port Binoculars (Falcon observer/performer, 4-bit); reproduce sanity AUROC.
- [ ] Port DetectGPT + classical (log-lik / rank / entropy) baselines.
- [ ] Define the canonical results schema `{id, label, generator, domain, attack, method, score}` → Parquet on Drive.

## Phase 1 — MRE statistic (kill-early gate)
- [ ] Load SMDM-1.1B (unquantized, T4); implement masked-fill scoring.
- [ ] Implement **MRE** with mask-ratio sweep `{0.15,0.3,0.5}` × `K` mask draws; tune `K` for stability vs cost.
- [ ] Compute clean-MAGE AUROC per generator/domain. **→ evaluate GO/NO-GO #1.**
- [ ] Sensitivity check: AUROC vs `K` and `r` (pick the cheapest stable config).

## Phase 2 — Curvature + trajectory statistics
- [ ] Implement **Diffusion Curvature (DC)** (conditional-sample analogue of Fast-DetectGPT).
- [ ] Load LLaDA-8B-Instruct (4-bit) and Dream-7B (4-bit); implement **DTD** features (entropy-AUC, commit-time, trajectory curvature).
- [ ] Per-statistic clean AUROC table (MRE vs DC vs DTD vs baselines). **→ M2.**
- [ ] Fit 3-feature logistic combiner on a small calibration split; report combined AUROC.

## Phase 3 — ROBUSTNESS (the decisive experiment)
- [ ] Build the clean↔attacked pairing across RAID attacks + MAGE paraphrase set.
- [ ] Compute ΔAUROC and ΔTPR@1%FPR for **every** method (DiffuDetect statistics + all baselines).
- [ ] Plot robustness curves (AUROC vs attack strength) per method. **→ evaluate GO/NO-GO #2 (decisive).**
- [ ] If GO: lock the headline figure. If NO-GO: write the honest negative/workshop framing, shift effort to Proposal 5.

## Phase 4 — Generalization & scale
- [ ] Cross-generator protocol: calibrate on generator set A, test on unseen B; report transfer AUROC.
- [ ] Confirm the robustness gap reproduces on a 7–8B diffusion model (not just SMDM-1.1B).
- [ ] Cost table: latency/passage + tokens/sec per method (own the slowness honestly).
- [ ] Signal-attribution ablation: which statistic drives robustness? Interpret why.

## Phase 5 — Paper
- [ ] Draft 7 pages: method → clean competitiveness → robustness headline → generalization → ablation.
- [ ] Final literature sweep (re-verify novelty the week before submission).
- [ ] Clean + release code; publish cached scores; write reproducibility appendix.
- [ ] Internal review pass; submit abstract (07-20), paper (07-27).

---

## Backlog / nice-to-have
- [ ] Multilingual slice via M4GT-Bench.
- [ ] Add an HC3 human-vs-ChatGPT slice.
- [ ] Try mask-ratio *scheduling* (curriculum of `r`) as a 4th statistic.
- [ ] Adversarial-aware variant: does scoring under multiple mask seeds defeat targeted attacks?
- [ ] Compute-matched comparison (equalize FLOPs across methods) for a fairer cost story.

## Discovered mid-process
- _(log surprises, dead ends, retuned hyperparameters, dataset quirks here as they appear)_

## Open questions
- Does 4-bit quantization measurably weaken the trajectory (DTD) signal vs full precision? (Validate on SMDM unquantized.)
- Is DC actually distinct from MRE, or do they collapse to the same ranking? (Correlate the two score vectors early.)
- What's the minimum passage length where the signal is reliable? (Short texts may be undetectable for everyone.)
