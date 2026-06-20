# TASK.md — DiffuDetect

> Working task tracker. `[ ]` todo, `[~]` in progress, `[x]` done, `[!]` blocked. Add anything discovered mid-process to the "Discovered mid-process" section at the bottom. Pair with `PLANNING.md`.

**Current phase:** Phase 1 re-run on v2 fixes — within-testbed eval + fixed FDGPT baseline
**Headline bet:** robustness-to-paraphrase win over Fast-DetectGPT (decided at Phase 3)
**Hard deadline:** AAAI-27 paper 2026-07-27 (fallback AAAI-28)
**v2 note:** the v1 "NO-GO" was a pooling artifact and the FDGPT baseline was broken;
both fixed. See "Discovered mid-process" below and `RUNBOOK.md` for the re-run plan.

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
- **v1 "NO-GO" was a measurement artifact, not a weak method.** The Phase-1 gate
  fired on a single AUROC over the fully-POOLED set (all human domains vs all
  machine generators) = 0.60. But per-generator AUROC was 0.9+ for 64 generators
  and MRE *beat* Fast-DetectGPT per-generator (0.742 vs 0.714, winning 176/303).
  Cause: Simpson's-paradox pooling — absolute MRE scale shifts across domains so
  one global threshold can't separate them. **Fix:** evaluation is now
  within-testbed (mean per-(dataset,domain) AUROC) as the primary metric; pooled
  kept only as a labelled reference. Both GO/NO-GO gates re-evaluated on it.
- **Fast-DetectGPT baseline was broken (M0 not actually met).** v1 used a
  Monte-Carlo approximation (~50–100 perturbations, full forward each) → 0.53
  AUROC, 13.96 s/passage. Replaced with the **analytic** closed-form sampling
  discrepancy (Bao et al.): one forward pass, ~40× faster, reproduces published
  within-testbed numbers. Until this, "MRE beats FDGPT" was comparing against a
  crippled baseline.
- **`cls_mean_entropy` came back AUROC=nan.** fp16 entropy summed over the ~50k
  GPT-Neo vocab overflows. Fixed by upcasting logits to float32 before
  softmax/log in all scorers (classical, MRE, DC, DTD, FDGPT).
- **MRE padded every passage to 512 tokens with EOS.** SMDM has no attention
  mask, so it attended over the pad wall (noise + wasted compute, and ~Nx wasted
  on the 8B models). Switched single-passage scoring to no fixed-length padding.
- **NB03 had a crash bug:** `get_device_properties().total_mem` (should be
  `total_memory`) aborted the GPU diagnostics before DTD could run — likely part
  of why LLaDA/Dream "wouldn't run". Fixed.
- **LLaDA-8B / Dream-7B quantization integrated.** Load via trust_remote_code +
  bitsandbytes 4-bit (NOT GGUF — we need raw masked-position logits). LLaDA →
  AutoModelForCausalLM, mask_token_id=126336; Dream → AutoModel,
  mask_token_id=151666 (both also in model.config). T4×1 @4-bit ~6 GB, or fp16 on
  T4×2 via device_map="auto". See `RUNBOOK.md` for the full run sequence.

## Status of gates after v2 fixes (to be filled in on the next run)
- M0 (baselines reproduced): FDGPT now analytic → expect within-testbed ≈0.9. **Re-verify.**
- GO/NO-GO #1: re-read `auroc_within_testbed` in NB05 STEP 2 (NOT pooled).
- GO/NO-GO #2: re-read within-testbed paraphrase ΔAUROC in NB05 STEP 3.

## Open questions
- Does 4-bit quantization measurably weaken the trajectory (DTD) signal vs full precision? (Validate on SMDM unquantized.)
- Is DC actually distinct from MRE, or do they collapse to the same ranking? (Correlate the two score vectors early.)
- What's the minimum passage length where the signal is reliable? (Short texts may be undetectable for everyone.)
