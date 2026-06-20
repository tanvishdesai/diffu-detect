# RUNBOOK — DiffuDetect v2 (what to run, in what order)

This is the run plan after the v2 fixes (within-testbed evaluation, analytic
Fast-DetectGPT, entropy-NaN fix, quantized LLaDA-8B / Dream-7B integration).

## TL;DR of what changed and why it matters

- **Evaluation is now within-testbed.** v1's headline AUROC (0.60, "NO-GO") was a
  pooling artifact — per-generator AUROC was already 0.9+. NB05 now reports
  **mean within-testbed (per-domain) AUROC** as the primary metric and re-runs
  both GO/NO-GO gates on it. Pooled AUROC is kept only as a labelled reference.
- **Fast-DetectGPT is fixed.** v1 ran a Monte-Carlo approximation (0.53 AUROC,
  13.96 s/passage). NB04/NB06 now use the **analytic** closed form: ~40× faster
  and reproduces the published within-testbed numbers. The baseline is now real.
- **`cls_mean_entropy` NaN fixed** (float32 upcast before softmax/entropy).
- **LLaDA-8B and Dream-7B are integrated** as 4-bit diffusion scorers for both
  MRE (NB02) and DTD (NB03). No GGUF — we need raw masked-position logits, which
  llama.cpp can't provide for these architectures; on-the-fly bitsandbytes 4-bit
  is the correct path (and fp16 on Kaggle T4×2 if you want zero quantization).

---

## Kaggle accelerator choice

- **SMDM / GPT-Neo / classical / FDGPT**: `GPU T4 ×1` is fine.
- **LLaDA-8B / Dream-7B**:
  - `GPU T4 ×1` with 4-bit (default, `QUANTIZE_BITS=4`) → ~6 GB, works.
  - **Preferred for the validation run: `GPU T4 ×2`** → `device_map="auto"` shards
    the model; you can even run fp16 (no quantization) across the two 16 GB GPUs.
    Use this to confirm 4-bit doesn't weaken the signal, then 4-bit at scale.

---

## STAGE A — Scoring (GPU). All cells below are INDEPENDENT → run in parallel
## across your 6–7 Kaggle accounts. Each writes one parquet to /kaggle/working/results.

| # | Notebook | Set in the config cell | Output parquet | Accel |
|---|----------|------------------------|----------------|-------|
| A1 | `02_mre_scoring.py` | `MODEL_NAME="smdm-1.1b"` | `scores_mage_mre_smdm-1.1b.parquet` | T4×1 |
| A2 | `02_mre_scoring.py` | `MODEL_NAME="llada-8b"` | `scores_mage_mre_llada-8b.parquet` | T4×1 (or ×2) |
| A3 | `02_mre_scoring.py` | `MODEL_NAME="dream-7b"` | `scores_mage_mre_dream-7b.parquet` | T4×1 (or ×2) |
| A4 | `04_baseline_scoring.py` | (defaults) | `scores_mage_classical_*` + `scores_mage_fast_detectgpt_*` | T4×1 |
| A5 | `03_dc_dtd_scoring.py` | `RUN_DTD=True` (LLaDA) | `scores_mage_dtd_llada-8b.parquet` | T4×1 (or ×2) |
| A6 | `06_raid_robustness.py` | `RUN_METHOD="mre"` | `scores_raid_mre_smdm-1.1b.parquet` | T4×1 |
| A7 | `06_raid_robustness.py` | `RUN_METHOD="fast_detectgpt"` | `scores_raid_fast_detectgpt_*` | T4×1 |
| A8 | `06_raid_robustness.py` | `RUN_METHOD="classical"` | `scores_raid_classical_*` | T4×1 |

Notes:
- **A1 + A4 are the minimum** to re-decide GO/NO-GO #1 (clean signal vs baselines).
- **A6–A8 are the minimum** to re-decide GO/NO-GO #2 (the robustness headline).
- A2/A3/A5 (the big diffusion models) are what push the hard generators past 0.85
  and unlock the DTD statistic — run them once A1/A4 confirm the protocol is sane.
- For the **final** MRE run, bump `MAX_SAMPLES` (e.g. 4000–6000) so each domain has
  plenty of human+machine; 2000 is fine for the gate, thin for per-generator.
- Validate quantization once: run A2 on T4×2 fp16 vs T4×1 4-bit on ~200 passages
  and confirm the MRE ranking is stable before trusting 4-bit at scale.

After each notebook finishes, **"Save & Output"** the `results/` folder as a Kaggle
Dataset so the evaluation notebooks can attach them.

## STAGE B — Evaluation (CPU only, no GPU). Run AFTER Stage A. Sequential.

| # | Notebook | Depends on | Produces |
|---|----------|-----------|----------|
| B1 | `05_evaluation.py` | all Stage-A parquets attached | within-testbed Table 1, robustness Table 2, GO/NO-GO #1 & #2, figures |
| B2 | `08_head_to_head.py` | A1 (MRE) + A4 (baselines) | per-generator MRE-vs-FDGPT scatter, combiner |

In **B1/B2**, edit the `SCORE_DIRS` / input-path lists at the top to point at the
Kaggle Datasets you saved in Stage A (the old hard-coded `vasuaashadesai` /
`shilpavdesai` paths are examples — replace with yours).

---

## What to read in the output

- **GO/NO-GO #1** (NB05, "STEP 2"): look at **`auroc_within_testbed`**, not
  `auroc_pooled`. Best DiffuDetect method ≥0.85 ⇒ GO; 0.70–0.85 ⇒ scale up model.
- **GO/NO-GO #2** (NB05, "STEP 3"): all ΔAUROC are within-testbed. The paragraph
  attack row is the headline. Want: DiffuDetect ΔAUROC clearly smaller (more
  robust) than Fast-DetectGPT's, advantage ≥0.08.
- If clean within-testbed AUROC for FDGPT is now ~0.9 (it should be, post-fix),
  the robustness experiment finally has headroom to show a collapse-vs-graceful
  gap. That is the whole paper.

## Dependency graph

```
A1 ─┐
A2 ─┤ (MAGE MRE: smdm/llada/dream)
A3 ─┤
A4 ─┼─────────────►  B1 (05_evaluation)  ─► GO/NO-GO #1 + #2, figures
A5 ─┤ (MAGE DTD)
A6 ─┤
A7 ─┤ (RAID robustness)
A8 ─┘
A1 + A4 ───────────►  B2 (08_head_to_head) ─► MRE-vs-FDGPT scatter, combiner
```
