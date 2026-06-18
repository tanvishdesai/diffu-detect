# DiffuDetect: Diffusion Language Models as Zero-Shot Detectors of AI-Generated Text

> **AAAI-27 submission.** Frozen diffusion LMs detect AI text via reconstruction error, curvature, and denoising-trajectory dynamics — without any training. The headline result: robustness to paraphrase attacks that collapse AR-based detectors.

---

## Quick Overview

| Component | What it does |
|---|---|
| **MRE** (Masked Reconstruction Error) | Mask tokens → let diffusion LM reconstruct → measure NLL. AI text is easier to reconstruct. |
| **DC** (Diffusion Curvature) | Perturbation-based curvature of the reconstruction landscape. Analogue of Fast-DetectGPT. |
| **DTD** (Denoising-Trajectory Dynamics) | Track the full iterative denoising process: entropy trajectory, commit times, token flips. |
| **Baselines** | Fast-DetectGPT, DetectGPT, Binoculars, Classical (GLTR-style log-lik/rank/entropy). |

---

## Repository Structure

```
diffudetect/
├── PLANNING-diffudetect.md     # Vision, architecture, constraints
├── TASK-diffudetect.md          # Working task tracker with milestones
├── README.md                    # This file
├── requirements.txt             # Python dependencies
│
├── src/                         # Core library (importable from notebooks)
│   ├── config.py                # All models, datasets, hyperparameters
│   ├── utils.py                 # Model loading, tokenization, masking, I/O
│   │
│   ├── data/                    # Dataset loaders
│   │   ├── mage_loader.py       # MAGE dataset (yaful/MAGE)
│   │   └── raid_loader.py       # RAID dataset (liamdugan/raid)
│   │
│   ├── scorers/                 # DiffuDetect statistics
│   │   ├── mre.py               # Masked Reconstruction Error
│   │   ├── dc.py                # Diffusion Curvature
│   │   └── dtd.py               # Denoising-Trajectory Dynamics
│   │
│   ├── baselines/               # Baseline detectors
│   │   ├── fast_detectgpt.py    # Fast-DetectGPT (primary baseline)
│   │   ├── detectgpt.py         # DetectGPT (weak OOD reference)
│   │   ├── binoculars.py        # Binoculars (observer/performer pair)
│   │   └── classical.py         # GLTR-style (log-lik, rank, entropy)
│   │
│   ├── eval/                    # Evaluation & analysis
│   │   ├── metrics.py           # AUROC, TPR@FPR, logistic combiner
│   │   ├── robustness.py        # ΔAUROC, GO/NO-GO decision
│   │   └── aggregator.py        # Results aggregation, tables, figures
│   │
│   ├── run_scoring.py           # CLI: run DiffuDetect scoring
│   ├── run_baselines.py         # CLI: run baseline scoring
│   └── run_evaluation.py        # CLI: aggregate + evaluate + plot
│
└── notebooks/                   # Kaggle notebooks (run on GPU)
    ├── 01_setup_data.py         # Data download & preprocessing
    ├── 02_mre_scoring.py        # Phase 1: MRE (kill-early gate)
    ├── 03_dc_dtd_scoring.py     # Phase 2: DC + DTD
    ├── 04_baseline_scoring.py   # Baseline methods
    ├── 05_evaluation.py         # Aggregate, metrics, figures
    └── 06_raid_robustness.py    # Phase 3: RAID robustness (decisive)
```

---

## How to Run on Kaggle — Step by Step

### Prerequisites

- 1-2 Kaggle accounts with **GPU T4** access (30h/week free per account)
- Internet must be **ON** for model/dataset downloads
- ~6-10 hours total GPU time for a full run

### Step 0: Upload the Source Code

1. Go to [kaggle.com/datasets/new](https://www.kaggle.com/datasets/new)
2. Upload this entire `diffudetect/` folder as a Kaggle Dataset
3. Name it `diffudetect-code`
4. This makes the `src/` library available to all notebooks

### Step 1: Data Preparation (Notebook 01) — ⏱️ ~10 min, NO GPU needed

```
Kaggle Settings:
  - Accelerator: None
  - Internet: ON
  - Input datasets: None (downloads from HuggingFace)
```

1. Create a new Kaggle Notebook
2. Copy the contents of `notebooks/01_setup_data.py` into cells
3. Run all cells
4. **Save the output as a Kaggle Dataset** named `diffudetect-data`

This downloads MAGE and RAID, standardizes columns, and saves as Parquet.

### Step 2: MRE Scoring (Notebook 02) — ⏱️ ~1-3 hours, GPU T4 required

```
Kaggle Settings:
  - Accelerator: GPU T4 x1
  - Internet: ON
  - Input datasets: diffudetect-data (from Step 1)
```

1. Create a new GPU notebook
2. Copy `notebooks/02_mre_scoring.py`
3. **Important configs to set:**
   - `MAX_SAMPLES = 2000` (start here; scale to 5000+ later)
   - `MODEL_HF_REPO = "nieshen/SMDM-1.1b"`
   - `NUM_MASK_DRAWS = 16`
4. Run all cells
5. **Check GO/NO-GO #1**: If best AUROC ≥ 0.85, proceed. If < 0.70, the premise is weak.
6. **Save output as Dataset** named `diffudetect-mre-scores`

### Step 3: DC + DTD Scoring (Notebook 03) — ⏱️ ~2-4 hours, GPU T4 required

```
Kaggle Settings:
  - Accelerator: GPU T4 x1
  - Internet: ON
  - Input datasets: diffudetect-data
```

1. Create a new GPU notebook
2. Copy `notebooks/03_dc_dtd_scoring.py`
3. **Important:** DC is slow. Start with `MAX_SAMPLES = 500`
4. **Configs:**
   - `RUN_DC = True`
   - `RUN_DTD = True` (set False if LLaDA-8B doesn't fit)
   - `DC_NUM_PERTURBATIONS = 30` (reduce for speed)
5. Run all cells
6. **Save output as Dataset** named `diffudetect-dc-dtd-scores`

> **Parallelism tip:** Run DC and DTD in separate notebooks on different accounts to save time.

### Step 4: Baseline Scoring (Notebook 04) — ⏱️ ~1-2 hours, GPU T4 required

```
Kaggle Settings:
  - Accelerator: GPU T4 x1
  - Internet: ON
  - Input datasets: diffudetect-data
```

1. Copy `notebooks/04_baseline_scoring.py`
2. **Configs:**
   - `RUN_CLASSICAL = True` (fast, always run)
   - `RUN_FAST_DETECTGPT = True` (primary baseline)
   - `RUN_DETECTGPT = False` (optional, slow)
   - `RUN_BINOCULARS = False` (optional, needs 2 models)
3. Run all cells
4. **Save output as Dataset** named `diffudetect-baseline-scores`

### Step 5: Evaluation & Figures (Notebook 05) — ⏱️ ~5 min, NO GPU needed

```
Kaggle Settings:
  - Accelerator: None
  - Internet: OFF is fine
  - Input datasets:
    - diffudetect-mre-scores
    - diffudetect-dc-dtd-scores
    - diffudetect-baseline-scores
```

1. Copy `notebooks/05_evaluation.py`
2. **Update SCORE_DIRS** to match your dataset names:
   ```python
   SCORE_DIRS = [
       "/kaggle/input/diffudetect-mre-scores/results",
       "/kaggle/input/diffudetect-dc-dtd-scores/results",
       "/kaggle/input/diffudetect-baseline-scores/results",
   ]
   ```
3. Run all cells → generates Table 1 (clean AUROC), Table 2 (robustness), figures

### Step 6: RAID Robustness — THE DECISIVE EXPERIMENT (Notebook 06) — ⏱️ ~2-4 hours

```
Kaggle Settings:
  - Accelerator: GPU T4 x1
  - Internet: ON
  - Input datasets: diffudetect-data
```

1. Copy `notebooks/06_raid_robustness.py`
2. **Run THREE times** with different `RUN_METHOD`:
   - Run 1: `RUN_METHOD = "mre"` → saves `scores_raid_mre_smdm-1.1b.parquet`
   - Run 2: `RUN_METHOD = "classical"` → saves `scores_raid_classical_gpt-neo-2.7b.parquet`
   - Run 3: `RUN_METHOD = "fast_detectgpt"` → saves `scores_raid_fast_detectgpt_gpt-neo-2.7b.parquet`
3. Save all outputs as `diffudetect-raid-scores`
4. Re-run Notebook 05 with RAID scores added → **GO/NO-GO #2 decision**

> **This is THE experiment.** If DiffuDetect's ΔAUROC is 8+ points better than Fast-DetectGPT's, you have a paper.

---

## Parallelization Strategy (Multiple Kaggle Accounts)

Each account gets 30h GPU/week. Here's how to parallelize:

| Account | Notebook | Model | Dataset | Time |
|---|---|---|---|---|
| Account 1 | 02_mre | SMDM-1.1B | MAGE | ~2h |
| Account 1 | 06_raid (mre) | SMDM-1.1B | RAID | ~3h |
| Account 2 | 03_dc | SMDM-1.1B | MAGE | ~3h |
| Account 2 | 04_baselines | GPT-Neo-2.7B | MAGE | ~2h |
| Account 3 | 03_dtd | LLaDA-8B | MAGE | ~4h |
| Account 3 | 06_raid (fdgpt) | GPT-Neo-2.7B | RAID | ~3h |

All results → Parquet on Kaggle Datasets → aggregate in Notebook 05 (CPU).

---

## Execution Sequence Summary

```
┌─────────────────────────────┐
│  01_setup_data.py (CPU)     │  Download MAGE + RAID → Parquet
└──────────┬──────────────────┘
           │
     ┌─────┴─────┬──────────────┐
     ▼           ▼              ▼
┌─────────┐ ┌─────────┐  ┌──────────┐
│ 02_mre  │ │ 03_dc   │  │ 04_base  │   All on GPU T4
│ (T4)    │ │ _dtd    │  │ lines    │   Can run in parallel
│         │ │ (T4)    │  │ (T4)     │
└────┬────┘ └────┬────┘  └────┬─────┘
     │           │            │
     └─────┬─────┴────────────┘
           ▼
┌──────────────────────────────┐
│  05_evaluation.py (CPU)      │  Aggregate → Tables → Figures
│  → GO/NO-GO #1 check        │  → Check: clean AUROC ≥ 0.85?
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│  06_raid_robustness.py (T4)  │  Score RAID (clean + attacks)
│  × 3 runs (mre, fdgpt, cls) │  with DiffuDetect + baselines
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│  05_evaluation.py (rerun)    │  Add RAID scores →
│  → GO/NO-GO #2 (DECISIVE)   │  ΔAUROC table + figures
└──────────────────────────────┘
```

---

## Key Hyperparameters to Tune

| Parameter | Default | Notes |
|---|---|---|
| `MASK_RATIOS` | [0.15, 0.30, 0.50] | Sweep for MRE; 0.30 is usually best |
| `NUM_MASK_DRAWS` (K) | 16 | Trade stability vs speed; 8 is often enough |
| `DC_NUM_PERTURBATIONS` | 30-50 | More = more stable curvature estimate |
| `DTD_NUM_STEPS` | 32-64 | Denoising schedule length |
| `MAX_LENGTH` | 512 | Token length; increase for longer passages |
| `MAX_SAMPLES` | 2000-5000 | Start small, scale up once signal confirmed |

---

## Models Used

| Model | Role | Size | Quantization | Fits T4? |
|---|---|---|---|---|
| SMDM-1.1B | Primary diffusion scorer | 1.1B | None | ✅ Yes |
| MDLM-110M | Fast ablation scorer | 110M | None | ✅ Yes |
| LLaDA-8B-Instruct | Iterative diffusion (DTD) | 8B | 4-bit | ✅ ~6GB |
| Dream-7B | Second trajectory model | 7B | 4-bit | ✅ ~5GB |
| GPT-Neo-2.7B | AR baseline scorer | 2.7B | None | ✅ Yes |
| Falcon-7B pair | Binoculars | 7B×2 | 4-bit each | ⚠️ Tight |

---

## License

Research code for academic purposes. Will be released with the paper.
