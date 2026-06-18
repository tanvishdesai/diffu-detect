"""
==========================================================================
  DiffuDetect — Kaggle Notebook 3: DC + DTD Scoring (Phase 2)
==========================================================================

PURPOSE:
  - Run Diffusion Curvature (DC) scoring with SMDM-1.1B
  - Run Denoising-Trajectory Dynamics (DTD) with LLaDA-8B (4-bit)
  - Build combined feature table
  - Compare per-statistic AUROCs → M2 milestone

KAGGLE SETTINGS:
  - GPU: T4 x1 (required)
  - Internet: ON
  - Accelerator: GPU T4
  - Persistence: Save outputs

NOTE: DC is SLOW (50 perturbations × 16 draws per passage).
      Use MAX_SAMPLES=500 for initial validation, scale up later.
"""

# !pip install -q torch transformers datasets accelerate bitsandbytes \
#     scikit-learn pandas pyarrow tqdm huggingface_hub sentencepiece protobuf

import os, sys, time, json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

# ─── Config ──────────────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)

# Choose which to run (DC is slow, DTD needs iterative model)
RUN_DC = True
RUN_DTD = True

# DC config
DC_MODEL_REPO = "nieshen/SMDM-1.1b"
DC_MODEL_NAME = "smdm-1.1b"
DC_MASK_RATIO = 0.30
DC_NUM_PERTURBATIONS = 30       # Reduced from 50 for speed
DC_NUM_MASK_DRAWS = 8           # Reduced from 16 for speed

# DTD config
DTD_MODEL_REPO = "GSAI-ML/LLaDA-8B-Instruct"
DTD_MODEL_NAME = "llada-8b"
DTD_QUANTIZE = 4
DTD_NUM_STEPS = 32              # Denoising steps
DTD_NUM_DRAWS = 4               # Few draws since each is multi-step
DTD_INITIAL_MASK_RATIO = 0.90

MAX_SAMPLES = 500               # Start small!
MAX_LENGTH = 512

DATA_DIR = "/kaggle/input/diffudetect-data/data"
RESULTS_DIR = "/kaggle/working/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─── Load data ───────────────────────────────────────────────────────────────

data_file = os.path.join(DATA_DIR, "mage_quick.parquet")
if not os.path.exists(data_file):
    from datasets import load_dataset
    ds = load_dataset("yaful/MAGE", trust_remote_code=True)
    split = "test" if "test" in ds else list(ds.keys())[0]
    df = ds[split].to_pandas()
    for src, dst in [("source_model", "generator"), ("category", "domain")]:
        if src in df.columns: df = df.rename(columns={src: dst})
    if "generator" not in df.columns: df["generator"] = df["label"].apply(lambda x: "machine" if x == 1 else "human")
    if "domain" not in df.columns: df["domain"] = "unknown"
    df["dataset"] = "mage"; df["attack"] = "none"
    df["id"] = [f"mage_{i}" for i in range(len(df))]
    df["label"] = df["label"].astype(int)
else:
    df = pd.read_parquet(data_file)

if MAX_SAMPLES and len(df) > MAX_SAMPLES:
    df = df.groupby("label", group_keys=False).apply(
        lambda x: x.sample(n=min(MAX_SAMPLES//2, len(x)), random_state=SEED)
    ).reset_index(drop=True)

print(f"Data: {len(df)} passages, labels={df['label'].value_counts().to_dict()}")

# ─── Helper functions ────────────────────────────────────────────────────────

def load_diffusion_model(repo, quantize_bits=None):
    """Load a diffusion model with optional quantization."""
    from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    kwargs = {"pretrained_model_name_or_path": repo, "trust_remote_code": True, "torch_dtype": torch.float16}
    if quantize_bits in (4, 8):
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=(quantize_bits==4), load_in_8bit=(quantize_bits==8),
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        kwargs["device_map"] = "auto"
    else:
        kwargs["device_map"] = {"": DEVICE}

    mdl = None
    for Cls in [AutoModelForMaskedLM, AutoModel]:
        try:
            mdl = Cls.from_pretrained(**kwargs)
            print(f"Loaded with {Cls.__name__}")
            break
        except: continue
    if mdl is None: raise RuntimeError(f"Cannot load {repo}")
    mdl.eval()
    return mdl, tok

def get_mask_token_id(model, tokenizer):
    for check in [
        lambda: tokenizer.mask_token_id if hasattr(tokenizer, 'mask_token_id') and tokenizer.mask_token_id else None,
        lambda: model.config.mask_token_id if hasattr(model.config, 'mask_token_id') else None,
        lambda: tokenizer.convert_tokens_to_ids("[MASK]") if "[MASK]" in tokenizer.get_vocab() else None,
        lambda: tokenizer.convert_tokens_to_ids("<mask>") if "<mask>" in tokenizer.get_vocab() else None,
        lambda: tokenizer.unk_token_id,
    ]:
        v = check()
        if v is not None: return v
    return len(tokenizer) - 1

def get_special_ids(tokenizer):
    ids = set()
    for attr in ['bos_token_id','eos_token_id','pad_token_id','cls_token_id','sep_token_id']:
        v = getattr(tokenizer, attr, None)
        if v is not None: ids.add(v)
    return ids

def make_mask(input_ids, mask_ratio, special_ids, pad_id):
    seq_len = input_ids.shape[1]
    eligible = torch.ones(seq_len, dtype=torch.bool, device=input_ids.device)
    if pad_id is not None: eligible &= (input_ids[0] != pad_id)
    for sid in special_ids: eligible &= (input_ids[0] != sid)
    eidx = eligible.nonzero(as_tuple=True)[0]
    n = max(1, int(len(eidx) * mask_ratio))
    perm = torch.randperm(len(eidx), device=input_ids.device)[:n]
    return eidx[perm]

def forward_get_logits(model, input_ids, attention_mask=None):
    try: out = model(input_ids=input_ids, attention_mask=attention_mask)
    except TypeError: out = model(input_ids=input_ids)
    if hasattr(out, 'logits'): return out.logits
    if isinstance(out, tuple): return out[0]
    return out

# =========================================================================
#  PART A: DIFFUSION CURVATURE (DC)
# =========================================================================

if RUN_DC:
    print("\n" + "=" * 60)
    print("DIFFUSION CURVATURE (DC) SCORING")
    print("=" * 60)

    model, tokenizer = load_diffusion_model(DC_MODEL_REPO)
    mask_id = get_mask_token_id(model, tokenizer)
    special_ids = get_special_ids(tokenizer)
    print(f"Mask token: {mask_id}, special: {special_ids}")

    @torch.no_grad()
    def compute_mre_for_ids(ids, attn, mask_ratio, n_draws):
        nlls = []
        for _ in range(n_draws):
            mpos = make_mask(ids, mask_ratio, special_ids, tokenizer.pad_token_id)
            m_ids = ids.clone(); m_ids[0, mpos] = mask_id
            logits = forward_get_logits(model, m_ids, attn)
            lp = F.log_softmax(logits, dim=-1)
            nll = -lp[0, mpos, :][torch.arange(len(mpos)), ids[0, mpos]].mean().item()
            nlls.append(nll)
        return np.mean(nlls)

    @torch.no_grad()
    def generate_perturbation(ids, attn, mask_ratio):
        mpos = make_mask(ids, mask_ratio, special_ids, tokenizer.pad_token_id)
        m_ids = ids.clone(); m_ids[0, mpos] = mask_id
        logits = forward_get_logits(model, m_ids, attn)
        probs = F.softmax(logits, dim=-1)
        perturbed = ids.clone()
        for p in mpos:
            perturbed[0, p] = torch.multinomial(probs[0, p], 1).item()
        return perturbed

    @torch.no_grad()
    def compute_dc(text):
        enc = tokenizer(text, max_length=MAX_LENGTH, truncation=True, padding="max_length", return_tensors="pt")
        ids = enc["input_ids"].to(DEVICE); attn = enc["attention_mask"].to(DEVICE)

        orig_mre = compute_mre_for_ids(ids, attn, DC_MASK_RATIO, DC_NUM_MASK_DRAWS)
        p_mres = []
        for _ in range(DC_NUM_PERTURBATIONS):
            p_ids = generate_perturbation(ids, attn, DC_MASK_RATIO)
            p_mre = compute_mre_for_ids(p_ids, attn, DC_MASK_RATIO, max(2, DC_NUM_MASK_DRAWS//4))
            p_mres.append(p_mre)

        p_mean = np.mean(p_mres); p_std = np.std(p_mres) + 1e-8
        return {
            "dc_curvature": orig_mre - p_mean,
            "dc_original_mre": orig_mre,
            "dc_perturb_mean_mre": p_mean,
            "dc_normalized": (orig_mre - p_mean) / p_std,
        }

    dc_results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="DC scoring"):
        try:
            dc_results.append(compute_dc(str(row["text"])))
        except Exception as e:
            dc_results.append({"dc_curvature": np.nan, "dc_original_mre": np.nan,
                             "dc_perturb_mean_mre": np.nan, "dc_normalized": np.nan})

    dc_df = pd.DataFrame(dc_results)
    for c in dc_df.columns: df[c] = dc_df[c].values

    # Save DC scores
    meta = ["id","text","label","generator","domain","dataset","attack"]
    save_cols = [c for c in meta if c in df.columns] + list(dc_df.columns)
    df[save_cols].to_parquet(os.path.join(RESULTS_DIR, f"scores_mage_dc_{DC_MODEL_NAME}.parquet"), index=False)

    # DC AUROC
    valid = ~np.isnan(df["dc_normalized"].values)
    if valid.sum() > 10:
        auroc = roc_auc_score(df["label"].values[valid], df["dc_normalized"].values[valid])
        print(f"\nDC Normalized AUROC: {auroc:.4f}")

    del model; torch.cuda.empty_cache()

# =========================================================================
#  PART B: DENOISING-TRAJECTORY DYNAMICS (DTD)
# =========================================================================

if RUN_DTD:
    print("\n" + "=" * 60)
    print("DENOISING-TRAJECTORY DYNAMICS (DTD) SCORING")
    print("=" * 60)

    model, tokenizer = load_diffusion_model(DTD_MODEL_REPO, DTD_QUANTIZE)
    mask_id = get_mask_token_id(model, tokenizer)
    special_ids = get_special_ids(tokenizer)
    print(f"Mask token: {mask_id}")

    @torch.no_grad()
    def compute_dtd(text):
        enc = tokenizer(text, max_length=MAX_LENGTH, truncation=True, padding="max_length", return_tensors="pt")
        ids = enc["input_ids"].to(DEVICE); attn = enc["attention_mask"].to(DEVICE)
        seq_len = ids.shape[1]

        all_features = []
        for _ in range(DTD_NUM_DRAWS):
            # Create initial heavy mask
            mpos = make_mask(ids, DTD_INITIAL_MASK_RATIO, special_ids, tokenizer.pad_token_id)
            cur = ids.clone(); cur[0, mpos] = mask_id
            masked_set = set(mpos.cpu().tolist())
            orig_masked = set(masked_set)

            step_entropies = []
            commit_times = {p: DTD_NUM_STEPS for p in orig_masked}
            committed = set()
            prev_preds = {}
            flips = {p: 0 for p in orig_masked}

            schedule = np.linspace(DTD_INITIAL_MASK_RATIO, 0.0, DTD_NUM_STEPS + 1)[:-1]

            for step_idx, target_ratio in enumerate(schedule):
                if not masked_set: break
                logits = forward_get_logits(model, cur, attn)
                probs = F.softmax(logits, dim=-1)
                entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)

                # Step entropy over masked positions
                mpos_list = list(masked_set)
                if mpos_list:
                    step_ent = entropy[0, mpos_list].mean().item()
                    step_entropies.append(step_ent)

                top1_probs, top1_ids = probs[0].max(dim=-1)

                # Track flips and commits
                for p in list(masked_set):
                    pred = top1_ids[p].item()
                    if step_idx > 0 and p in prev_preds and pred != prev_preds[p]:
                        flips[p] += 1
                    prev_preds[p] = pred
                    if top1_probs[p].item() > 0.9 and p not in committed:
                        commit_times[p] = step_idx
                        committed.add(p)

                # Unmask most confident tokens
                n_target = max(0, int(target_ratio * len(orig_masked)))
                n_unmask = max(0, len(masked_set) - n_target)
                if n_unmask > 0 and mpos_list:
                    ents = [(entropy[0, p].item(), p) for p in mpos_list]
                    ents.sort()
                    for _, p in ents[:n_unmask]:
                        cur[0, p] = top1_ids[p]
                        masked_set.discard(p)

            # Extract features
            feat = {}
            if len(step_entropies) >= 2:
                feat["dtd_entropy_auc"] = float(np.trapz(step_entropies))
            else:
                feat["dtd_entropy_auc"] = step_entropies[0] if step_entropies else 0.0

            ct_vals = [commit_times[p] / max(len(schedule), 1) for p in orig_masked]
            feat["dtd_mean_commit_time"] = float(np.mean(ct_vals)) if ct_vals else 1.0

            if len(step_entropies) >= 3:
                feat["dtd_trajectory_curvature"] = float(np.mean(np.abs(np.diff(step_entropies, n=2))))
            else:
                feat["dtd_trajectory_curvature"] = 0.0

            flip_vals = [flips[p] for p in orig_masked]
            feat["dtd_mean_flips"] = float(np.mean(flip_vals)) if flip_vals else 0.0
            feat["dtd_final_entropy"] = step_entropies[-1] if step_entropies else 0.0
            feat["dtd_entropy_drop"] = (step_entropies[0] - step_entropies[-1]) if len(step_entropies) >= 2 else 0.0

            all_features.append(feat)

        # Average over draws
        avg = {}
        for k in all_features[0]:
            avg[k] = float(np.mean([f[k] for f in all_features]))
        return avg

    dtd_results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="DTD scoring"):
        try:
            dtd_results.append(compute_dtd(str(row["text"])))
        except Exception as e:
            dtd_results.append({k: np.nan for k in ["dtd_entropy_auc","dtd_mean_commit_time",
                               "dtd_trajectory_curvature","dtd_mean_flips","dtd_final_entropy","dtd_entropy_drop"]})

    dtd_df = pd.DataFrame(dtd_results)
    for c in dtd_df.columns: df[c] = dtd_df[c].values

    meta = ["id","text","label","generator","domain","dataset","attack"]
    save_cols = [c for c in meta if c in df.columns] + list(dtd_df.columns)
    df[save_cols].to_parquet(os.path.join(RESULTS_DIR, f"scores_mage_dtd_{DTD_MODEL_NAME}.parquet"), index=False)

    # DTD AUROCs
    for col in dtd_df.columns:
        valid = ~np.isnan(df[col].values)
        if valid.sum() < 10: continue
        try:
            auroc = roc_auc_score(df["label"].values[valid], -df[col].values[valid])
            print(f"  {col}: AUROC={auroc:.4f}")
        except: pass

    del model; torch.cuda.empty_cache()

# ─── Summary ─────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("NOTEBOOK 3 COMPLETE — DC + DTD Scoring Done")
print("=" * 60)
print(f"Results saved to: {RESULTS_DIR}/")
print("Next: Run Notebook 4 (baselines) then Notebook 5 (evaluation)")
