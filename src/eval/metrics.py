"""
DiffuDetect — Evaluation Metrics

All detection metrics from the planning doc:
  - AUROC (primary)
  - TPR@1%FPR and TPR@5%FPR (operating points)
  - Accuracy / F1 at optimal threshold
  - Cost metrics (latency reporting)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)


def compute_detection_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    score_direction: str = "higher_is_machine",
) -> Dict[str, float]:
    """
    Compute all detection metrics for a single method.

    Args:
        labels: binary labels (0=human, 1=machine)
        scores: scalar detection scores
        score_direction: "higher_is_machine" or "lower_is_machine"
            MRE: lower_is_machine (AI text is easier to reconstruct)
            DC/Fast-DetectGPT: higher_is_machine
            Binoculars: lower_is_machine (lower ratio = more machine-like)
            Classical LL: higher_is_machine
            Classical entropy: lower_is_machine

    Returns dict with AUROC, TPR@1%FPR, TPR@5%FPR, accuracy, F1, etc.
    """
    labels = np.asarray(labels).ravel()
    scores = np.asarray(scores).ravel()

    # Remove NaN/inf
    valid = np.isfinite(scores)
    if valid.sum() < 10:
        return _empty_metrics()

    labels = labels[valid]
    scores = scores[valid]

    # Flip scores if lower = machine (so higher always = machine for sklearn)
    if score_direction == "lower_is_machine":
        scores = -scores
    elif score_direction == "auto":
        # Auto-detect: try both directions, pick whichever gives AUROC > 0.5
        try:
            auroc_pos = roc_auc_score(labels, scores)
            auroc_neg = roc_auc_score(labels, -scores)
            if auroc_neg > auroc_pos:
                scores = -scores
        except ValueError:
            pass

    results = {}

    # AUROC
    try:
        results["auroc"] = roc_auc_score(labels, scores)
    except ValueError:
        results["auroc"] = 0.5

    # TPR at specific FPR thresholds
    try:
        fpr, tpr, thresholds = roc_curve(labels, scores)

        # TPR@1%FPR
        idx_1 = np.searchsorted(fpr, 0.01, side="right") - 1
        idx_1 = max(0, min(idx_1, len(tpr) - 1))
        results["tpr_at_1fpr"] = tpr[idx_1]

        # TPR@5%FPR
        idx_5 = np.searchsorted(fpr, 0.05, side="right") - 1
        idx_5 = max(0, min(idx_5, len(tpr) - 1))
        results["tpr_at_5fpr"] = tpr[idx_5]

        # TPR@10%FPR
        idx_10 = np.searchsorted(fpr, 0.10, side="right") - 1
        idx_10 = max(0, min(idx_10, len(tpr) - 1))
        results["tpr_at_10fpr"] = tpr[idx_10]

        # Optimal threshold (Youden's J)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0

        # Accuracy and F1 at optimal threshold
        preds = (scores >= best_threshold).astype(int)
        results["accuracy"] = accuracy_score(labels, preds)
        results["f1"] = f1_score(labels, preds, zero_division=0)
        results["precision"] = precision_score(labels, preds, zero_division=0)
        results["recall"] = recall_score(labels, preds, zero_division=0)
        results["best_threshold"] = float(best_threshold)

    except Exception as e:
        print(f"[metrics] Warning computing ROC metrics: {e}")
        results.update({
            "tpr_at_1fpr": 0.0,
            "tpr_at_5fpr": 0.0,
            "tpr_at_10fpr": 0.0,
            "accuracy": 0.5,
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "best_threshold": 0.0,
        })

    results["n_samples"] = len(labels)
    results["n_human"] = int((labels == 0).sum())
    results["n_machine"] = int((labels == 1).sum())

    return results


def _empty_metrics() -> Dict[str, float]:
    """Return empty/default metrics for degenerate cases."""
    return {
        "auroc": 0.5,
        "tpr_at_1fpr": 0.0,
        "tpr_at_5fpr": 0.0,
        "tpr_at_10fpr": 0.0,
        "accuracy": 0.5,
        "f1": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "best_threshold": 0.0,
        "n_samples": 0,
        "n_human": 0,
        "n_machine": 0,
    }


# Score direction for each method (needed for AUROC sign)
SCORE_DIRECTIONS = {
    # DiffuDetect
    "mre_mean": "lower_is_machine",
    "mre_r0.15": "lower_is_machine",
    "mre_r0.30": "lower_is_machine",
    "mre_r0.50": "lower_is_machine",
    "dc_curvature": "higher_is_machine",
    "dc_normalized": "higher_is_machine",
    "dtd_entropy_auc": "lower_is_machine",
    "dtd_mean_commit_time": "lower_is_machine",
    "dtd_trajectory_curvature": "lower_is_machine",
    "dtd_mean_flips": "lower_is_machine",
    # Baselines
    "fdgpt_curvature": "higher_is_machine",
    "dgpt_curvature": "higher_is_machine",
    "bino_score": "lower_is_machine",
    "cls_log_likelihood": "higher_is_machine",
    "cls_mean_rank": "lower_is_machine",
    "cls_mean_log_rank": "lower_is_machine",
    "cls_mean_entropy": "lower_is_machine",
    "cls_perplexity": "lower_is_machine",
    # Combiner
    "combined_logistic": "higher_is_machine",
}


def compute_all_metrics(
    df: pd.DataFrame,
    score_columns: List[str],
    label_col: str = "label",
    group_by: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compute detection metrics for multiple score columns, optionally grouped.

    Args:
        df: DataFrame with labels and score columns
        score_columns: list of score column names to evaluate
        label_col: column containing binary labels
        group_by: optional columns to group by (e.g., ["dataset", "generator"])

    Returns:
        DataFrame with metrics for each (method × group) combination.
    """
    rows = []

    if group_by:
        groups = df.groupby(group_by)
    else:
        groups = [("all", df)]

    for group_key, group_df in groups:
        if isinstance(group_key, str):
            group_key = (group_key,)

        labels = group_df[label_col].values

        for score_col in score_columns:
            if score_col not in group_df.columns:
                continue

            scores = group_df[score_col].values
            direction = SCORE_DIRECTIONS.get(score_col, "higher_is_machine")

            metrics = compute_detection_metrics(labels, scores, direction)
            metrics["method"] = score_col

            if group_by:
                for i, g_col in enumerate(group_by):
                    metrics[g_col] = group_key[i] if len(group_key) > 1 else group_key[0]

            rows.append(metrics)

    return pd.DataFrame(rows)


def compute_within_testbed_auroc(
    df: pd.DataFrame,
    score_col: str,
    testbed_col: str = "domain",
    label_col: str = "label",
    min_per_class: int = 20,
) -> Dict[str, float]:
    """
    Mean within-testbed AUROC — the PRIMARY clean-text metric.

    Why this exists: a single AUROC over the fully-pooled set mixes testbeds
    (domains) whose score scales differ, so one global threshold can't separate
    them and the number collapses (~0.60) even when per-testbed separation is
    excellent (0.9+). The MAGE / Fast-DetectGPT protocol evaluates WITHIN each
    testbed and averages. Orientation is fixed globally (via SCORE_DIRECTIONS),
    not flipped per testbed, to avoid optimistic bias.

    Returns dict: {auroc_mean, auroc_weighted, n_testbeds}.
    """
    s = pd.to_numeric(df[score_col], errors="coerce").to_numpy(dtype=float)
    direction = SCORE_DIRECTIONS.get(score_col, "higher_is_machine")
    if direction == "lower_is_machine":
        s = -s
    elif direction not in ("higher_is_machine",):
        valid = np.isfinite(s)
        if valid.sum() >= 10:
            try:
                if roc_auc_score(df[label_col].to_numpy()[valid], s[valid]) < 0.5:
                    s = -s
            except ValueError:
                pass

    labels = df[label_col].to_numpy()
    testbeds = df[testbed_col].to_numpy()
    aurocs, weights = [], []
    for tb in pd.unique(testbeds):
        m = testbeds == tb
        y = labels[m]
        sv = s[m]
        v = np.isfinite(sv)
        y, sv = y[v], sv[v]
        if (y == 0).sum() < min_per_class or (y == 1).sum() < min_per_class:
            continue
        try:
            aurocs.append(roc_auc_score(y, sv))
            weights.append(len(y))
        except ValueError:
            continue
    if not aurocs:
        return {"auroc_mean": float("nan"), "auroc_weighted": float("nan"), "n_testbeds": 0}
    aurocs = np.asarray(aurocs)
    weights = np.asarray(weights)
    return {
        "auroc_mean": float(aurocs.mean()),
        "auroc_weighted": float(np.average(aurocs, weights=weights)),
        "n_testbeds": int(len(aurocs)),
    }


def compute_combined_score(
    df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str = "label",
    calibration_df: Optional[pd.DataFrame] = None,
) -> np.ndarray:
    """
    Fit a logistic regression combiner on calibration data and predict.

    This is the "essentially training-free" 3-parameter head from the planning doc.

    Args:
        df: DataFrame to score
        feature_cols: which score columns to combine
        label_col: label column
        calibration_df: separate calibration data (if None, fit on df itself)

    Returns:
        Combined score array (probability of machine-generated)
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    # Use calibration data if provided, else use df itself
    fit_df = calibration_df if calibration_df is not None else df

    # Prepare features (handle missing by filling median)
    def prepare_features(data, cols):
        X = data[cols].copy()
        for col in cols:
            if col in X.columns:
                X[col] = X[col].fillna(X[col].median())
            else:
                X[col] = 0.0
        return X.values

    X_fit = prepare_features(fit_df, feature_cols)
    y_fit = fit_df[label_col].values

    X_score = prepare_features(df, feature_cols)

    # Standardize
    scaler = StandardScaler()
    X_fit_scaled = scaler.fit_transform(X_fit)
    X_score_scaled = scaler.transform(X_score)

    # Fit logistic regression
    clf = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight="balanced",
    )
    clf.fit(X_fit_scaled, y_fit)

    # Predict probability of machine class
    probs = clf.predict_proba(X_score_scaled)[:, 1]

    print(f"[metrics] Logistic combiner: {len(feature_cols)} features, "
          f"train accuracy={clf.score(X_fit_scaled, y_fit):.3f}")

    return probs
