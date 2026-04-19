"""
metrics.py
----------
Core evaluation metrics for POPE-style binary VQA benchmarks.

Metrics
-------
- accuracy, precision, recall, f1
- yes_rate          : fraction of "yes" predictions
- yes_bias          : yes_rate - 0.5  (signed; 0 = unbiased)
- confusion         : (TP, FP, TN, FN) dict
- precision_recall  : (precision, recall) tuple
- cv_across_models  : coefficient of variation across model scores
                      (Module 1 saturation diagnostic)
- per_category_metrics : break metrics down by category tag
- compute_all          : all scalar metrics in one call

Label convention: "yes"=1 / "no"=0  (case-insensitive strings OR ints)
"""

from __future__ import annotations

import numpy as np
from typing import Sequence, Union

Label = Union[str, int]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_binary(labels: Sequence[Label]) -> np.ndarray:
    """Convert 'yes'/'no' strings or 0/1 ints to a numpy int8 array."""
    arr = []
    for v in labels:
        if isinstance(v, str):
            v_lower = v.strip().lower()
            if v_lower == "yes":
                arr.append(1)
            elif v_lower == "no":
                arr.append(0)
            else:
                raise ValueError(f"Unknown label string: '{v}'. Expected 'yes' or 'no'.")
        else:
            arr.append(int(v))
    return np.array(arr, dtype=np.int8)


def _confusion_arrays(p: np.ndarray, l: np.ndarray):
    """Return (TP, FP, TN, FN)."""
    tp = int(np.sum((p == 1) & (l == 1)))
    fp = int(np.sum((p == 1) & (l == 0)))
    tn = int(np.sum((p == 0) & (l == 0)))
    fn = int(np.sum((p == 0) & (l == 1)))
    return tp, fp, tn, fn


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def accuracy(preds: Sequence[Label], labels: Sequence[Label]) -> float:
    """Fraction of correct predictions."""
    p, l = _to_binary(preds), _to_binary(labels)
    return float(np.mean(p == l))


def confusion(
    preds: Sequence[Label],
    labels: Sequence[Label],
) -> dict[str, int]:
    """Return TP / FP / TN / FN counts (positive class = 'yes')."""
    p, l = _to_binary(preds), _to_binary(labels)
    tp, fp, tn, fn = _confusion_arrays(p, l)
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn}


def precision(preds: Sequence[Label], labels: Sequence[Label]) -> float:
    """TP / (TP + FP). Returns 0.0 if no positive predictions."""
    p, l = _to_binary(preds), _to_binary(labels)
    tp, fp, _, _ = _confusion_arrays(p, l)
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall(preds: Sequence[Label], labels: Sequence[Label]) -> float:
    """TP / (TP + FN). Returns 0.0 if no positive ground-truth."""
    p, l = _to_binary(preds), _to_binary(labels)
    tp, _, _, fn = _confusion_arrays(p, l)
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def precision_recall(
    preds: Sequence[Label],
    labels: Sequence[Label],
) -> tuple[float, float]:
    """Return (precision, recall) as a tuple."""
    return precision(preds, labels), recall(preds, labels)


def f1(preds: Sequence[Label], labels: Sequence[Label]) -> float:
    """Harmonic mean of precision and recall."""
    prec = precision(preds, labels)
    rec  = recall(preds, labels)
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


def yes_rate(preds: Sequence[Label]) -> float:
    """Fraction of 'yes' predictions. On balanced data, ideal ≈ 0.5."""
    return float(np.mean(_to_binary(preds)))


def yes_bias(preds: Sequence[Label]) -> float:
    """
    Signed yes-bias = yes_rate - 0.5.
      > 0  → over-predicts 'yes'
      < 0  → over-predicts 'no'
      = 0  → unbiased
    """
    return yes_rate(preds) - 0.5


# ---------------------------------------------------------------------------
# Per-category breakdown
# ---------------------------------------------------------------------------

def per_category_metrics(
    preds:      Sequence[Label],
    labels:     Sequence[Label],
    categories: Sequence[str],
) -> dict[str, dict]:
    """
    Compute per-category metrics.

    Parameters
    ----------
    categories : per-sample string tag, e.g.
                 'random' | 'popular' | 'adversarial'   (POPE splits)
                 'existence' | 'attribute' | 'relation'  (X-POPE types)

    Returns
    -------
    {category: {"n", "accuracy", "f1", "precision", "recall",
                "yes_rate", "yes_bias"}}
    """
    preds_l  = list(preds)
    labels_l = list(labels)
    cats_l   = list(categories)
    assert len(preds_l) == len(labels_l) == len(cats_l), \
        "preds, labels, categories must have equal length"

    result: dict[str, dict] = {}
    for cat in sorted(set(cats_l)):
        idx   = [i for i, c in enumerate(cats_l) if c == cat]
        p_cat = [preds_l[i]  for i in idx]
        l_cat = [labels_l[i] for i in idx]
        result[cat] = {
            "n":         len(idx),
            "accuracy":  accuracy(p_cat, l_cat),
            "f1":        f1(p_cat, l_cat),
            "precision": precision(p_cat, l_cat),
            "recall":    recall(p_cat, l_cat),
            "yes_rate":  yes_rate(p_cat),
            "yes_bias":  yes_bias(p_cat),
        }
    return result


# ---------------------------------------------------------------------------
# Saturation diagnostics (Module 1)
# ---------------------------------------------------------------------------

def coefficient_of_variation(scores: Sequence[float]) -> float:
    """
    CV = std / mean across N model scores.

    Low CV (< ~0.02) on POPE → benchmark is saturated; models can no longer
    be meaningfully ranked.
    """
    arr  = np.array(scores, dtype=float)
    mean = arr.mean()
    return float(arr.std(ddof=0) / mean) if mean != 0 else 0.0

# Alias used in Module 1 analysis scripts
cv_across_models = coefficient_of_variation


def saturation_report(
    model_scores: dict[str, float],
    threshold_cv: float = 0.02,
) -> dict:
    """
    Given {model_name: metric_score}, return a saturation summary.

    Returns
    -------
    {
      "scores":    {model: score},
      "mean":      float,
      "std":       float,
      "cv":        float,
      "max_gap":   float,   # best_score - worst_score
      "saturated": bool,    # True if cv < threshold_cv
    }

    Example
    -------
    >>> saturation_report({
    ...     "Qwen2.5-VL-7B":   0.891,
    ...     "InternVL2.5-8B":  0.883,
    ...     "LLaVA-OV-7B":     0.879,
    ...     "Llama-3.2-11B":   0.861,
    ...     "PaliGemma2-3B":   0.847,
    ...     "Idefics3-8B":     0.856,
    ... })
    """
    scores = list(model_scores.values())
    arr    = np.array(scores, dtype=float)
    cv     = coefficient_of_variation(scores)
    return {
        "scores":    model_scores,
        "mean":      float(arr.mean()),
        "std":       float(arr.std(ddof=0)),
        "cv":        cv,
        "max_gap":   float(arr.max() - arr.min()),
        "saturated": cv < threshold_cv,
    }


# ---------------------------------------------------------------------------
# Full metric bundle for one (model, split) pair
# ---------------------------------------------------------------------------

def compute_all(
    preds:      Sequence[Label],
    labels:     Sequence[Label],
    categories: Sequence[str] | None = None,
) -> dict:
    """
    Return a complete metric dict for one model on one benchmark split.

    Parameters
    ----------
    preds, labels : predictions and ground-truth
    categories    : optional per-sample category for breakdown table

    Returns
    -------
    {
      "n", "accuracy", "precision", "recall", "f1",
      "yes_rate", "yes_bias",
      "by_category": {...}   # only if categories is not None
    }
    """
    out = {
        "n":         len(list(preds)),
        "accuracy":  accuracy(preds, labels),
        "precision": precision(preds, labels),
        "recall":    recall(preds, labels),
        "f1":        f1(preds, labels),
        "yes_rate":  yes_rate(preds),
        "yes_bias":  yes_bias(preds),
    }
    if categories is not None:
        out["by_category"] = per_category_metrics(preds, labels, categories)
    return out
