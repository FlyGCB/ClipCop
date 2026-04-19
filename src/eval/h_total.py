"""
h_total.py
----------
Unified Hallucination Score  H_total

Motivation
----------
POPE only measures existence hallucination.  X-POPE adds attribute and
relation dimensions.  We need a single number that:

  1. Aggregates all three hallucination types.
  2. Lets researchers weight them differently (default: equal weights).
  3. Is interpretable: higher = less hallucination = better model.
  4. Degrades gracefully when a subset is missing
     (e.g. running only on POPE → only existence score available).

Formula
-------
H_total is a weighted harmonic mean of per-dimension F1 scores:

    H_total = (Σ w_i) / (Σ w_i / F1_i)

where i ∈ {existence, attribute, relation} and w_i are user-supplied
weights (default 1/3 each, i.e. equal weighting).

We use the harmonic mean (not arithmetic) because:
  - It penalises weak dimensions heavily — a model that is great at
    existence but terrible at relations should not score well overall.
  - It mirrors the logic of the standard F1 formula itself.

If a dimension's data is unavailable, it is excluded from the mean
and the remaining weights are renormalised automatically.

Usage
-----
    from eval.h_total import compute_h_total

    # Full X-POPE run
    score = compute_h_total(
        existence_f1 = 0.912,
        attribute_f1 = 0.743,
        relation_f1  = 0.681,
    )

    # POPE-only run (attribute / relation not available)
    score = compute_h_total(existence_f1=0.912)

    # Custom weights emphasising relation
    score = compute_h_total(
        existence_f1 = 0.912,
        attribute_f1 = 0.743,
        relation_f1  = 0.681,
        weights = {"existence": 1, "attribute": 1, "relation": 2},
    )

    # Full pipeline: raw predictions → H_total in one call
    score = compute_h_total_from_predictions(
        existence_preds=..., existence_gts=...,
        attribute_preds=..., attribute_gts=...,
        relation_preds=...,  relation_gts=...,
    )
"""

from __future__ import annotations

from typing import Optional

from .metrics import f1 as compute_f1, per_category_metrics, compute_all


# ---------------------------------------------------------------------------
# Default weights
# ---------------------------------------------------------------------------
DEFAULT_WEIGHTS: dict[str, float] = {
    "existence": 1.0,
    "attribute": 1.0,
    "relation":  1.0,
}

DIMENSIONS = ("existence", "attribute", "relation")


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def compute_h_total(
    existence_f1: Optional[float] = None,
    attribute_f1: Optional[float] = None,
    relation_f1:  Optional[float] = None,
    weights: Optional[dict[str, float]] = None,
) -> dict[str, float]:
    """
    Compute H_total from pre-computed per-dimension F1 scores.

    Parameters
    ----------
    existence_f1 : F1 on existence questions  (None → dimension excluded)
    attribute_f1 : F1 on attribute questions  (None → dimension excluded)
    relation_f1  : F1 on relation  questions  (None → dimension excluded)
    weights      : dict with keys "existence", "attribute", "relation"
                   (missing keys default to DEFAULT_WEIGHTS values)

    Returns
    -------
    {
        "h_total"       : float,   # weighted harmonic mean of available F1s
        "existence_f1"  : float | None,
        "attribute_f1"  : float | None,
        "relation_f1"   : float | None,
        "weights_used"  : dict,    # effective (renormalised) weights
        "n_dimensions"  : int,     # how many dimensions were available
    }
    """
    # Resolve weights
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        for k, v in weights.items():
            if k not in DIMENSIONS:
                raise ValueError(f"Unknown dimension '{k}'. Choose from {DIMENSIONS}.")
            w[k] = float(v)

    # Collect available (weight, f1) pairs
    dim_values = {
        "existence": existence_f1,
        "attribute": attribute_f1,
        "relation":  relation_f1,
    }

    available = {
        dim: f1_val
        for dim, f1_val in dim_values.items()
        if f1_val is not None
    }

    if not available:
        raise ValueError(
            "At least one of existence_f1 / attribute_f1 / relation_f1 must be provided."
        )

    # Validate F1 range
    for dim, val in available.items():
        if not (0.0 <= val <= 1.0):
            raise ValueError(f"{dim}_f1 must be in [0, 1], got {val}")

    # Renormalise weights to only available dimensions
    total_w = sum(w[dim] for dim in available)
    effective_weights = {dim: w[dim] / total_w for dim in available}

    # Weighted harmonic mean
    # H = 1 / Σ (w_i / F1_i)   [weights already normalised to sum=1]
    denom = sum(effective_weights[dim] / val for dim, val in available.items() if val > 0)

    # Edge case: all F1 scores are 0
    if denom == 0:
        h_total = 0.0
    else:
        h_total = 1.0 / denom

    return {
        "h_total":      round(h_total, 6),
        "existence_f1": existence_f1,
        "attribute_f1": attribute_f1,
        "relation_f1":  relation_f1,
        "weights_used": effective_weights,
        "n_dimensions": len(available),
    }


# ---------------------------------------------------------------------------
# End-to-end pipeline: raw predictions → H_total
# ---------------------------------------------------------------------------

def compute_h_total_from_predictions(
    existence_preds: Optional[list[str]] = None,
    existence_gts:   Optional[list[str]] = None,
    attribute_preds: Optional[list[str]] = None,
    attribute_gts:   Optional[list[str]] = None,
    relation_preds:  Optional[list[str]] = None,
    relation_gts:    Optional[list[str]] = None,
    weights: Optional[dict[str, float]] = None,
) -> dict[str, float]:
    """
    Compute H_total directly from raw model predictions.

    Each dimension is optional — pass None for both preds and gts to skip it.
    If only one of (preds, gts) is None for a dimension, raises ValueError.

    Returns the same dict as compute_h_total(), plus per-dimension dicts
    with full metric breakdowns under keys like "existence_metrics", etc.
    """
    def _get_f1(preds, gts, dim_name):
        if preds is None and gts is None:
            return None, None
        if (preds is None) ^ (gts is None):
            raise ValueError(
                f"For dimension '{dim_name}', provide both preds and gts or neither."
            )
        if len(preds) != len(gts):
            raise ValueError(
                f"'{dim_name}': preds ({len(preds)}) and gts ({len(gts)}) length mismatch."
            )
        metrics = compute_all(preds, gts)
        return metrics["f1"], metrics

    ex_f1,  ex_metrics  = _get_f1(existence_preds, existence_gts,   "existence")
    att_f1, att_metrics = _get_f1(attribute_preds, attribute_gts,   "attribute")
    rel_f1, rel_metrics = _get_f1(relation_preds,  relation_gts,    "relation")

    result = compute_h_total(
        existence_f1 = ex_f1,
        attribute_f1 = att_f1,
        relation_f1  = rel_f1,
        weights      = weights,
    )

    # Attach full per-dimension breakdowns for downstream use
    if ex_metrics:
        result["existence_metrics"] = ex_metrics
    if att_metrics:
        result["attribute_metrics"] = att_metrics
    if rel_metrics:
        result["relation_metrics"] = rel_metrics

    return result


# ---------------------------------------------------------------------------
# Multi-model comparison helper
# ---------------------------------------------------------------------------

def rank_models_by_h_total(
    model_results: dict[str, dict],
    weights: Optional[dict[str, float]] = None,
) -> list[dict]:
    """
    Given a dict of {model_name: result_dict}, compute H_total for each
    model and return a ranked list (best first).

    Parameters
    ----------
    model_results : {
        "Qwen2.5-VL-7B": {
            "existence_f1": 0.91,
            "attribute_f1": 0.74,
            "relation_f1":  0.68,
        },
        ...
    }

    Returns
    -------
    [
        {"rank": 1, "model": "Qwen2.5-VL-7B", "h_total": 0.773, ...},
        {"rank": 2, ...},
        ...
    ]
    """
    scored = []
    for model_name, res in model_results.items():
        h = compute_h_total(
            existence_f1 = res.get("existence_f1"),
            attribute_f1 = res.get("attribute_f1"),
            relation_f1  = res.get("relation_f1"),
            weights      = weights,
        )
        scored.append({"model": model_name, **h})

    scored.sort(key=lambda x: x["h_total"], reverse=True)
    for i, entry in enumerate(scored):
        entry["rank"] = i + 1

    return scored
