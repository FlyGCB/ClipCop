"""
src/eval/
---------
Evaluation module for the beyond-POPE project.

Public API
----------
    from eval.metrics   import (accuracy, f1, precision, recall,
                                precision_recall, confusion,
                                yes_rate, yes_bias,
                                coefficient_of_variation, cv_across_models,
                                per_category_metrics, compute_all,
                                saturation_report)
    from eval.h_total   import (compute_h_total, compute_h_total_from_predictions,
                                rank_models_by_h_total)
    from eval.evaluator import (Evaluator, batch_evaluate, saturation_report)
"""

from .metrics import (
    accuracy,
    f1,
    precision,
    recall,
    precision_recall,
    confusion,
    yes_rate,
    yes_bias,
    coefficient_of_variation,
    cv_across_models,
    per_category_metrics,
    compute_all,
    saturation_report,
)

from .h_total import (
    compute_h_total,
    compute_h_total_from_predictions,
    rank_models_by_h_total,
)

from .evaluator import (
    Evaluator,
    batch_evaluate,
)

__all__ = [
    # metrics
    "accuracy", "f1", "precision", "recall", "precision_recall", "confusion",
    "yes_rate", "yes_bias",
    "coefficient_of_variation", "cv_across_models",
    "per_category_metrics", "compute_all", "saturation_report",
    # h_total
    "compute_h_total", "compute_h_total_from_predictions", "rank_models_by_h_total",
    # evaluator
    "Evaluator", "batch_evaluate",
]
