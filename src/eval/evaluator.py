"""
evaluator.py
------------
High-level Evaluator class.

Takes raw inference output (a JSONL file produced by run_inference.py)
and produces a structured results dict suitable for the analysis/ and
viz/ modules.

Expected input format (one JSON object per line)
-------------------------------------------------
{
    "model":       "Qwen2.5-VL-7B",
    "benchmark":   "pope_adversarial",   // or "repope", "dashb", "xpope"
    "image_id":    "COCO_val2014_000000...",
    "question":    "Is there a dog in the image?",
    "prediction":  "yes",
    "ground_truth":"yes",
    "category":    "existence"           // existence | attribute | relation
}

Usage
-----
    from eval.evaluator import Evaluator

    ev = Evaluator("results/qwen_pope_adversarial.jsonl")
    report = ev.evaluate()
    print(report)

    # Multi-file batch (e.g. all 6 models × 3 POPE splits)
    from eval.evaluator import batch_evaluate
    reports = batch_evaluate("results/")
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

from .metrics import compute_all, per_category_metrics, coefficient_of_variation, confusion, saturation_report as _sat_report
from .h_total import compute_h_total, rank_models_by_h_total


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class Evaluator:
    """
    Loads a JSONL inference file and computes all eval metrics.

    Attributes
    ----------
    path      : source file path
    records   : loaded list of dicts
    model     : model name (inferred from records or overridden)
    benchmark : benchmark name (inferred or overridden)
    """

    def __init__(
        self,
        path: str | Path,
        model: Optional[str] = None,
        benchmark: Optional[str] = None,
    ):
        self.path = Path(path)
        self.records = self._load(self.path)

        # Infer model / benchmark from data if not provided
        self.model     = model     or self._infer_field("model")
        self.benchmark = benchmark or self._infer_field("benchmark")

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    @staticmethod
    def _load(path: Path) -> list[dict]:
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"JSON parse error at line {lineno}: {e}") from e
        if not records:
            raise ValueError(f"No records found in {path}")
        return records

    def _infer_field(self, field: str) -> str:
        values = {r[field] for r in self.records if field in r}
        if len(values) == 1:
            return values.pop()
        if len(values) == 0:
            return "unknown"
        # Multiple values — return as sorted tuple string for transparency
        return str(sorted(values))

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def evaluate(self) -> dict:
        """
        Run full evaluation and return a structured report.

        Returns
        -------
        {
            "model":      str,
            "benchmark":  str,
            "overall":    {accuracy, f1, precision, recall, yes_rate, yes_bias, n_samples},
            "by_category":{
                "existence": {accuracy, f1, ...},
                "attribute": {...},
                "relation":  {...},
            },
            "h_total":    {h_total, existence_f1, attribute_f1, relation_f1, ...},
            "raw_cm":     {TP, FP, TN, FN},
        }
        """
        preds  = [r["prediction"]   for r in self.records]
        gts    = [r["ground_truth"] for r in self.records]
        cats   = [r.get("category", "existence") for r in self.records]

        # Overall (no category breakdown at this level)
        overall = compute_all(preds, gts)

        # Per category
        by_cat = per_category_metrics(preds, gts, cats)

        # H_total (uses per-category F1s; falls back gracefully if dims missing)
        h = compute_h_total(
            existence_f1 = by_cat.get("existence", {}).get("f1"),
            attribute_f1 = by_cat.get("attribute", {}).get("f1"),
            relation_f1  = by_cat.get("relation",  {}).get("f1"),
        )

        # Confusion matrix at overall level
        cm = confusion(preds, gts)

        return {
            "model":       self.model,
            "benchmark":   self.benchmark,
            "overall":     overall,
            "by_category": by_cat,
            "h_total":     h,
            "raw_cm":      cm,
        }

    def evaluate_by_strategy(self) -> dict[str, dict]:
        """
        Break results down by sampling strategy
        (random / popular / adversarial) if the field is present.

        Returns {strategy: full_eval_report} for each strategy found.
        """
        by_strategy: dict[str, list[dict]] = defaultdict(list)
        for r in self.records:
            strategy = r.get("strategy", "unknown")
            by_strategy[strategy].append(r)

        results = {}
        for strategy, records in sorted(by_strategy.items()):
            ev = Evaluator.__new__(Evaluator)
            ev.path      = self.path
            ev.records   = records
            ev.model     = self.model
            ev.benchmark = f"{self.benchmark}_{strategy}"
            results[strategy] = ev.evaluate()

        return results

    # ------------------------------------------------------------------
    # Pretty print
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Evaluator(model={self.model!r}, "
            f"benchmark={self.benchmark!r}, "
            f"n={len(self.records)})"
        )

    def summary(self) -> str:
        """Return a compact human-readable summary string."""
        report = self.evaluate()
        ov = report["overall"]
        ht = report["h_total"]
        lines = [
            f"Model     : {self.model}",
            f"Benchmark : {self.benchmark}",
            f"N samples : {ov['n_samples']}",
            f"Accuracy  : {ov['accuracy']:.4f}",
            f"F1        : {ov['f1']:.4f}",
            f"Yes-rate  : {ov['yes_rate']:.4f}  (bias {ov['yes_bias']:+.4f})",
            f"H_total   : {ht['h_total']:.4f}  ({ht['n_dimensions']} dims)",
        ]
        for cat, m in report["by_category"].items():
            lines.append(f"  {cat:<12}: Acc={m['accuracy']:.4f}  F1={m['f1']:.4f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def batch_evaluate(
    results_dir: str | Path,
    pattern: str = "*.jsonl",
) -> dict[str, dict]:
    """
    Evaluate all JSONL files in results_dir matching pattern.

    Returns
    -------
    {
        "<model>_<benchmark>": report_dict,
        ...
    }
    """
    results_dir = Path(results_dir)
    all_reports = {}
    for path in sorted(results_dir.glob(pattern)):
        try:
            ev = Evaluator(path)
            key = f"{ev.model}__{ev.benchmark}"
            all_reports[key] = ev.evaluate()
        except Exception as e:
            print(f"[WARN] Skipping {path.name}: {e}")
    return all_reports


def saturation_report(
    reports: dict[str, dict],
    benchmark_filter: Optional[str] = None,
) -> dict:
    """
    Module 1 helper: given a collection of eval reports,
    compute CV across models on the same benchmark to
    quantify saturation.

    Parameters
    ----------
    reports          : output of batch_evaluate()
    benchmark_filter : if given, only include reports whose benchmark
                       field matches (e.g. "pope_adversarial")

    Returns
    -------
    {
        "benchmark": str,
        "models"   : [str, ...],
        "acc_scores": [float, ...],
        "f1_scores" : [float, ...],
        "cv_acc"   : float,    # < 0.01 → saturated
        "cv_f1"    : float,
        "yes_bias" : {model: bias, ...},
        "saturated": bool,     # True if cv_acc < 0.01
    }
    """
    SAT_THRESHOLD = 0.01   # CV below this → saturated

    filtered = {
        k: v for k, v in reports.items()
        if benchmark_filter is None or v.get("benchmark") == benchmark_filter
    }

    if not filtered:
        raise ValueError(f"No reports match benchmark_filter={benchmark_filter!r}")

    models     = [v["model"]                    for v in filtered.values()]
    acc_scores = [v["overall"]["accuracy"]      for v in filtered.values()]
    f1_scores  = [v["overall"]["f1"]            for v in filtered.values()]
    biases     = {v["model"]: v["overall"]["yes_bias"] for v in filtered.values()}

    cv_acc = coefficient_of_variation(acc_scores) if len(acc_scores) >= 2 else float("nan")
    cv_f1  = coefficient_of_variation(f1_scores)  if len(f1_scores)  >= 2 else float("nan")
    return {
        "benchmark":  benchmark_filter or "all",
        "models":     models,
        "acc_scores": acc_scores,
        "f1_scores":  f1_scores,
        "cv_acc":     cv_acc,
        "cv_f1":      cv_f1,
        "yes_bias":   biases,
        "saturated":  cv_acc < SAT_THRESHOLD,
    }
