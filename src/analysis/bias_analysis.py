"""
src/analysis/bias_analysis.py

Analyse Yes-bias distribution across models and benchmarks.

Yes-bias = yes_rate - 0.5
  > 0  → model over-predicts "yes" (hallucination-prone)
  < 0  → model over-predicts "no"  (over-cautious)
  = 0  → perfectly balanced
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BIAS_SEVERE_THRESHOLD = 0.15   # |yes_bias| ≥ this → severe
BIAS_MODERATE_THRESHOLD = 0.07 # |yes_bias| ≥ this → moderate

BENCHMARK_LABELS = {
    "pope_adversarial":   "POPE",
    "repope_adversarial": "RePOPE",
    "dashb_adversarial":  "DASH-B",
    "xpope_existence":    "X-POPE (exist)",
    "xpope_attribute":    "X-POPE (attr)",
    "xpope_relation":     "X-POPE (rel)",
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

def _bias_level(yes_bias: float) -> str:
    ab = abs(yes_bias)
    if ab >= BIAS_SEVERE_THRESHOLD:
        return "severe"
    if ab >= BIAS_MODERATE_THRESHOLD:
        return "moderate"
    return "mild"


def _bias_direction(yes_bias: float) -> str:
    if yes_bias > 0.01:
        return "yes-biased"
    if yes_bias < -0.01:
        return "no-biased"
    return "balanced"


@dataclass
class ModelBiasEntry:
    """Yes-bias for one model on one benchmark."""
    model: str
    benchmark: str
    yes_rate: float        # fraction of "yes" answers (0–1)
    yes_bias: float        # yes_rate - 0.5
    level: str             # mild / moderate / severe
    direction: str         # yes-biased / no-biased / balanced

    @classmethod
    def from_report(cls, model: str, benchmark: str, report: dict | object) -> "ModelBiasEntry":
        yr = report["yes_rate"] if isinstance(report, dict) else report.yes_rate
        yb = float(yr) - 0.5
        return cls(
            model=model,
            benchmark=benchmark,
            yes_rate=float(yr),
            yes_bias=yb,
            level=_bias_level(yb),
            direction=_bias_direction(yb),
        )


@dataclass
class BiasSummary:
    """Aggregated yes-bias statistics for one model across all benchmarks."""
    model: str
    entries: list[ModelBiasEntry]
    mean_bias: float = field(init=False)
    std_bias: float  = field(init=False)
    max_abs_bias: float = field(init=False)
    worst_benchmark: str = field(init=False)

    def __post_init__(self):
        biases = [e.yes_bias for e in self.entries]
        self.mean_bias = float(np.mean(biases))
        self.std_bias  = float(np.std(biases))
        worst = max(self.entries, key=lambda e: abs(e.yes_bias))
        self.max_abs_bias  = abs(worst.yes_bias)
        self.worst_benchmark = worst.benchmark

    @property
    def overall_level(self) -> str:
        return _bias_level(self.mean_bias)

    @property
    def overall_direction(self) -> str:
        return _bias_direction(self.mean_bias)


@dataclass
class BiasAnalysisReport:
    """Full yes-bias report across all models and benchmarks."""
    entries: list[ModelBiasEntry]           # flat list
    summaries: dict[str, BiasSummary]       # model → summary
    # benchmark → list of (model, yes_bias) sorted by |bias|
    per_benchmark: dict[str, list[tuple[str, float]]]

    def print(self) -> None:
        print("=" * 70)
        print("  YES-BIAS ANALYSIS REPORT")
        print(f"  Thresholds — moderate: ±{BIAS_MODERATE_THRESHOLD}, severe: ±{BIAS_SEVERE_THRESHOLD}")
        print("=" * 70)

        # Per-model summary
        print("
Per-Model Summary (mean yes_bias across benchmarks):")
        header = f"  {'Model':<30} {'Mean bias':>10} {'Std':>8} {'Max|bias|':>10} {'Level':<10} {'Direction'}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for model, s in sorted(self.summaries.items(), key=lambda x: abs(x[1].mean_bias), reverse=True):
            print(
                f"  {model:<30} {s.mean_bias:>+10.4f} {s.std_bias:>8.4f} "
                f"{s.max_abs_bias:>10.4f} {s.overall_level:<10} {s.overall_direction}"
            )

        # Per-benchmark detail
        print("
Per-Benchmark Bias (sorted by |bias|):")
        for bench, ranked in self.per_benchmark.items():
            label = BENCHMARK_LABELS.get(bench, bench)
            print(f"
  [{label}]")
            for model, yb in ranked:
                bar = "█" * int(abs(yb) * 40)
                direction = "→yes" if yb > 0 else "→no "
                level = _bias_level(yb)
                print(f"    {model:<30} {yb:>+7.4f} {direction}  {bar}  [{level}]")

        print("=" * 70)


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_bias_analysis(
    model_results: dict[str, dict],
    benchmarks: Optional[list[str]] = None,
) -> BiasAnalysisReport:
    """
    Compute yes-bias analysis from batch_evaluate() output.

    Parameters
    ----------
    model_results : dict
        Output of batch_evaluate():
        { model_name → { benchmark_name → EvalReport (dict or object with .yes_rate) } }
    benchmarks : list[str], optional
        Subset of benchmarks to analyse. None = all available.

    Returns
    -------
    BiasAnalysisReport
    """
    entries: list[ModelBiasEntry] = []

    for model, bench_reports in model_results.items():
        for bench, report in bench_reports.items():
            if benchmarks and bench not in benchmarks:
                continue
            # guard: report must have yes_rate
            yr = report.get("yes_rate") if isinstance(report, dict) else getattr(report, "yes_rate", None)
            if yr is None:
                warnings.warn(
                    f"Model '{model}' benchmark '{bench}' missing yes_rate. Skipping.",
                    UserWarning,
                )
                continue
            entries.append(ModelBiasEntry.from_report(model, bench, report))

    if not entries:
        raise ValueError("No valid entries found. Check that model_results contains yes_rate fields.")

    # Per-model summaries
    models = sorted({e.model for e in entries})
    summaries: dict[str, BiasSummary] = {}
    for model in models:
        model_entries = [e for e in entries if e.model == model]
        if model_entries:
            summaries[model] = BiasSummary(model=model, entries=model_entries)

    # Per-benchmark ranked lists
    all_benchmarks = sorted({e.benchmark for e in entries})
    per_benchmark: dict[str, list[tuple[str, float]]] = {}
    for bench in all_benchmarks:
        bench_entries = [e for e in entries if e.benchmark == bench]
        per_benchmark[bench] = sorted(
            [(e.model, e.yes_bias) for e in bench_entries],
            key=lambda x: abs(x[1]),
            reverse=True,
        )

    return BiasAnalysisReport(
        entries=entries,
        summaries=summaries,
        per_benchmark=per_benchmark,
    )


def bias_analysis_from_evaluator(results_dir: str) -> BiasAnalysisReport:
    """Load from JSONL results directory and run bias analysis."""
    from eval.evaluator import batch_evaluate  # noqa: PLC0415
    raw = batch_evaluate(results_dir)
    return compute_bias_analysis(raw)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(description="Yes-bias distribution analysis.")
    parser.add_argument("results_dir", help="Directory containing JSONL inference results.")
    parser.add_argument("--benchmarks", nargs="+", default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = bias_analysis_from_evaluator(args.results_dir)

    if args.json:
        out = {
            "summaries": {
                m: {
                    "mean_bias": s.mean_bias,
                    "std_bias": s.std_bias,
                    "max_abs_bias": s.max_abs_bias,
                    "worst_benchmark": s.worst_benchmark,
                    "overall_level": s.overall_level,
                    "overall_direction": s.overall_direction,
                }
                for m, s in report.summaries.items()
            },
            "entries": [
                {
                    "model": e.model,
                    "benchmark": e.benchmark,
                    "yes_rate": e.yes_rate,
                    "yes_bias": e.yes_bias,
                    "level": e.level,
                    "direction": e.direction,
                }
                for e in report.entries
            ],
        }
        print(json.dumps(out, indent=2))
    else:
        report.print()
