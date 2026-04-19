"""
src/analysis/saturation_diag.py

Coefficient of Variation (CV) saturation diagnostics.

CV = std(F1) / mean(F1)  across models for a given benchmark.

Low CV → models are clustered together → benchmark is saturated → poor discriminability.
High CV → benchmark still separates models well.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CV_SATURATED_THRESHOLD  = 0.02   # CV ≤ this → saturated
CV_MARGINAL_THRESHOLD   = 0.05   # CV ≤ this → marginal discriminability

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

def _cv_status(cv: float) -> str:
    if cv <= CV_SATURATED_THRESHOLD:
        return "SATURATED"
    if cv <= CV_MARGINAL_THRESHOLD:
        return "marginal"
    return "discriminative"


@dataclass
class BenchmarkSaturation:
    """Saturation statistics for a single benchmark."""
    benchmark: str
    model_scores: dict[str, float]    # model → F1
    mean_f1: float
    std_f1: float
    cv: float                         # coefficient of variation
    min_f1: float
    max_f1: float
    f1_range: float                   # max - min
    status: str                       # SATURATED / marginal / discriminative
    n_models: int

    @classmethod
    def from_scores(cls, benchmark: str, scores: dict[str, float]) -> "BenchmarkSaturation":
        vals = np.array(list(scores.values()), dtype=float)
        mean = float(np.mean(vals))
        std  = float(np.std(vals))
        cv   = float(std / mean) if mean > 0 else 0.0
        return cls(
            benchmark=benchmark,
            model_scores=scores,
            mean_f1=mean,
            std_f1=std,
            cv=cv,
            min_f1=float(np.min(vals)),
            max_f1=float(np.max(vals)),
            f1_range=float(np.max(vals) - np.min(vals)),
            status=_cv_status(cv),
            n_models=len(scores),
        )

    @property
    def label(self) -> str:
        return BENCHMARK_LABELS.get(self.benchmark, self.benchmark)

    def summary_line(self) -> str:
        bar_len = int(self.cv * 500)   # scale CV to visible bar
        bar = "█" * min(bar_len, 40)
        return (
            f"  {self.label:<22} CV={self.cv:.4f}  mean={self.mean_f1:.4f}  "
            f"range={self.f1_range:.4f}  {bar}  [{self.status}]"
        )


@dataclass
class SaturationReport:
    """Full saturation diagnostics across benchmarks."""
    benchmarks: list[BenchmarkSaturation]
    saturated: list[str]    = field(init=False)   # benchmark names
    marginal: list[str]     = field(init=False)
    discriminative: list[str] = field(init=False)

    def __post_init__(self):
        self.saturated      = [b.benchmark for b in self.benchmarks if b.status == "SATURATED"]
        self.marginal       = [b.benchmark for b in self.benchmarks if b.status == "marginal"]
        self.discriminative = [b.benchmark for b in self.benchmarks if b.status == "discriminative"]

    def print(self) -> None:
        print("=" * 75)
        print("  SATURATION DIAGNOSTIC REPORT")
        print(f"  Thresholds — saturated: CV ≤ {CV_SATURATED_THRESHOLD}, "
              f"marginal: CV ≤ {CV_MARGINAL_THRESHOLD}")
        print("=" * 75)

        print("
Benchmark CV Overview (sorted by CV ascending = most saturated first):")
        for b in sorted(self.benchmarks, key=lambda x: x.cv):
            print(b.summary_line())

        print("
Per-Benchmark Model Scores:")
        for b in self.benchmarks:
            print(f"
  [{b.label}]  status={b.status}  CV={b.cv:.4f}")
            sorted_models = sorted(b.model_scores, key=lambda m: b.model_scores[m], reverse=True)
            for rank, model in enumerate(sorted_models, 1):
                score = b.model_scores[model]
                bar = "█" * int(score * 30)
                print(f"    #{rank}  {model:<30} F1={score:.4f}  {bar}")

        print("
Summary:")
        if self.saturated:
            labels = [BENCHMARK_LABELS.get(b, b) for b in self.saturated]
            print(f"  ⚠  SATURATED ({len(self.saturated)}): {', '.join(labels)}")
        if self.marginal:
            labels = [BENCHMARK_LABELS.get(b, b) for b in self.marginal]
            print(f"  ~  Marginal  ({len(self.marginal)}): {', '.join(labels)}")
        if self.discriminative:
            labels = [BENCHMARK_LABELS.get(b, b) for b in self.discriminative]
            print(f"  ✓  Discriminative ({len(self.discriminative)}): {', '.join(labels)}")

        print("=" * 75)


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_saturation(
    model_results: dict[str, dict],
    benchmarks: Optional[list[str]] = None,
    metric: str = "f1",
) -> SaturationReport:
    """
    Run saturation diagnostics.

    Parameters
    ----------
    model_results : dict
        Output of batch_evaluate():
        { model_name → { benchmark_name → EvalReport } }
    benchmarks : list[str], optional
        Subset of benchmarks. None = all available.
    metric : str
        Which metric to use for CV calculation. Default 'f1'.

    Returns
    -------
    SaturationReport
    """
    # Collect all benchmarks present in results
    all_benchmarks: set[str] = set()
    for bench_reports in model_results.values():
        all_benchmarks.update(bench_reports.keys())

    if benchmarks:
        all_benchmarks = all_benchmarks & set(benchmarks)

    bench_stats: list[BenchmarkSaturation] = []

    for bench in sorted(all_benchmarks):
        scores: dict[str, float] = {}
        for model, bench_reports in model_results.items():
            report = bench_reports.get(bench)
            if report is None:
                continue
            val = report.get(metric) if isinstance(report, dict) else getattr(report, metric, None)
            if val is None:
                warnings.warn(
                    f"Model '{model}' benchmark '{bench}' missing '{metric}'. Skipping.",
                    UserWarning,
                )
                continue
            scores[model] = float(val)

        if len(scores) < 2:
            warnings.warn(
                f"Benchmark '{bench}' has fewer than 2 models with '{metric}'. "
                "CV not meaningful. Skipping.",
                UserWarning,
            )
            continue

        bench_stats.append(BenchmarkSaturation.from_scores(bench, scores))

    if not bench_stats:
        raise ValueError("No valid benchmark data found for saturation analysis.")

    return SaturationReport(benchmarks=bench_stats)


def saturation_from_evaluator(
    results_dir: str,
    benchmarks: Optional[list[str]] = None,
    metric: str = "f1",
) -> SaturationReport:
    """Load JSONL results and run saturation diagnostics."""
    from eval.evaluator import batch_evaluate  # noqa: PLC0415
    raw = batch_evaluate(results_dir)
    return compute_saturation(raw, benchmarks=benchmarks, metric=metric)


# ---------------------------------------------------------------------------
# Utilities for cross-benchmark comparison
# ---------------------------------------------------------------------------

def compare_cv_across_benchmarks(
    report: SaturationReport,
) -> dict[str, float]:
    """Return {benchmark_name: cv} dict sorted descending (most discriminative first)."""
    return dict(
        sorted(
            {b.benchmark: b.cv for b in report.benchmarks}.items(),
            key=lambda x: x[1],
            reverse=True,
        )
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(description="CV-based saturation diagnostics.")
    parser.add_argument("results_dir", help="Directory containing JSONL inference results.")
    parser.add_argument("--benchmarks", nargs="+", default=None)
    parser.add_argument("--metric", default="f1", help="Metric to compute CV on (default: f1).")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = saturation_from_evaluator(args.results_dir, benchmarks=args.benchmarks, metric=args.metric)

    if args.json:
        out = {
            "benchmarks": [
                {
                    "benchmark": b.benchmark,
                    "label": b.label,
                    "cv": b.cv,
                    "mean_f1": b.mean_f1,
                    "std_f1": b.std_f1,
                    "f1_range": b.f1_range,
                    "status": b.status,
                    "model_scores": b.model_scores,
                }
                for b in report.benchmarks
            ],
            "saturated": report.saturated,
            "marginal": report.marginal,
            "discriminative": report.discriminative,
        }
        print(json.dumps(out, indent=2))
    else:
        report.print()
