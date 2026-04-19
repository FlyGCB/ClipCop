"""
src/analysis/ranking_shift.py

Compute Spearman rank correlations across POPE → RePOPE → DASH-B benchmarks
to quantify whether model rankings are stable or significantly shift.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.stats import spearmanr


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BENCHMARK_ORDER = ["pope_adversarial", "repope_adversarial", "dashb_adversarial"]
BENCHMARK_LABELS = {
    "pope_adversarial":   "POPE",
    "repope_adversarial": "RePOPE",
    "dashb_adversarial":  "DASH-B",
}

# ρ below this threshold is considered a "significant" ranking shift
RHO_SHIFT_THRESHOLD = 0.7


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkRanking:
    """Ranked model list for a single benchmark."""
    benchmark: str
    # model_name → F1 score (0–1)
    scores: dict[str, float]
    # model_name → rank (1 = best)
    ranks: dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        sorted_models = sorted(self.scores, key=lambda m: self.scores[m], reverse=True)
        self.ranks = {model: i + 1 for i, model in enumerate(sorted_models)}


@dataclass
class PairwiseShift:
    """Spearman ρ between two consecutive benchmarks."""
    benchmark_a: str
    benchmark_b: str
    rho: float
    p_value: float
    common_models: list[str]
    significant_shift: bool  # True when ρ < RHO_SHIFT_THRESHOLD

    @property
    def label_a(self) -> str:
        return BENCHMARK_LABELS.get(self.benchmark_a, self.benchmark_a)

    @property
    def label_b(self) -> str:
        return BENCHMARK_LABELS.get(self.benchmark_b, self.benchmark_b)

    def summary(self) -> str:
        shift_tag = "⚠ SIGNIFICANT SHIFT" if self.significant_shift else "✓ stable"
        return (
            f"{self.label_a} → {self.label_b}: "
            f"ρ = {self.rho:+.4f}, p = {self.p_value:.4f}  [{shift_tag}]"
        )


@dataclass
class RankingShiftReport:
    """Full report: ranking matrices + pairwise Spearman ρ values."""
    rankings: list[BenchmarkRanking]          # one per benchmark
    pairwise: list[PairwiseShift]             # consecutive pairs
    rank_matrix: dict[str, dict[str, int]]    # model → {benchmark → rank}
    score_matrix: dict[str, dict[str, float]] # model → {benchmark → F1}

    def print(self) -> None:
        print("=" * 60)
        print("  RANKING SHIFT REPORT")
        print("=" * 60)

        # Score matrix
        benchmarks = [r.benchmark for r in self.rankings]
        labels = [BENCHMARK_LABELS.get(b, b) for b in benchmarks]
        all_models = list(self.rank_matrix.keys())

        col_w = 10
        header = f"{'Model':<30}" + "".join(f"{lb:>{col_w}}" for lb in labels)
        print("
F1 Scores:")
        print(header)
        print("-" * len(header))
        for model in all_models:
            row = f"{model:<30}"
            for b in benchmarks:
                score = self.score_matrix[model].get(b)
                row += f"{score:>{col_w}.4f}" if score is not None else f"{'N/A':>{col_w}}"
            print(row)

        # Rank matrix
        print("
Ranks (1 = best):")
        print(header)
        print("-" * len(header))
        for model in all_models:
            row = f"{model:<30}"
            for b in benchmarks:
                rank = self.rank_matrix[model].get(b)
                row += f"{rank:>{col_w}}" if rank is not None else f"{'N/A':>{col_w}}"
            print(row)

        # Pairwise ρ
        print("
Spearman ρ (pairwise):")
        for pw in self.pairwise:
            print(f"  {pw.summary()}")
            print(f"    (n={len(pw.common_models)} common models)")

        print("=" * 60)


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def _extract_f1_per_benchmark(
    model_results: dict[str, dict],
    benchmarks: list[str],
) -> dict[str, dict[str, float]]:
    """
    Extract F1 scores from batch_evaluate() output.

    Expected input shape (one entry per model):
        {
          "Qwen2.5-VL-7B": {
              "pope_adversarial":   {"f1": 0.91, ...},
              "repope_adversarial": {"f1": 0.88, ...},
              ...
          },
          ...
        }

    Returns: model → {benchmark → f1}
    """
    result: dict[str, dict[str, float]] = {}
    for model, bench_reports in model_results.items():
        result[model] = {}
        for bench in benchmarks:
            report = bench_reports.get(bench)
            if report is None:
                warnings.warn(
                    f"Model '{model}' has no results for benchmark '{bench}'. "
                    "It will be excluded from pairwise ρ involving this benchmark.",
                    UserWarning,
                    stacklevel=3,
                )
                continue
            # Support both dict-style and object-style reports
            f1 = report["f1"] if isinstance(report, dict) else report.f1
            result[model][bench] = float(f1)
    return result


def _compute_pairwise_spearman(
    rankings: list[BenchmarkRanking],
) -> list[PairwiseShift]:
    """Compute Spearman ρ for each consecutive benchmark pair."""
    pairs: list[PairwiseShift] = []
    for a, b in zip(rankings, rankings[1:]):
        common = sorted(set(a.scores) & set(b.scores))
        if len(common) < 3:
            warnings.warn(
                f"Only {len(common)} common models between "
                f"'{a.benchmark}' and '{b.benchmark}'. "
                "Spearman ρ may not be reliable.",
                UserWarning,
            )
        ranks_a = [a.ranks[m] for m in common]
        ranks_b = [b.ranks[m] for m in common]
        rho, p_val = spearmanr(ranks_a, ranks_b)
        pairs.append(PairwiseShift(
            benchmark_a=a.benchmark,
            benchmark_b=b.benchmark,
            rho=float(rho),
            p_value=float(p_val),
            common_models=common,
            significant_shift=float(rho) < RHO_SHIFT_THRESHOLD,
        ))
    return pairs


def compute_ranking_shift(
    model_results: dict[str, dict],
    benchmarks: Optional[list[str]] = None,
) -> RankingShiftReport:
    """
    Main entry point.

    Parameters
    ----------
    model_results : dict
        Output of batch_evaluate(), keyed by model name.
        Each value is a dict mapping benchmark name → eval report.
    benchmarks : list[str], optional
        Ordered benchmark names to compare.
        Defaults to BENCHMARK_ORDER = [pope, repope, dashb].

    Returns
    -------
    RankingShiftReport
    """
    if benchmarks is None:
        benchmarks = BENCHMARK_ORDER

    score_matrix = _extract_f1_per_benchmark(model_results, benchmarks)

    # Build BenchmarkRanking for each benchmark (only models with a score)
    rankings: list[BenchmarkRanking] = []
    for bench in benchmarks:
        scores = {
            model: score_matrix[model][bench]
            for model in score_matrix
            if bench in score_matrix[model]
        }
        if not scores:
            warnings.warn(f"No model has results for benchmark '{bench}'. Skipping.", UserWarning)
            continue
        rankings.append(BenchmarkRanking(benchmark=bench, scores=scores))

    pairwise = _compute_pairwise_spearman(rankings)

    # Build rank matrix (None if model missing for a benchmark)
    all_models = sorted(score_matrix.keys())
    rank_matrix: dict[str, dict[str, int]] = {m: {} for m in all_models}
    for br in rankings:
        for model in all_models:
            if model in br.ranks:
                rank_matrix[model][br.benchmark] = br.ranks[model]

    return RankingShiftReport(
        rankings=rankings,
        pairwise=pairwise,
        rank_matrix=rank_matrix,
        score_matrix=score_matrix,
    )


# ---------------------------------------------------------------------------
# Convenience: load from evaluator output
# ---------------------------------------------------------------------------

def ranking_shift_from_evaluator(results_dir: str) -> RankingShiftReport:
    """
    Load batch_evaluate() results from a directory of JSONL files
    and compute ranking shift.

    Expects files named like:
        results/qwen_pope_adversarial.jsonl
        results/qwen_repope_adversarial.jsonl
        ...

    Delegates to evaluator.batch_evaluate() then calls compute_ranking_shift().
    """
    # Import here to avoid circular dependency
    from eval.evaluator import batch_evaluate  # noqa: PLC0415

    raw = batch_evaluate(results_dir)          # dict[model → dict[bench → EvalReport]]
    return compute_ranking_shift(raw)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, json, sys

    parser = argparse.ArgumentParser(description="Compute ranking shift (Spearman ρ) across benchmarks.")
    parser.add_argument("results_dir", help="Directory containing JSONL inference results.")
    parser.add_argument("--benchmarks", nargs="+", default=None,
                        help="Benchmark names in order (default: pope → repope → dashb).")
    parser.add_argument("--json", action="store_true", help="Output raw JSON instead of formatted report.")
    args = parser.parse_args()

    report = ranking_shift_from_evaluator(args.results_dir)

    if args.json:
        out = {
            "pairwise": [
                {
                    "benchmark_a": pw.benchmark_a,
                    "benchmark_b": pw.benchmark_b,
                    "rho": pw.rho,
                    "p_value": pw.p_value,
                    "significant_shift": pw.significant_shift,
                    "common_models": pw.common_models,
                }
                for pw in report.pairwise
            ],
            "rank_matrix": report.rank_matrix,
            "score_matrix": report.score_matrix,
        }
        print(json.dumps(out, indent=2))
    else:
        report.print()
