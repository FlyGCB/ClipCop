#!/usr/bin/env bash
# experiments/run_module1.sh
# Module 1: Run POPE inference on all 6 models, then compute CV saturation + Yes-bias diagnostics.
#
# Usage:
#   bash experiments/run_module1.sh [--dry-run]
#
# Outputs:
#   results/module1/          — JSONL inference results
#   figures/bias_bar.png      — Yes-bias diverging bar chart
#   reports/module1_saturation.json
#   reports/module1_bias.json

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RESULTS_DIR="results/module1"
FIGURES_DIR="figures"
REPORTS_DIR="reports"
DRY_RUN=false

mkdir -p "$RESULTS_DIR" "$FIGURES_DIR" "$REPORTS_DIR"

# ── Parse args ────────────────────────────────────────────────────────────────
for arg in "$@"; do
  case $arg in
    --dry-run) DRY_RUN=true ;;
    *) echo "Unknown argument: $arg"; exit 1 ;;
  esac
done

MODELS=(
  "qwen2.5-vl-7b"
  "internvl2.5-8b"
  "llava-onevision-7b"
  "llama-3.2-11b-vision"
  "paligemma2-3b"
  "idefics3-8b"
)

BENCHMARKS=(
  "pope_random"
  "pope_popular"
  "pope_adversarial"
)

# ── Step 1: Inference ─────────────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  MODULE 1 — Step 1/3: Running POPE inference"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

for model in "${MODELS[@]}"; do
  for bench in "${BENCHMARKS[@]}"; do
    out_file="$RESULTS_DIR/${model}_${bench}.jsonl"
    if [[ -f "$out_file" ]]; then
      echo "[skip] $out_file already exists"
      continue
    fi
    cmd="python -m src.models.run_inference \
      --model $model \
      --benchmark $bench \
      --output $out_file"
    echo "[run] $model × $bench"
    if [[ "$DRY_RUN" == false ]]; then
      eval "$cmd"
    else
      echo "  (dry-run) $cmd"
    fi
  done
done

# ── Step 2: Saturation diagnostics (CV) ──────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  MODULE 1 — Step 2/3: CV Saturation diagnostics"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [[ "$DRY_RUN" == false ]]; then
  python -m src.analysis.saturation_diag "$RESULTS_DIR" \
    --benchmarks pope_random pope_popular pope_adversarial \
    --json > "$REPORTS_DIR/module1_saturation.json"
  echo "[saved] $REPORTS_DIR/module1_saturation.json"
  python -m src.analysis.saturation_diag "$RESULTS_DIR" \
    --benchmarks pope_random pope_popular pope_adversarial
else
  echo "(dry-run) saturation_diag $RESULTS_DIR --json > $REPORTS_DIR/module1_saturation.json"
fi

# ── Step 3: Yes-bias analysis + chart ────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  MODULE 1 — Step 3/3: Yes-bias analysis + chart"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [[ "$DRY_RUN" == false ]]; then
  python -m src.analysis.bias_analysis "$RESULTS_DIR" \
    --benchmarks pope_random pope_popular pope_adversarial \
    --json > "$REPORTS_DIR/module1_bias.json"
  echo "[saved] $REPORTS_DIR/module1_bias.json"

  python -m src.viz.plot_bias "$REPORTS_DIR/module1_bias.json" \
    --output "$FIGURES_DIR/module1_bias_bar.png" \
    --title "Module 1 — Yes-Bias on POPE (random/popular/adversarial)"
else
  echo "(dry-run) bias_analysis → $REPORTS_DIR/module1_bias.json"
  echo "(dry-run) plot_bias    → $FIGURES_DIR/module1_bias_bar.png"
fi

echo ""
echo "✅  Module 1 complete."
echo "    Saturation report : $REPORTS_DIR/module1_saturation.json"
echo "    Bias report       : $REPORTS_DIR/module1_bias.json"
echo "    Bias chart        : $FIGURES_DIR/module1_bias_bar.png"
