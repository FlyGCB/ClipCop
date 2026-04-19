#!/usr/bin/env bash
# experiments/run_module3.sh
# Module 3: Re-run on DASH-B (harder benchmark), check if discriminability recovers.
#           Key metrics: CV (vs POPE), Spearman ρ (RePOPE → DASH-B).
#
# Usage:
#   bash experiments/run_module3.sh [--dry-run]
#
# Prerequisites: run_module1.sh and run_module2.sh must have completed.
#
# Outputs:
#   results/module3/                    — DASH-B JSONL results
#   figures/module3_bump.png            — Bump chart POPE → RePOPE → DASH-B
#   figures/module3_bias_bar.png        — Bias chart on DASH-B
#   reports/module3_ranking.json
#   reports/module3_saturation.json

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RESULTS_DIR="results/module3"
FIGURES_DIR="figures"
REPORTS_DIR="reports"
DRY_RUN=false

mkdir -p "$RESULTS_DIR" "$FIGURES_DIR" "$REPORTS_DIR"

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

# ── Step 1: DASH-B inference ──────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  MODULE 3 — Step 1/3: DASH-B inference"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

for model in "${MODELS[@]}"; do
  out_file="$RESULTS_DIR/${model}_dashb_adversarial.jsonl"
  if [[ -f "$out_file" ]]; then
    echo "[skip] $out_file already exists"
    continue
  fi
  cmd="python -m src.models.run_inference \
    --model $model \
    --benchmark dashb_adversarial \
    --output $out_file"
  echo "[run] $model × dashb_adversarial"
  if [[ "$DRY_RUN" == false ]]; then
    eval "$cmd"
  else
    echo "  (dry-run) $cmd"
  fi
done

# ── Step 2: Three-way ranking shift POPE → RePOPE → DASH-B ───────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  MODULE 3 — Step 2/3: Three-way ranking shift"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

MERGED_DIR="results/module3_merged"
mkdir -p "$MERGED_DIR"

if [[ "$DRY_RUN" == false ]]; then
  for f in results/module1/*_pope_adversarial.jsonl \
           results/module2/*_repope_adversarial.jsonl \
           "$RESULTS_DIR"/*.jsonl; do
    [[ -f "$f" ]] && ln -sf "$(realpath "$f")" "$MERGED_DIR/$(basename "$f")" 2>/dev/null || true
  done

  python -m src.analysis.ranking_shift "$MERGED_DIR" \
    --benchmarks pope_adversarial repope_adversarial dashb_adversarial \
    --json > "$REPORTS_DIR/module3_ranking.json"
  echo "[saved] $REPORTS_DIR/module3_ranking.json"

  python -m src.analysis.ranking_shift "$MERGED_DIR" \
    --benchmarks pope_adversarial repope_adversarial dashb_adversarial

  python -m src.viz.plot_ranking "$REPORTS_DIR/module3_ranking.json" \
    --output "$FIGURES_DIR/module3_bump.png" \
    --title "Module 3 — Ranking Shift: POPE → RePOPE → DASH-B"
else
  echo "(dry-run) ranking_shift (3-way) → $REPORTS_DIR/module3_ranking.json"
  echo "(dry-run) plot_ranking          → $FIGURES_DIR/module3_bump.png"
fi

# ── Step 3: CV saturation comparison (POPE vs DASH-B) ────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  MODULE 3 — Step 3/3: CV saturation comparison"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [[ "$DRY_RUN" == false ]]; then
  python -m src.analysis.saturation_diag "$MERGED_DIR" \
    --benchmarks pope_adversarial dashb_adversarial \
    --json > "$REPORTS_DIR/module3_saturation.json"
  echo "[saved] $REPORTS_DIR/module3_saturation.json"
  python -m src.analysis.saturation_diag "$MERGED_DIR" \
    --benchmarks pope_adversarial dashb_adversarial
else
  echo "(dry-run) saturation_diag → $REPORTS_DIR/module3_saturation.json"
fi

echo ""
echo "✅  Module 3 complete."
echo "    Ranking report     : $REPORTS_DIR/module3_ranking.json"
echo "    Saturation report  : $REPORTS_DIR/module3_saturation.json"
echo "    Bump chart         : $FIGURES_DIR/module3_bump.png"
