#!/usr/bin/env bash
# experiments/run_module2.sh
# Module 2: Re-run inference on RePOPE (annotation-corrected), compare ranking with POPE.
#           Key metric: Spearman ρ (POPE → RePOPE). Low ρ → annotation errors matter.
#
# Usage:
#   bash experiments/run_module2.sh [--dry-run]
#
# Prerequisites: run_module1.sh must have completed (reuses POPE results).
#
# Outputs:
#   results/module2/               — RePOPE JSONL results
#   figures/module2_bump.png       — Bump chart POPE → RePOPE
#   reports/module2_ranking.json

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

POPE_RESULTS="results/module1"
RESULTS_DIR="results/module2"
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

# ── Step 1: RePOPE inference ──────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  MODULE 2 — Step 1/2: RePOPE inference"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

for model in "${MODELS[@]}"; do
  out_file="$RESULTS_DIR/${model}_repope_adversarial.jsonl"
  if [[ -f "$out_file" ]]; then
    echo "[skip] $out_file already exists"
    continue
  fi
  cmd="python -m src.models.run_inference \
    --model $model \
    --benchmark repope_adversarial \
    --output $out_file"
  echo "[run] $model × repope_adversarial"
  if [[ "$DRY_RUN" == false ]]; then
    eval "$cmd"
  else
    echo "  (dry-run) $cmd"
  fi
done

# ── Step 2: Ranking shift POPE → RePOPE ──────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  MODULE 2 — Step 2/2: Ranking shift + bump chart"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Merge POPE adversarial + RePOPE results into a single temp directory for analysis
MERGED_DIR="results/module2_merged"
mkdir -p "$MERGED_DIR"

if [[ "$DRY_RUN" == false ]]; then
  # Symlink pope_adversarial results from module1
  for f in "$POPE_RESULTS"/*_pope_adversarial.jsonl; do
    ln -sf "$(realpath "$f")" "$MERGED_DIR/$(basename "$f")" 2>/dev/null || true
  done
  # Symlink repope results
  for f in "$RESULTS_DIR"/*.jsonl; do
    ln -sf "$(realpath "$f")" "$MERGED_DIR/$(basename "$f")" 2>/dev/null || true
  done

  python -m src.analysis.ranking_shift "$MERGED_DIR" \
    --benchmarks pope_adversarial repope_adversarial \
    --json > "$REPORTS_DIR/module2_ranking.json"
  echo "[saved] $REPORTS_DIR/module2_ranking.json"

  python -m src.analysis.ranking_shift "$MERGED_DIR" \
    --benchmarks pope_adversarial repope_adversarial

  python -m src.viz.plot_ranking "$REPORTS_DIR/module2_ranking.json" \
    --output "$FIGURES_DIR/module2_bump.png" \
    --title "Module 2 — Ranking Shift: POPE → RePOPE" \
    --benchmarks pope_adversarial repope_adversarial
else
  echo "(dry-run) ranking_shift $MERGED_DIR → $REPORTS_DIR/module2_ranking.json"
  echo "(dry-run) plot_ranking  → $FIGURES_DIR/module2_bump.png"
fi

echo ""
echo "✅  Module 2 complete."
echo "    Ranking report : $REPORTS_DIR/module2_ranking.json"
echo "    Bump chart     : $FIGURES_DIR/module2_bump.png"
