#!/usr/bin/env bash
# experiments/run_module4.sh
# Module 4: X-POPE — existence + attribute + relation hallucination.
#           Compute H_total, rank models, draw radar chart.
#
# Usage:
#   bash experiments/run_module4.sh [--dry-run]
#
# Outputs:
#   results/module4/              — X-POPE JSONL results (3 splits × 6 models)
#   figures/module4_radar.png     — Three-dimension radar chart
#   figures/module4_bias_bar.png  — Yes-bias across X-POPE splits
#   reports/module4_htotal.json   — H_total ranking
#   reports/module4_bias.json

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RESULTS_DIR="results/module4"
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

XPOPE_SPLITS=(
  "xpope_existence"
  "xpope_attribute"
  "xpope_relation"
)

# ── Step 1: X-POPE inference ──────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  MODULE 4 — Step 1/4: X-POPE inference (3 splits)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

for model in "${MODELS[@]}"; do
  for split in "${XPOPE_SPLITS[@]}"; do
    out_file="$RESULTS_DIR/${model}_${split}.jsonl"
    if [[ -f "$out_file" ]]; then
      echo "[skip] $out_file already exists"
      continue
    fi
    cmd="python -m src.models.run_inference \
      --model $model \
      --benchmark $split \
      --output $out_file"
    echo "[run] $model × $split"
    if [[ "$DRY_RUN" == false ]]; then
      eval "$cmd"
    else
      echo "  (dry-run) $cmd"
    fi
  done
done

# ── Step 2: H_total ranking ───────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  MODULE 4 — Step 2/4: H_total ranking"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [[ "$DRY_RUN" == false ]]; then
  python - <<'PYEOF'
import json, sys
sys.path.insert(0, "src")
from eval.evaluator import batch_evaluate
from eval.h_total import rank_models_by_h_total

results = batch_evaluate("results/module4")

# Build per-model dimension F1 dict
model_dim_f1 = {}
for model, bench_reports in results.items():
    model_dim_f1[model] = {
        "existence_f1": bench_reports.get("xpope_existence", {}).get("f1"),
        "attribute_f1": bench_reports.get("xpope_attribute", {}).get("f1"),
        "relation_f1":  bench_reports.get("xpope_relation",  {}).get("f1"),
    }

ranking = rank_models_by_h_total(model_dim_f1)
print("\nH_total Ranking:")
for rank, (model, score) in enumerate(ranking, 1):
    print(f"  #{rank}  {model:<30}  H_total={score:.4f}")

with open("reports/module4_htotal.json", "w") as f:
    json.dump({
        "ranking": [{"model": m, "h_total": s} for m, s in ranking],
        "model_dim_f1": model_dim_f1,
    }, f, indent=2)
print("\n[saved] reports/module4_htotal.json")
PYEOF
else
  echo "(dry-run) H_total ranking → reports/module4_htotal.json"
fi

# ── Step 3: Radar chart ───────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  MODULE 4 — Step 3/4: Radar chart"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [[ "$DRY_RUN" == false ]]; then
  python - <<'PYEOF'
import json
with open("reports/module4_htotal.json") as f:
    data = json.load(f)
radar_input = {
    model: {
        "existence": dims.get("existence_f1") or 0.0,
        "attribute": dims.get("attribute_f1") or 0.0,
        "relation":  dims.get("relation_f1")  or 0.0,
    }
    for model, dims in data["model_dim_f1"].items()
}
with open("reports/module4_radar_input.json", "w") as f:
    json.dump(radar_input, f, indent=2)
PYEOF

  python -m src.viz.plot_radar "reports/module4_radar_input.json" \
    --output "$FIGURES_DIR/module4_radar.png" \
    --title "Module 4 — X-POPE: Hallucination by Dimension"
else
  echo "(dry-run) plot_radar → $FIGURES_DIR/module4_radar.png"
fi

# ── Step 4: Yes-bias on X-POPE splits ────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  MODULE 4 — Step 4/4: Yes-bias on X-POPE splits"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [[ "$DRY_RUN" == false ]]; then
  python -m src.analysis.bias_analysis "$RESULTS_DIR" \
    --benchmarks xpope_existence xpope_attribute xpope_relation \
    --json > "$REPORTS_DIR/module4_bias.json"
  echo "[saved] $REPORTS_DIR/module4_bias.json"

  python -m src.viz.plot_bias "$REPORTS_DIR/module4_bias.json" \
    --output "$FIGURES_DIR/module4_bias_bar.png" \
    --title "Module 4 — Yes-Bias on X-POPE (existence/attribute/relation)"
else
  echo "(dry-run) bias_analysis → $REPORTS_DIR/module4_bias.json"
  echo "(dry-run) plot_bias     → $FIGURES_DIR/module4_bias_bar.png"
fi

echo ""
echo "✅  Module 4 complete."
echo "    H_total ranking : $REPORTS_DIR/module4_htotal.json"
echo "    Radar chart     : $FIGURES_DIR/module4_radar.png"
echo "    Bias chart      : $FIGURES_DIR/module4_bias_bar.png"
