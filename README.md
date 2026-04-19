# Beyond-POPE

A research project demonstrating that POPE — the most widely used VLM hallucination benchmark — is saturated, contains annotation errors, and only measures existence hallucination, masking true model failures on attribute and relation hallucination.

---

## Core Argument

| Problem | Evidence | Module |
|---|---|---|
| POPE is saturated | CV across 6 models ≤ 0.02 on adversarial split | Module 1 |
| Annotation errors distort rankings | Spearman ρ (POPE → RePOPE) < 0.7 | Module 2 |
| Harder benchmarks restore discriminability | CV recovers on DASH-B | Module 3 |
| Existence hallucination is only the tip | Attribute/relation F1 drops significantly | Module 4 |

**Conclusion**: POPE is neither accurate (annotation noise), sensitive (saturated), nor complete (dimension coverage). Beyond-POPE proposes a more reliable evaluation framework: RePOPE + DASH-B + X-POPE + H_total.

---

## Models Evaluated

| Model | Parameters | Source |
|---|---|---|
| Qwen2.5-VL | 7B | Qwen/Qwen2-VL-7B-Instruct |
| InternVL2.5 | 8B | OpenGVLab/InternVL2_5-8B |
| LLaVA-OneVision | 7B | lmms-lab/llava-onevision-qwen2-7b-ov |
| Llama-3.2-Vision | 11B | meta-llama/Llama-3.2-11B-Vision-Instruct |
| PaliGemma2 | 3B | google/paligemma2-3b-pt-448 |
| Idefics3 | 8B | HuggingFaceM4/Idefics3-8B-Llama3 |

---

## Project Structure

```
beyond-pope/
├── src/
│   ├── models/              Unified inference wrappers for all 6 models
│   │   ├── base.py          Abstract base: prompt templates, answer parser, batching
│   │   ├── qwen2vl.py
│   │   ├── internvl2.py
│   │   ├── llava_ov.py
│   │   ├── llama32v.py
│   │   ├── paligemma2.py
│   │   ├── idefics3.py
│   │   ├── __init__.py      MODEL_REGISTRY + get_model()
│   │   └── run_inference.py CLI entry point
│   │
│   ├── dataset/             Dataset construction
│   │   ├── parse_vg.py      Parse Visual Genome annotations
│   │   ├── build_existence.py
│   │   ├── build_attribute.py
│   │   ├── build_relation.py
│   │   ├── build_xpope.py   Build all X-POPE splits in one call
│   │   └── __init__.py
│   │
│   ├── eval/                Evaluation metrics
│   │   ├── metrics.py       accuracy / F1 / yes_bias / CV / per_category
│   │   ├── h_total.py       H_total weighted harmonic mean + model ranking
│   │   ├── evaluator.py     Evaluator + batch_evaluate + saturation_report
│   │   └── __init__.py
│   │
│   ├── analysis/            Statistical analysis
│   │   ├── ranking_shift.py Spearman rho across POPE → RePOPE → DASH-B
│   │   ├── bias_analysis.py Yes-bias distribution per model and benchmark
│   │   └── saturation_diag.py CV saturation diagnostics
│   │
│   ├── viz/                 Visualizations
│   │   ├── plot_radar.py    3-dimension radar chart (existence/attribute/relation)
│   │   ├── plot_ranking.py  Bump chart of ranking shifts
│   │   └── plot_bias.py     Diverging bar chart of yes-bias
│   │
│   └── experiments/         Experiment runner scripts
│       ├── run_module1.sh   POPE saturation + yes-bias
│       ├── run_module2.sh   RePOPE ranking shift
│       ├── run_module3.sh   DASH-B discriminability recovery
│       └── run_module4.sh   X-POPE three-dimension evaluation
│
├── data/
│   ├── raw/
│   │   ├── coco/val2014/    COCO val2014 images (~6GB)
│   │   └── benchmarks/
│   │       ├── pope/        pope_random/popular/adversarial.jsonl
│   │       ├── repope/      repope_adversarial.jsonl
│   │       └── dashb/       dashb_adversarial.jsonl
│   └── processed/
│       ├── xpope_existence.jsonl
│       ├── xpope_attribute.jsonl
│       └── xpope_relation.jsonl
│
├── results/
│   └── predictions/         Per-model per-benchmark JSONL prediction files
│
├── reports/                 JSON analysis outputs
├── figures/                 Generated PNG charts
└── venv/                    Local Python environment
```

---

## Setup

### 1. Clone

```bash
git clone git@github.com:FlyGCB/beyond-pope.git
cd beyond-pope
```

### 2. Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate einops timm sentencepiece
pip install scipy numpy matplotlib Pillow tqdm
```

### 3. Data

**COCO val2014 images** (~6GB):
```bash
mkdir -p data/raw/coco
wget -P data/raw/coco http://images.cocodataset.org/zips/val2014.zip
wget -P data/raw/coco http://images.cocodataset.org/annotations/annotations_trainval2014.zip
cd data/raw/coco && unzip val2014.zip && unzip annotations_trainval2014.zip
```

**POPE benchmark files**:
```bash
mkdir -p data/raw/benchmarks/pope
# Download + convert from POPE official repo
python3 scripts/convert_pope.py
```

**Visual Genome** (for X-POPE, Module 4):
```bash
mkdir -p data/raw/vg
wget -P data/raw/vg https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/objects.json.zip
wget -P data/raw/vg https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/attributes.json.zip
wget -P data/raw/vg https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/relationships.json.zip
cd data/raw/vg && unzip "*.zip"

# Build X-POPE splits
python -m src.dataset.build_xpope     --vg-dir data/raw/vg     --output-dir data/processed
```

---

## Running Experiments

Each module is self-contained and runs sequentially.

### Module 1 — POPE Saturation + Yes-bias
```bash
bash src/experiments/run_module1.sh
# Outputs: reports/module1_saturation.json, figures/module1_bias_bar.png
```

### Module 2 — RePOPE Ranking Shift
```bash
bash src/experiments/run_module2.sh
# Outputs: reports/module2_ranking.json, figures/module2_bump.png
```

### Module 3 — DASH-B Discriminability Recovery
```bash
bash src/experiments/run_module3.sh
# Outputs: reports/module3_ranking.json, reports/module3_saturation.json, figures/module3_bump.png
```

### Module 4 — X-POPE Three-Dimension Evaluation
```bash
bash src/experiments/run_module4.sh
# Outputs: reports/module4_htotal.json, figures/module4_radar.png, figures/module4_bias_bar.png
```

### Single model, single benchmark
```bash
python -m src.models.run_inference     --model qwen2vl_7b     --benchmark pope_adversarial     --image-dir data/raw/coco/val2014

# With 4-bit quantization (saves ~8GB VRAM)
python -m src.models.run_inference     --model qwen2vl_7b     --benchmark pope_adversarial     --image-dir data/raw/coco/val2014     --load-in-4bit
```

### All models, all benchmarks
```bash
python -m src.models.run_inference --model all --benchmark all     --image-dir data/raw/coco/val2014
```

---

## Evaluation

```python
from src.eval.evaluator import Evaluator, batch_evaluate
from src.eval.h_total import rank_models_by_h_total

# Single file
ev = Evaluator("results/predictions/qwen2vl_7b_pope_adversarial.jsonl")
print(ev.summary())

# All results in a directory
reports = batch_evaluate("results/predictions/")
```

---

## Analysis

```bash
# Ranking shift (Spearman rho)
python -m src.analysis.ranking_shift results/predictions/     --benchmarks pope_adversarial repope_adversarial dashb_adversarial

# Yes-bias distribution
python -m src.analysis.bias_analysis results/predictions/ --json > reports/bias.json

# CV saturation diagnostics
python -m src.analysis.saturation_diag results/predictions/     --benchmarks pope_adversarial dashb_adversarial
```

---

## Visualization

```bash
# Radar chart (X-POPE three dimensions)
python -m src.viz.plot_radar reports/module4_radar_input.json     --output figures/radar_xpope.png

# Bump chart (ranking shift)
python -m src.viz.plot_ranking reports/module3_ranking.json     --output figures/bump_ranking.png

# Yes-bias bar chart
python -m src.viz.plot_bias reports/module1_bias.json     --output figures/bias_bar.png
```

---

## Key Design Decisions

**Single prompt template across all models**: prevents prompt-engineering confounds — any performance difference reflects model capability, not prompt sensitivity.

**Conservative answer parser**: returns `unknown` rather than guessing when a model response is ambiguous. Unknown responses are excluded from all metrics and logged.

**H_total**: weighted harmonic mean of existence / attribute / relation F1. Missing dimensions are dropped and weights re-normalized, so models are never penalized for untested dimensions.

**CV as saturation metric**: coefficient of variation (std / mean) across model F1 scores. Low CV indicates benchmark cannot distinguish models. Threshold: CV ≤ 0.02 → saturated.

---

## Requirements

- Python 3.11+
- CUDA 12.1+ (tested on A100 80GB)
- ~80GB disk space for images + model weights
- ~40GB VRAM for full-precision inference (use `--load-in-4bit` for lower-end GPUs)

---

## License

MIT
