"""
Master script: build all three X-POPE splits in one command.

Usage:
    python -m src.dataset.build_xpope \
        --coco-dir data/raw/coco \
        --vg-dir data/raw/visual_genome \
        --output-dir data/processed

This runs in order:
    1. parse_vg.py       → data/processed/vg_parsed/
    2. build_existence.py → data/processed/xpope_existence.jsonl
    3. build_attribute.py → data/processed/xpope_attribute.jsonl
    4. build_relation.py  → data/processed/xpope_relation.jsonl

Then prints a combined summary.
"""

import json
import argparse
from pathlib import Path

from src.dataset.parse_vg import (
    load_coco_image_ids, load_vg_coco_mapping,
    parse_attributes, parse_relations,
)
from src.dataset.build_existence import build_existence_split
from src.dataset.build_attribute import load_attributes, build_attribute_split
from src.dataset.build_relation import load_relations, build_relation_split


def save_jsonl(questions: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")


def print_summary(existence, attribute, relation):
    total = len(existence) + len(attribute) + len(relation)

    print("\n" + "=" * 50)
    print("  X-POPE Dataset Summary")
    print("=" * 50)
    print(f"  Existence  : {len(existence):>5,} questions")
    print(f"  Attribute  : {len(attribute):>5,} questions")
    print(f"  Relation   : {len(relation):>5,} questions")
    print(f"  {'─'*30}")
    print(f"  Total      : {total:>5,} questions")

    all_images = (
        {q["coco_image_id"] for q in existence} |
        {q["coco_image_id"] for q in attribute} |
        {q["coco_image_id"] for q in relation}
    )
    print(f"  Unique images : {len(all_images):,}")

    for split_name, split in [
        ("existence", existence),
        ("attribute", attribute),
        ("relation",  relation),
    ]:
        pos = sum(1 for q in split if q["label"] == "yes")
        neg = len(split) - pos
        print(f"\n  {split_name}:")
        print(f"    pos={pos:,}  neg={neg:,}  ratio={pos/len(split):.2f}")

    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Build full X-POPE dataset")
    parser.add_argument("--coco-dir",   default="data/raw/coco")
    parser.add_argument("--vg-dir",     default="data/raw/visual_genome")
    parser.add_argument("--output-dir", default="data/processed")
    args = parser.parse_args()

    coco_dir   = Path(args.coco_dir)
    vg_dir     = Path(args.vg_dir)
    output_dir = Path(args.output_dir)
    vg_parsed  = output_dir / "vg_parsed"

    # ── Step 1: Parse VG ─────────────────────────────────────────────────────
    print("\n[1/4] Parsing Visual Genome...")
    coco_ids    = load_coco_image_ids(coco_dir)
    vg_coco_map = load_vg_coco_mapping(vg_dir)

    attr_records = parse_attributes(vg_dir, coco_ids, vg_coco_map)
    rel_records  = parse_relations(vg_dir, coco_ids, vg_coco_map)

    vg_parsed.mkdir(parents=True, exist_ok=True)
    attr_parsed_path = vg_parsed / "vg_attributes_filtered.json"
    rel_parsed_path  = vg_parsed / "vg_relations_filtered.json"

    with open(attr_parsed_path, "w") as f:
        json.dump(attr_records, f)
    with open(rel_parsed_path, "w") as f:
        json.dump(rel_records, f)

    print(f"  Saved {len(attr_records):,} attributes → {attr_parsed_path}")
    print(f"  Saved {len(rel_records):,} relations  → {rel_parsed_path}")

    # ── Step 2: Existence ─────────────────────────────────────────────────────
    print("\n[2/4] Building existence split...")
    existence = build_existence_split(coco_dir)
    exist_path = output_dir / "xpope_existence.jsonl"
    save_jsonl(existence, exist_path)
    print(f"  Saved {len(existence):,} questions → {exist_path}")

    # ── Step 3: Attribute ─────────────────────────────────────────────────────
    print("\n[3/4] Building attribute split...")
    attribute = build_attribute_split(
        records=attr_records,
        target_pos=1100,
        target_neg_a=550,
        target_neg_b=550,
        max_per_image=6,
    )
    attr_path = output_dir / "xpope_attribute.jsonl"
    save_jsonl(attribute, attr_path)
    print(f"  Saved {len(attribute):,} questions → {attr_path}")

    # ── Step 4: Relation ──────────────────────────────────────────────────────
    print("\n[4/4] Building relation split...")
    relation = build_relation_split(
        records=rel_records,
        target_pos=900,
        target_neg_a=450,
        target_neg_b=450,
        max_per_image=6,
    )
    rel_path = output_dir / "xpope_relation.jsonl"
    save_jsonl(relation, rel_path)
    print(f"  Saved {len(relation):,} questions → {rel_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print_summary(existence, attribute, relation)

    print("\nDone. Next step:")
    print("  python -m src.models.run_inference --model all --benchmark all")


if __name__ == "__main__":
    main()
