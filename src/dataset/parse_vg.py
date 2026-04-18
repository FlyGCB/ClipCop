"""
Parse and filter Visual Genome annotations for X-POPE construction.

Downloads needed (put in data/raw/visual_genome/):
  attributes.json  - from http://visualgenome.org/static/data/dataset/attributes.json.zip
  relationships.json - from http://visualgenome.org/static/data/dataset/relationships.json.zip
  image_data.json  - from http://visualgenome.org/static/data/dataset/image_data.json.zip

Usage:
    python -m src.dataset.parse_vg \
        --vg-dir data/raw/visual_genome \
        --coco-dir data/raw/coco \
        --output-dir data/processed/vg_parsed
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict


# ── Filter config ─────────────────────────────────────────────────────────────

# Only keep these attribute types (objective, verifiable)
VALID_ATTRIBUTE_TYPES = {
    "color":    {"red", "blue", "green", "yellow", "white", "black", "brown",
                 "orange", "purple", "pink", "gray", "grey", "silver", "gold",
                 "beige", "tan", "navy", "dark", "light"},
    "material": {"wooden", "metal", "metallic", "plastic", "glass", "concrete",
                 "stone", "brick", "leather", "fabric", "cloth", "paper",
                 "rubber", "steel", "iron", "wood"},
    "size":     {"large", "small", "big", "tiny", "huge", "tall", "short",
                 "wide", "narrow", "long", "little"},
    "shape":    {"round", "circular", "square", "rectangular", "oval",
                 "triangular", "flat", "curved", "straight"},
}

# Flatten to a single set for quick lookup
ALL_VALID_ATTRS = {a for attrs in VALID_ATTRIBUTE_TYPES.values() for a in attrs}

# Subjective / ambiguous attributes to always exclude
SUBJECTIVE_ATTRS = {
    "beautiful", "pretty", "nice", "good", "bad", "old", "new", "young",
    "clean", "dirty", "bright", "dark", "open", "closed", "empty", "full",
    "happy", "sad", "calm", "busy", "simple", "complex", "natural",
}

# COCO 80 object classes (we only keep VG objects that match these)
COCO_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
}

# Valid VG relationship predicates (spatial / functional, not subjective)
VALID_RELATIONS = {
    # Spatial
    "on", "in", "on top of", "next to", "beside", "near", "above",
    "below", "behind", "in front of", "under", "underneath", "inside",
    "outside", "across from", "along", "at", "around",
    # Functional
    "holding", "wearing", "carrying", "sitting on", "standing on",
    "riding", "eating", "drinking", "using", "playing with",
    "attached to", "hanging from", "leaning on", "lying on",
}


# ── COCO image ID lookup ──────────────────────────────────────────────────────

def load_coco_image_ids(coco_dir: Path) -> set[int]:
    """
    Return the set of COCO val2014 image IDs.
    We only keep VG images that appear in COCO val2014.
    """
    ann_file = coco_dir / "annotations" / "instances_val2014.json"
    if not ann_file.exists():
        print(f"[warn] COCO annotations not found at {ann_file}")
        print("       Skipping COCO filter — all VG images will be used.")
        return set()

    with open(ann_file) as f:
        coco = json.load(f)

    ids = {img["id"] for img in coco["images"]}
    print(f"Loaded {len(ids):,} COCO val2014 image IDs")
    return ids


def load_vg_coco_mapping(vg_dir: Path) -> dict[int, int]:
    """
    Returns {vg_image_id: coco_image_id} for images that appear in both.
    Requires image_data.json from Visual Genome.
    """
    image_data_file = vg_dir / "image_data.json"
    if not image_data_file.exists():
        print(f"[warn] image_data.json not found — skipping COCO alignment")
        return {}

    with open(image_data_file) as f:
        image_data = json.load(f)

    mapping = {}
    for img in image_data:
        if img.get("coco_id"):
            mapping[img["image_id"]] = img["coco_id"]

    print(f"VG→COCO mapping: {len(mapping):,} images have COCO IDs")
    return mapping


# ── Attribute parser ──────────────────────────────────────────────────────────

def parse_attributes(vg_dir: Path, coco_image_ids: set[int],
                     vg_coco_map: dict[int, int]) -> list[dict]:
    """
    Parse VG attributes.json and return clean attribute records.

    Each record:
    {
        "vg_image_id": int,
        "coco_image_id": int,          # -1 if not in COCO
        "object_name": str,            # normalized COCO class name
        "attribute": str,              # e.g. "red"
        "attribute_type": str,         # "color" / "material" / "size" / "shape"
        "object_id": int,
        "bbox": [x, y, w, h],
    }
    """
    attr_file = vg_dir / "attributes.json"
    print(f"Loading {attr_file} ...")
    with open(attr_file) as f:
        raw = json.load(f)

    records = []
    skipped = defaultdict(int)

    for image in raw:
        vg_id = image["image_id"]
        coco_id = vg_coco_map.get(vg_id, -1)

        # Only keep images that are in COCO val2014 (if we have the mapping)
        if coco_image_ids and coco_id not in coco_image_ids:
            skipped["not_in_coco"] += 1
            continue

        for obj in image.get("attributes", []):
            obj_name = obj.get("names", [""])[0].lower().strip()

            # Normalize to COCO class
            coco_class = _match_coco_class(obj_name)
            if coco_class is None:
                skipped["no_coco_class"] += 1
                continue

            attrs = obj.get("attributes", [])
            if not attrs:
                skipped["no_attributes"] += 1
                continue

            for attr in attrs:
                attr = attr.lower().strip()

                # Skip subjective
                if attr in SUBJECTIVE_ATTRS:
                    skipped["subjective"] += 1
                    continue

                # Must be in our valid set
                attr_type = _get_attribute_type(attr)
                if attr_type is None:
                    skipped["invalid_attr"] += 1
                    continue

                bbox = obj.get("x", None)
                if bbox is not None:
                    bbox = [obj["x"], obj["y"], obj["w"], obj["h"]]

                records.append({
                    "vg_image_id": vg_id,
                    "coco_image_id": coco_id,
                    "object_name": coco_class,
                    "attribute": attr,
                    "attribute_type": attr_type,
                    "object_id": obj["object_id"],
                    "bbox": bbox,
                })

    print(f"Parsed {len(records):,} valid attribute records")
    print(f"Skipped: { {k: v for k, v in skipped.items()} }")
    return records


# ── Relation parser ───────────────────────────────────────────────────────────

def parse_relations(vg_dir: Path, coco_image_ids: set[int],
                    vg_coco_map: dict[int, int]) -> list[dict]:
    """
    Parse VG relationships.json and return clean relation records.

    Each record:
    {
        "vg_image_id": int,
        "coco_image_id": int,
        "subject_name": str,
        "object_name": str,
        "relation": str,
        "subject_id": int,
        "object_id": int,
        "relationship_id": int,
    }
    """
    rel_file = vg_dir / "relationships.json"
    print(f"Loading {rel_file} ...")
    with open(rel_file) as f:
        raw = json.load(f)

    records = []
    skipped = defaultdict(int)

    for image in raw:
        vg_id = image["image_id"]
        coco_id = vg_coco_map.get(vg_id, -1)

        if coco_image_ids and coco_id not in coco_image_ids:
            skipped["not_in_coco"] += 1
            continue

        for rel in image.get("relationships", []):
            predicate = rel.get("predicate", "").lower().strip()

            if predicate not in VALID_RELATIONS:
                skipped["invalid_relation"] += 1
                continue

            subj_name = rel["subject"].get("names", [""])[0].lower().strip()
            obj_name = rel["object"].get("names", [""])[0].lower().strip()

            subj_coco = _match_coco_class(subj_name)
            obj_coco = _match_coco_class(obj_name)

            if subj_coco is None or obj_coco is None:
                skipped["no_coco_class"] += 1
                continue

            # Skip self-relations
            if subj_coco == obj_coco:
                skipped["self_relation"] += 1
                continue

            records.append({
                "vg_image_id": vg_id,
                "coco_image_id": coco_id,
                "subject_name": subj_coco,
                "object_name": obj_coco,
                "relation": predicate,
                "subject_id": rel["subject"]["object_id"],
                "object_id": rel["object"]["object_id"],
                "relationship_id": rel["relationship_id"],
            })

    print(f"Parsed {len(records):,} valid relation records")
    print(f"Skipped: { {k: v for k, v in skipped.items()} }")
    return records


# ── Helper functions ──────────────────────────────────────────────────────────

def _match_coco_class(name: str) -> str | None:
    """
    Try to match a VG object name to a COCO class.
    Returns the COCO class name or None if no match.
    """
    name = name.lower().strip()

    # Direct match
    if name in COCO_CLASSES:
        return name

    # Partial match (e.g. "wooden chair" → "chair")
    for coco_class in COCO_CLASSES:
        if coco_class in name or name in coco_class:
            return coco_class

    return None


def _get_attribute_type(attr: str) -> str | None:
    """Return the attribute type category, or None if not valid."""
    for attr_type, attr_set in VALID_ATTRIBUTE_TYPES.items():
        if attr in attr_set:
            return attr_type
    return None


# ── Statistics ────────────────────────────────────────────────────────────────

def print_attribute_stats(records: list[dict]):
    type_counts = defaultdict(int)
    object_counts = defaultdict(int)
    attr_counts = defaultdict(int)

    for r in records:
        type_counts[r["attribute_type"]] += 1
        object_counts[r["object_name"]] += 1
        attr_counts[r["attribute"]] += 1

    print("\n── Attribute type distribution ──")
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {t:<12} {c:>6,}")

    print("\n── Top 15 objects ──")
    for obj, c in sorted(object_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"  {obj:<20} {c:>6,}")

    print("\n── Top 20 attributes ──")
    for attr, c in sorted(attr_counts.items(), key=lambda x: -x[1])[:20]:
        print(f"  {attr:<20} {c:>6,}")


def print_relation_stats(records: list[dict]):
    rel_counts = defaultdict(int)

    for r in records:
        rel_counts[r["relation"]] += 1

    print("\n── Top 20 relations ──")
    for rel, c in sorted(rel_counts.items(), key=lambda x: -x[1])[:20]:
        print(f"  {rel:<25} {c:>6,}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Parse Visual Genome for X-POPE")
    parser.add_argument("--vg-dir", default="data/raw/visual_genome")
    parser.add_argument("--coco-dir", default="data/raw/coco")
    parser.add_argument("--output-dir", default="data/processed/vg_parsed")
    args = parser.parse_args()

    vg_dir = Path(args.vg_dir)
    coco_dir = Path(args.coco_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load COCO image IDs for filtering
    coco_ids = load_coco_image_ids(coco_dir)
    vg_coco_map = load_vg_coco_mapping(vg_dir)

    # Parse attributes
    print("\n=== Parsing attributes ===")
    attr_records = parse_attributes(vg_dir, coco_ids, vg_coco_map)
    print_attribute_stats(attr_records)

    attr_out = output_dir / "vg_attributes_filtered.json"
    with open(attr_out, "w") as f:
        json.dump(attr_records, f)
    print(f"\nSaved {len(attr_records):,} attribute records → {attr_out}")

    # Parse relations
    print("\n=== Parsing relations ===")
    rel_records = parse_relations(vg_dir, coco_ids, vg_coco_map)
    print_relation_stats(rel_records)

    rel_out = output_dir / "vg_relations_filtered.json"
    with open(rel_out, "w") as f:
        json.dump(rel_records, f)
    print(f"\nSaved {len(rel_records):,} relation records → {rel_out}")

    print("\nDone. Next step: run src/dataset/build_attribute.py")


if __name__ == "__main__":
    main()
