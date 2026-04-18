"""
Build the attribute split of X-POPE.

Input:  data/processed/vg_parsed/vg_attributes_filtered.json
Output: data/processed/xpope_attribute.jsonl

Each question:
{
    "question_id": "attr_0001",
    "image": "COCO_val2014_000000XXXXXX.jpg",
    "question": "Is the car red? Please answer Yes or No.",
    "label": "yes" | "no",
    "question_type": "attribute",
    "attribute_type": "color" | "material" | "size" | "shape",
    "object_name": "car",
    "attribute": "red",
    "negative_type": null | "wrong_attr" | "cross_object",
    "coco_image_id": int,
}

Negative sample strategy:
  Type A (wrong_attr)    - same object, different attribute from same category
                           e.g. car is red → "Is the car blue?"
  Type B (cross_object)  - attribute belongs to another object in same image
                           e.g. car is red, dog is white → "Is the car white?"
  Ratio: 50% Type A, 50% Type B  (equal split, both challenging)

Final dataset: 2,200 questions, 1:1 pos/neg
"""

import json
import random
import argparse
from pathlib import Path
from collections import defaultdict


random.seed(42)  # reproducible


# ── Config ────────────────────────────────────────────────────────────────────

TARGET_TOTAL      = 2200   # total questions in attribute split
TARGET_POS        = 1100   # positive (label=yes)
TARGET_NEG        = 1100   # negative (label=no)
TARGET_NEG_TYPE_A = 550    # wrong attribute, same category
TARGET_NEG_TYPE_B = 550    # attribute from another object in same image

# Max questions per image (to avoid one image dominating)
MAX_PER_IMAGE = 6

# COCO image filename template
COCO_FILENAME = "COCO_val2014_{:012d}.jpg"


# ── Load data ─────────────────────────────────────────────────────────────────

def load_attributes(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def group_by_image(records: list[dict]) -> dict[int, list[dict]]:
    """Group attribute records by coco_image_id."""
    groups = defaultdict(list)
    for r in records:
        if r["coco_image_id"] != -1:
            groups[r["coco_image_id"]].append(r)
    return groups


def group_by_attr_type(records: list[dict]) -> dict[str, list[str]]:
    """
    Build a lookup: {attribute_type: [all attributes of that type]}.
    Used to sample wrong attributes for Type A negatives.
    """
    lookup = defaultdict(set)
    for r in records:
        lookup[r["attribute_type"]].add(r["attribute"])
    return {k: list(v) for k, v in lookup.items()}


# ── Question builders ─────────────────────────────────────────────────────────

def build_question(obj: str, attr: str) -> str:
    return (
        f"Is the {obj} in the image {attr}? "
        f"Please answer Yes or No."
    )


def make_positive(record: dict, qid: int) -> dict:
    return {
        "question_id": f"attr_{qid:05d}",
        "image": COCO_FILENAME.format(record["coco_image_id"]),
        "question": build_question(record["object_name"], record["attribute"]),
        "label": "yes",
        "question_type": "attribute",
        "attribute_type": record["attribute_type"],
        "object_name": record["object_name"],
        "attribute": record["attribute"],
        "negative_type": None,
        "coco_image_id": record["coco_image_id"],
    }


def make_negative_type_a(
    record: dict,
    attr_pool: dict[str, list[str]],
    qid: int,
) -> dict | None:
    """
    Type A: same object, wrong attribute from same category.
    e.g. car is red → ask "Is the car blue?"
    """
    candidates = [
        a for a in attr_pool.get(record["attribute_type"], [])
        if a != record["attribute"]
    ]
    if not candidates:
        return None

    wrong_attr = random.choice(candidates)
    return {
        "question_id": f"attr_{qid:05d}",
        "image": COCO_FILENAME.format(record["coco_image_id"]),
        "question": build_question(record["object_name"], wrong_attr),
        "label": "no",
        "question_type": "attribute",
        "attribute_type": record["attribute_type"],
        "object_name": record["object_name"],
        "attribute": wrong_attr,
        "negative_type": "wrong_attr",
        "coco_image_id": record["coco_image_id"],
    }


def make_negative_type_b(
    record: dict,
    image_records: list[dict],
    qid: int,
) -> dict | None:
    """
    Type B: attribute from another object in the same image.
    e.g. image has red car + white dog → ask "Is the car white?"

    This is the hardest negative — the attribute exists in the image
    but belongs to a different object.
    """
    # Find other objects in the same image with different object names
    other = [
        r for r in image_records
        if r["object_id"] != record["object_id"]
        and r["object_name"] != record["object_name"]
        and r["attribute"] != record["attribute"]  # must be a different attr value
    ]
    if not other:
        return None

    donor = random.choice(other)
    return {
        "question_id": f"attr_{qid:05d}",
        "image": COCO_FILENAME.format(record["coco_image_id"]),
        "question": build_question(record["object_name"], donor["attribute"]),
        "label": "no",
        "question_type": "attribute",
        "attribute_type": donor["attribute_type"],
        "object_name": record["object_name"],
        "attribute": donor["attribute"],
        "negative_type": "cross_object",
        "coco_image_id": record["coco_image_id"],
    }


# ── Main builder ──────────────────────────────────────────────────────────────

def build_attribute_split(
    records: list[dict],
    target_pos: int,
    target_neg_a: int,
    target_neg_b: int,
    max_per_image: int,
) -> list[dict]:

    by_image = group_by_image(records)
    attr_pool = group_by_attr_type(records)
    image_ids = list(by_image.keys())
    random.shuffle(image_ids)

    positives = []
    neg_a_list = []
    neg_b_list = []
    image_counts = defaultdict(int)

    for img_id in image_ids:
        img_records = by_image[img_id]

        for record in img_records:
            if image_counts[img_id] >= max_per_image:
                break

            # Try to build positive
            if len(positives) < target_pos:
                positives.append(record)
                image_counts[img_id] += 1
                continue

            # Try to build Type A negative
            if len(neg_a_list) < target_neg_a:
                neg = make_negative_type_a(record, attr_pool, qid=0)
                if neg:
                    neg_a_list.append((record, "A"))
                    image_counts[img_id] += 1
                    continue

            # Try to build Type B negative
            if len(neg_b_list) < target_neg_b:
                neg = make_negative_type_b(record, img_records, qid=0)
                if neg:
                    neg_b_list.append((record, img_records, "B"))
                    image_counts[img_id] += 1
                    continue

        if (len(positives) >= target_pos and
                len(neg_a_list) >= target_neg_a and
                len(neg_b_list) >= target_neg_b):
            break

    # Now build actual question dicts with proper IDs
    questions = []
    qid = 1

    for record in positives[:target_pos]:
        questions.append(make_positive(record, qid))
        qid += 1

    for record, _ in neg_a_list[:target_neg_a]:
        q = make_negative_type_a(record, attr_pool, qid)
        if q:
            questions.append(q)
            qid += 1

    for record, img_records, _ in neg_b_list[:target_neg_b]:
        q = make_negative_type_b(record, img_records, qid)
        if q:
            questions.append(q)
            qid += 1

    # Shuffle final list
    random.shuffle(questions)
    return questions


# ── Stats ─────────────────────────────────────────────────────────────────────

def print_stats(questions: list[dict]):
    total = len(questions)
    pos = sum(1 for q in questions if q["label"] == "yes")
    neg = sum(1 for q in questions if q["label"] == "no")
    neg_a = sum(1 for q in questions if q["negative_type"] == "wrong_attr")
    neg_b = sum(1 for q in questions if q["negative_type"] == "cross_object")

    type_counts = defaultdict(int)
    for q in questions:
        type_counts[q["attribute_type"]] += 1

    obj_counts = defaultdict(int)
    for q in questions:
        obj_counts[q["object_name"]] += 1

    image_counts = defaultdict(int)
    for q in questions:
        image_counts[q["coco_image_id"]] += 1

    print(f"\n── X-POPE Attribute Split Statistics ──")
    print(f"  Total questions : {total:,}")
    print(f"  Positive (yes)  : {pos:,}")
    print(f"  Negative (no)   : {neg:,}")
    print(f"    Type A (wrong_attr)    : {neg_a:,}")
    print(f"    Type B (cross_object)  : {neg_b:,}")
    print(f"  Unique images   : {len(image_counts):,}")
    print(f"  Max q/image     : {max(image_counts.values()):,}")

    print(f"\n── By attribute type ──")
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {t:<12} {c:>5,}")

    print(f"\n── Top 10 objects ──")
    for obj, c in sorted(obj_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {obj:<20} {c:>5,}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build X-POPE attribute split")
    parser.add_argument(
        "--input",
        default="data/processed/vg_parsed/vg_attributes_filtered.json",
    )
    parser.add_argument(
        "--output",
        default="data/processed/xpope_attribute.jsonl",
    )
    parser.add_argument("--target-total", type=int, default=TARGET_TOTAL)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    target_pos   = args.target_total // 2
    target_neg_a = args.target_total // 4
    target_neg_b = args.target_total // 4

    print(f"Loading {input_path} ...")
    records = load_attributes(input_path)
    print(f"Loaded {len(records):,} attribute records")

    print(f"\nBuilding attribute split "
          f"(pos={target_pos}, neg_A={target_neg_a}, neg_B={target_neg_b}) ...")

    questions = build_attribute_split(
        records=records,
        target_pos=target_pos,
        target_neg_a=target_neg_a,
        target_neg_b=target_neg_b,
        max_per_image=MAX_PER_IMAGE,
    )

    print_stats(questions)

    # Save as JSONL
    with open(output_path, "w") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(questions):,} questions → {output_path}")
    print("Next step: run src/dataset/build_relation.py")


if __name__ == "__main__":
    main()
