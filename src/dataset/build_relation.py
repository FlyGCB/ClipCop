"""
Build the relation split of X-POPE.

Input:  data/processed/vg_parsed/vg_relations_filtered.json
Output: data/processed/xpope_relation.jsonl

Each question:
{
    "question_id": "rel_0001",
    "image": "COCO_val2014_000000XXXXXX.jpg",
    "question": "Is the person sitting on the chair? Please answer Yes or No.",
    "label": "yes" | "no",
    "question_type": "relation",
    "subject_name": "person",
    "object_name": "chair",
    "relation": "sitting on",
    "negative_type": null | "reversed" | "wrong_pair",
    "coco_image_id": int,
}

Negative sample strategy:
  Type A (reversed)    - swap subject and object
                         e.g. person ON chair → "Is the chair on the person?"
  Type B (wrong_pair)  - use a real relation from image but with wrong object pair
                         e.g. person-chair + dog-mat → "Is the person on the mat?"
  Ratio: 50% Type A, 50% Type B
"""

import json
import random
import argparse
from pathlib import Path
from collections import defaultdict


random.seed(42)


# ── Config ────────────────────────────────────────────────────────────────────

TARGET_TOTAL      = 1800
TARGET_POS        = 900
TARGET_NEG_A      = 450   # reversed
TARGET_NEG_B      = 450   # wrong pair
MAX_PER_IMAGE     = 6

COCO_FILENAME = "COCO_val2014_{:012d}.jpg"


# ── Load ──────────────────────────────────────────────────────────────────────

def load_relations(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def group_by_image(records: list[dict]) -> dict[int, list[dict]]:
    groups = defaultdict(list)
    for r in records:
        if r["coco_image_id"] != -1:
            groups[r["coco_image_id"]].append(r)
    return groups


# ── Question builders ─────────────────────────────────────────────────────────

def build_question(subj: str, rel: str, obj: str) -> str:
    return (
        f"Is the {subj} {rel} the {obj} in the image? "
        f"Please answer Yes or No."
    )


def make_positive(record: dict, qid: int) -> dict:
    return {
        "question_id": f"rel_{qid:05d}",
        "image": COCO_FILENAME.format(record["coco_image_id"]),
        "question": build_question(
            record["subject_name"], record["relation"], record["object_name"]
        ),
        "label": "yes",
        "question_type": "relation",
        "subject_name": record["subject_name"],
        "object_name": record["object_name"],
        "relation": record["relation"],
        "negative_type": None,
        "coco_image_id": record["coco_image_id"],
    }


def make_negative_type_a(record: dict, qid: int) -> dict:
    """
    Type A: reversed relation.
    person ON chair → "Is the chair on the person?"
    """
    return {
        "question_id": f"rel_{qid:05d}",
        "image": COCO_FILENAME.format(record["coco_image_id"]),
        "question": build_question(
            record["object_name"], record["relation"], record["subject_name"]
        ),
        "label": "no",
        "question_type": "relation",
        "subject_name": record["object_name"],   # swapped
        "object_name": record["subject_name"],   # swapped
        "relation": record["relation"],
        "negative_type": "reversed",
        "coco_image_id": record["coco_image_id"],
    }


def make_negative_type_b(
    record: dict,
    image_records: list[dict],
    qid: int,
) -> dict | None:
    """
    Type B: wrong object pair.
    image has person-chair and dog-mat
    → ask "Is the person on the mat?"

    Subject stays the same, object comes from a different relation in the image.
    """
    candidates = [
        r for r in image_records
        if r["relationship_id"] != record["relationship_id"]
        and r["object_name"] != record["object_name"]
        and r["subject_name"] == record["subject_name"]
    ]

    # Fallback: any other relation with a different object
    if not candidates:
        candidates = [
            r for r in image_records
            if r["relationship_id"] != record["relationship_id"]
            and r["object_name"] != record["object_name"]
        ]

    if not candidates:
        return None

    donor = random.choice(candidates)
    return {
        "question_id": f"rel_{qid:05d}",
        "image": COCO_FILENAME.format(record["coco_image_id"]),
        "question": build_question(
            record["subject_name"], record["relation"], donor["object_name"]
        ),
        "label": "no",
        "question_type": "relation",
        "subject_name": record["subject_name"],
        "object_name": donor["object_name"],   # wrong object
        "relation": record["relation"],
        "negative_type": "wrong_pair",
        "coco_image_id": record["coco_image_id"],
    }


# ── Main builder ──────────────────────────────────────────────────────────────

def build_relation_split(
    records: list[dict],
    target_pos: int,
    target_neg_a: int,
    target_neg_b: int,
    max_per_image: int,
) -> list[dict]:

    by_image = group_by_image(records)
    image_ids = list(by_image.keys())
    random.shuffle(image_ids)

    pos_records  = []
    neg_a_records = []
    neg_b_records = []
    image_counts = defaultdict(int)

    for img_id in image_ids:
        img_records = by_image[img_id]

        for record in img_records:
            if image_counts[img_id] >= max_per_image:
                break

            if len(pos_records) < target_pos:
                pos_records.append(record)
                image_counts[img_id] += 1
            elif len(neg_a_records) < target_neg_a:
                neg_a_records.append(record)
                image_counts[img_id] += 1
            elif len(neg_b_records) < target_neg_b:
                neg_b_records.append((record, img_records))
                image_counts[img_id] += 1

        if (len(pos_records) >= target_pos and
                len(neg_a_records) >= target_neg_a and
                len(neg_b_records) >= target_neg_b):
            break

    # Build question dicts
    questions = []
    qid = 1

    for record in pos_records[:target_pos]:
        questions.append(make_positive(record, qid))
        qid += 1

    for record in neg_a_records[:target_neg_a]:
        questions.append(make_negative_type_a(record, qid))
        qid += 1

    for record, img_records in neg_b_records[:target_neg_b]:
        q = make_negative_type_b(record, img_records, qid)
        if q:
            questions.append(q)
            qid += 1

    random.shuffle(questions)
    return questions


# ── Stats ─────────────────────────────────────────────────────────────────────

def print_stats(questions: list[dict]):
    total = len(questions)
    pos   = sum(1 for q in questions if q["label"] == "yes")
    neg_a = sum(1 for q in questions if q["negative_type"] == "reversed")
    neg_b = sum(1 for q in questions if q["negative_type"] == "wrong_pair")

    rel_counts = defaultdict(int)
    for q in questions:
        rel_counts[q["relation"]] += 1

    image_counts = defaultdict(int)
    for q in questions:
        image_counts[q["coco_image_id"]] += 1

    print(f"\n── X-POPE Relation Split Statistics ──")
    print(f"  Total questions : {total:,}")
    print(f"  Positive (yes)  : {pos:,}")
    print(f"  Negative (no)   : {total - pos:,}")
    print(f"    Type A (reversed)   : {neg_a:,}")
    print(f"    Type B (wrong_pair) : {neg_b:,}")
    print(f"  Unique images   : {len(image_counts):,}")

    print(f"\n── Top 15 relations ──")
    for rel, c in sorted(rel_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"  {rel:<25} {c:>5,}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build X-POPE relation split")
    parser.add_argument(
        "--input",
        default="data/processed/vg_parsed/vg_relations_filtered.json",
    )
    parser.add_argument(
        "--output",
        default="data/processed/xpope_relation.jsonl",
    )
    parser.add_argument("--target-total", type=int, default=TARGET_TOTAL)
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    target_pos   = args.target_total // 2
    target_neg_a = args.target_total // 4
    target_neg_b = args.target_total // 4

    print(f"Loading {input_path} ...")
    records = load_relations(input_path)
    print(f"Loaded {len(records):,} relation records")

    print(f"\nBuilding relation split "
          f"(pos={target_pos}, neg_A={target_neg_a}, neg_B={target_neg_b}) ...")

    questions = build_relation_split(
        records=records,
        target_pos=target_pos,
        target_neg_a=target_neg_a,
        target_neg_b=target_neg_b,
        max_per_image=MAX_PER_IMAGE,
    )

    print_stats(questions)

    with open(output_path, "w") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(questions):,} questions → {output_path}")
    print("Next step: run src/dataset/build_existence.py")


if __name__ == "__main__":
    main()
