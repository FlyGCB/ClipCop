"""
Build the existence split of X-POPE.

Aligned with POPE adversarial strategy:
- Positive: objects that exist in the image (from COCO annotations)
- Negative: co-occurring objects that do NOT exist in the image
            (same adversarial sampling as POPE)

Input:  data/raw/coco/annotations/instances_val2014.json
Output: data/processed/xpope_existence.jsonl

Target: 3,000 questions, 1:1 pos/neg, same 500 images as POPE
"""

import json
import random
import argparse
from pathlib import Path
from collections import defaultdict


random.seed(42)

TARGET_TOTAL  = 3000
TARGET_POS    = 1500
TARGET_NEG    = 1500
MAX_PER_IMAGE = 6

COCO_FILENAME = "COCO_val2014_{:012d}.jpg"

# COCO 80 class names (id → name)
COCO_CLASSES = {
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane",
    6: "bus", 7: "train", 8: "truck", 9: "boat", 10: "traffic light",
    11: "fire hydrant", 13: "stop sign", 14: "parking meter", 15: "bench",
    16: "bird", 17: "cat", 18: "dog", 19: "horse", 20: "sheep",
    21: "cow", 22: "elephant", 23: "bear", 24: "zebra", 25: "giraffe",
    27: "backpack", 28: "umbrella", 31: "handbag", 32: "tie", 33: "suitcase",
    34: "frisbee", 35: "skis", 36: "snowboard", 37: "sports ball", 38: "kite",
    39: "baseball bat", 40: "baseball glove", 41: "skateboard", 42: "surfboard",
    43: "tennis racket", 44: "bottle", 46: "wine glass", 47: "cup",
    48: "fork", 49: "knife", 50: "spoon", 51: "bowl", 52: "banana",
    53: "apple", 54: "sandwich", 55: "orange", 56: "broccoli", 57: "carrot",
    58: "hot dog", 59: "pizza", 60: "donut", 61: "cake", 62: "chair",
    63: "couch", 64: "potted plant", 65: "bed", 67: "dining table",
    70: "toilet", 72: "tv", 73: "laptop", 74: "mouse", 75: "remote",
    76: "keyboard", 77: "cell phone", 78: "microwave", 79: "oven",
    80: "toaster", 81: "sink", 82: "refrigerator", 84: "book", 85: "clock",
    86: "vase", 87: "scissors", 88: "teddy bear", 89: "hair drier",
    90: "toothbrush",
}


def load_coco_annotations(coco_dir: Path):
    ann_file = coco_dir / "annotations" / "instances_val2014.json"
    with open(ann_file) as f:
        coco = json.load(f)

    # image_id → set of category names present
    image_objects = defaultdict(set)
    for ann in coco["annotations"]:
        cat_name = COCO_CLASSES.get(ann["category_id"])
        if cat_name:
            image_objects[ann["image_id"]].add(cat_name)

    return image_objects


def build_cooccurrence_matrix(image_objects: dict) -> dict[str, dict[str, int]]:
    """Build object co-occurrence counts (for adversarial negative sampling)."""
    cooccur = defaultdict(lambda: defaultdict(int))
    for objects in image_objects.values():
        obj_list = list(objects)
        for i, a in enumerate(obj_list):
            for b in obj_list[i+1:]:
                cooccur[a][b] += 1
                cooccur[b][a] += 1
    return cooccur


def get_adversarial_negative(
    present: set[str],
    all_classes: list[str],
    cooccur: dict,
) -> str | None:
    """
    Sample a negative object using adversarial strategy:
    prefer objects that frequently co-occur with present objects
    but are NOT in the image.
    """
    absent = [c for c in all_classes if c not in present]
    if not absent:
        return None

    # Score each absent object by co-occurrence with present objects
    scores = {}
    for obj in absent:
        score = sum(cooccur.get(p, {}).get(obj, 0) for p in present)
        scores[obj] = score

    # Sample from top-20 by score (with some randomness)
    top = sorted(scores, key=lambda x: -scores[x])[:20]
    return random.choice(top)


def build_existence_split(coco_dir: Path) -> list[dict]:
    print("Loading COCO annotations...")
    image_objects = load_coco_annotations(coco_dir)

    print("Building co-occurrence matrix...")
    cooccur = build_cooccurrence_matrix(image_objects)
    all_classes = list(COCO_CLASSES.values())

    image_ids = list(image_objects.keys())
    random.shuffle(image_ids)

    questions = []
    image_counts = defaultdict(int)
    qid = 1

    pos_count = 0
    neg_count = 0

    for img_id in image_ids:
        if pos_count >= TARGET_POS and neg_count >= TARGET_NEG:
            break

        present = image_objects[img_id]
        if not present:
            continue

        img_file = COCO_FILENAME.format(img_id)

        for obj in list(present):
            if image_counts[img_id] >= MAX_PER_IMAGE:
                break

            # Positive
            if pos_count < TARGET_POS:
                questions.append({
                    "question_id": f"exist_{qid:05d}",
                    "image": img_file,
                    "question": (
                        f"Is there a {obj} in the image? "
                        f"Please answer Yes or No."
                    ),
                    "label": "yes",
                    "question_type": "existence",
                    "object_name": obj,
                    "negative_type": None,
                    "coco_image_id": img_id,
                })
                pos_count += 1
                image_counts[img_id] += 1
                qid += 1

            if image_counts[img_id] >= MAX_PER_IMAGE:
                break

            # Negative (adversarial)
            if neg_count < TARGET_NEG:
                neg_obj = get_adversarial_negative(present, all_classes, cooccur)
                if neg_obj:
                    questions.append({
                        "question_id": f"exist_{qid:05d}",
                        "image": img_file,
                        "question": (
                            f"Is there a {neg_obj} in the image? "
                            f"Please answer Yes or No."
                        ),
                        "label": "no",
                        "question_type": "existence",
                        "object_name": neg_obj,
                        "negative_type": "adversarial",
                        "coco_image_id": img_id,
                    })
                    neg_count += 1
                    image_counts[img_id] += 1
                    qid += 1

    random.shuffle(questions)
    return questions


def main():
    parser = argparse.ArgumentParser(description="Build X-POPE existence split")
    parser.add_argument("--coco-dir", default="data/raw/coco")
    parser.add_argument("--output", default="data/processed/xpope_existence.jsonl")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    questions = build_existence_split(Path(args.coco_dir))

    pos = sum(1 for q in questions if q["label"] == "yes")
    neg = sum(1 for q in questions if q["label"] == "no")
    imgs = len({q["coco_image_id"] for q in questions})

    print(f"\n── X-POPE Existence Split Statistics ──")
    print(f"  Total     : {len(questions):,}")
    print(f"  Positive  : {pos:,}")
    print(f"  Negative  : {neg:,}")
    print(f"  Images    : {imgs:,}")

    with open(output_path, "w") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    print(f"\nSaved → {output_path}")


if __name__ == "__main__":
    main()
