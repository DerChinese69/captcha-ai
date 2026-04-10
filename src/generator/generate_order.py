"""
generate_order.py — Generate a balanced CAPTCHA order manifest.

Creates a CSV of (sample_id, label, output_path) entries with exact
per-position class quotas enforced, plus a JSON config and summary.
Output is written under data/orders/<order_id>/.

Usage:
    python src/generator/generate_order.py \\
        --classes "ABCDEFGHIJKLMNOPQRSTUVWXYZ" \\
        --length 5 \\
        --target-per-class-per-position 1000

Arguments:
    --classes                       Exact character set to sample from
    --length                        Number of characters per CAPTCHA label
    --target-per-class-per-position Target count for each class at each position

After running, pass the generated order folder to render_captchas.php:
    php src/generator/render_captchas.php data/orders/<order_id> <dataset_name>
"""

import argparse
import csv
import json
import os
import random
import re
from collections import Counter
from datetime import datetime


ORDERS_ROOT = "data/orders"
IMAGE_FORMAT = "png"
WIDTH = 564
HEIGHT = 284
SEED = 42
EPSILON = 1e-3


def positive_int(value):
    parsed_value = int(value)
    if parsed_value <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive integer.")
    return parsed_value


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a balanced CAPTCHA order. "
            "You must explicitly provide classes, label length, "
            "and the target number of character occurrences per class at each position."
        )
    )
    parser.add_argument(
        "--classes",
        required=True,
        help='Classes to use for captcha generation. Example: "ABCDEFGHIJKLMNOPQRSTUVWXYZ".',
    )
    parser.add_argument(
        "--length",
        type=positive_int,
        required=True,
        help="Number of characters per captcha label.",
    )
    parser.add_argument(
        "--target-per-class-per-position",
        dest="target_per_class_per_position",
        type=positive_int,
        required=True,
        help="Target number of character occurrences for each class at each position.",
    )
    args = parser.parse_args()
    if not args.classes:
        parser.error("--classes must not be empty.")
    return args


def weighted_choice(candidates, weights):
    total = sum(weights)
    r = random.uniform(0, total)
    cum = 0
    for c, w in zip(candidates, weights):
        cum += w
        if r <= cum:
            return c
    return candidates[-1]


def classify_charset(classes):
    if classes == "0123456789":
        return "digits"
    if classes == "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        return "uppercase_letters"
    if classes == "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789":
        return "uppercase_alphanumeric"
    return "custom"


def next_order_id(orders_root):
    os.makedirs(orders_root, exist_ok=True)
    existing = [d for d in os.listdir(orders_root) if re.match(r"order_\d{4}", d)]
    if existing:
        nums = [int(d.split("_")[1]) for d in existing]
        return f"order_{str(max(nums) + 1).zfill(4)}"
    return "order_0001"


def generate_labels(classes, length, target_per_class_per_position):
    """Generate balanced CAPTCHA labels with exact per-position class quotas."""
    class_count = len(classes)
    total_labels = class_count * target_per_class_per_position
    pos_counts = {pos: {c: 0 for c in classes} for pos in range(length)}
    labels = []

    for _ in range(total_labels):
        label = ""
        for pos in range(length):
            deficits = {
                c: target_per_class_per_position - pos_counts[pos][c]
                for c in classes
            }
            candidates = [c for c in classes if deficits[c] > 0] or list(classes)
            weights = [deficits[c] + EPSILON for c in candidates]
            chosen = weighted_choice(candidates, weights)
            label += chosen
            pos_counts[pos][chosen] += 1
        labels.append(label)

    random.shuffle(labels)
    return labels


def main():
    args = parse_args()
    random.seed(SEED)

    classes = args.classes
    length = args.length
    target = args.target_per_class_per_position
    class_count = len(classes)
    total_samples = class_count * target

    order_id = next_order_id(ORDERS_ROOT)
    base_dir = os.path.join(ORDERS_ROOT, order_id)
    os.makedirs(base_dir, exist_ok=True)

    labels = generate_labels(classes, length, target)

    # Write order CSV
    csv_path = os.path.join(base_dir, "captcha_order.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "label", "output_path"])
        for i, label in enumerate(labels, start=1):
            sample_id = str(i).zfill(6)
            writer.writerow([sample_id, label, f"images/{sample_id}.{IMAGE_FORMAT}"])

    # Compute stats
    total_counter = Counter()
    position_counter = {f"position_{i+1}": Counter() for i in range(length)}
    for label in labels:
        for pos, char in enumerate(label):
            total_counter[char] += 1
            position_counter[f"position_{pos+1}"][char] += 1

    label_counter = Counter(labels)
    duplicates = sum(1 for v in label_counter.values() if v > 1)
    excess_duplicates = sum(v - 1 for v in label_counter.values() if v > 1)

    # Write config JSON
    config = {
        "order_id": order_id,
        "timestamp": str(datetime.now()),
        "global_seed": SEED,
        "captcha": {
            "length": length,
            "charset": classify_charset(classes),
            "classes": classes,
            "class_count": class_count,
            "width": WIDTH,
            "height": HEIGHT,
        },
        "generation": {
            "target_per_class_per_position": target,
            "total_samples": total_samples,
            "total_character_slots": total_samples * length,
            "expected_occurrences_per_class": length * target,
            "balancing": "position_quotas",
            "balancing_note": (
                "Exact per-position class quotas are enforced. Within each position, "
                "sampling is weighted by the remaining deficit for each class."
            ),
            "label_uniqueness_enforced": False,
        },
        "output": {
            "image_format": IMAGE_FORMAT,
            "base_path": f"data/raw/{order_id}/",
        },
    }
    with open(os.path.join(base_dir, "order_config.json"), "w") as f:
        json.dump(config, f, indent=4)

    # Write summary JSON
    summary = {
        "order_id": order_id,
        "total_samples": total_samples,
        "label_length": length,
        "class_count": class_count,
        "target_occurrences_per_class_per_position": target,
        "expected_occurrences_per_class": length * target,
        "class_counts_total": dict(total_counter),
        "class_counts_per_position": {k: dict(v) for k, v in position_counter.items()},
        "duplicates": {
            "num_duplicate_labels": duplicates,
            "num_excess_duplicate_samples": excess_duplicates,
            "most_common_labels": label_counter.most_common(5),
        },
    }
    with open(os.path.join(base_dir, "order_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    print(f"Order generated: {base_dir}")
    print(f"  {total_samples} samples  |  {class_count} classes  |  length {length}")
    print(f"  Next step: php src/generator/render_captchas.php {base_dir} <dataset_name>")


if __name__ == "__main__":
    main()
