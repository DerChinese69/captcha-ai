import os
import json
import re
import csv
import random
from collections import Counter
from datetime import datetime

# =========================
# CONFIG
# =========================

SEED = 42
random.seed(SEED)

#CLASSES = [str(i) for i in range(10)] #0 to 9
CLASSES = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" #A to Z
LENGTH = 5

TARGET_PER_CLASS_PER_POSITION = 10000
TOTAL_SAMPLES = len(CLASSES) * TARGET_PER_CLASS_PER_POSITION

IMAGE_FORMAT = "png"
EPSILON = 1e-3

WIDTH = 192
HEIGHT = 64

# =========================
# ORDER ID & DIR
# =========================

ORDERS_ROOT = "data/orders"
os.makedirs(ORDERS_ROOT, exist_ok=True)

existing = [
    d for d in os.listdir(ORDERS_ROOT)
    if re.match(r"order_\d{4}", d)
]

if existing:
    nums = [int(d.split("_")[1]) for d in existing]
    next_num = max(nums) + 1
else:
    next_num = 1

ORDER_ID = f"order_{str(next_num).zfill(4)}"
BASE_DIR = os.path.join(ORDERS_ROOT, ORDER_ID)
os.makedirs(BASE_DIR, exist_ok=True)

# =========================
# INIT TRACKING
# =========================
pos_counts = {
    pos: {c: 0 for c in CLASSES}
    for pos in range(LENGTH)
}

labels = []

# =========================
# HELPER
# =========================
def weighted_choice(candidates, weights):
    total = sum(weights)
    r = random.uniform(0, total)
    cum = 0
    for c, w in zip(candidates, weights):
        cum += w
        if r <= cum:
            return c
    return candidates[-1]

# =========================
# GENERATE LABELS
# =========================
for _ in range(TOTAL_SAMPLES):
    label = ""

    for pos in range(LENGTH):
        deficits = {
            c: TARGET_PER_CLASS_PER_POSITION - pos_counts[pos][c]
            for c in CLASSES
        }

        candidates = [c for c in CLASSES if deficits[c] > 0]

        if not candidates:
            candidates = CLASSES

        weights = [deficits[c] + EPSILON for c in candidates]

        chosen = weighted_choice(candidates, weights)

        label += chosen
        pos_counts[pos][chosen] += 1

    labels.append(label)

# =========================
# SHUFFLE
# =========================
random.shuffle(labels)

# =========================
# WRITE CSV
# =========================
csv_path = os.path.join(BASE_DIR, "captcha_order.csv")

with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["sample_id", "label", "output_path"])

    for i, label in enumerate(labels, start=1):
        sample_id = str(i).zfill(6)
        output_path = f"images/{sample_id}.{IMAGE_FORMAT}"
        writer.writerow([sample_id, label, output_path])

# =========================
# SUMMARY STATS
# =========================
total_counter = Counter()
position_counter = {
    f"position_{i+1}": Counter()
    for i in range(LENGTH)
}

for label in labels:
    for pos, char in enumerate(label):
        total_counter[char] += 1
        position_counter[f"position_{pos+1}"][char] += 1

label_counter = Counter(labels)
duplicates = sum(1 for v in label_counter.values() if v > 1)
most_common = label_counter.most_common(5)

# =========================
# WRITE CONFIG JSON
# =========================
config = {
    "order_id": ORDER_ID,
    "timestamp": str(datetime.now()),
    "global_seed": SEED,

    "captcha": {
    "length": LENGTH,
    "charset": "digits",
    "classes": CLASSES,
    "width": WIDTH,
    "height": HEIGHT
    },

    "generation": {
        "target_per_class_per_position": TARGET_PER_CLASS_PER_POSITION,
        "total_samples": TOTAL_SAMPLES,
        "balancing": "position-wise soft balancing",
        "duplicates_allowed": True
    },

    "output": {
        "image_format": IMAGE_FORMAT,
        "base_path": f"data/raw/{ORDER_ID}/"
    }
}

with open(os.path.join(BASE_DIR, "order_config.json"), "w") as f:
    json.dump(config, f, indent=4)

# =========================
# WRITE SUMMARY JSON
# =========================
summary = {
    "order_id": ORDER_ID,
    "total_samples": TOTAL_SAMPLES,

    "class_counts_total": dict(total_counter),
    "class_counts_per_position": {
        k: dict(v) for k, v in position_counter.items()
    },

    "duplicates": {
        "num_duplicate_labels": duplicates,
        "most_common_labels": most_common
    }
}

with open(os.path.join(BASE_DIR, "order_summary.json"), "w") as f:
    json.dump(summary, f, indent=4)

print(f"Order generated at: {BASE_DIR}")