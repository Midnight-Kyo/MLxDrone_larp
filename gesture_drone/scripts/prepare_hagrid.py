"""
Extract HaGRID 512p parquet shards into dataset_cropped/ for retraining.

Maps HaGRID class names to our gesture labels:
    peace + peace_inverted  →  two_fingers
    fist                    →  fist
    palm                    →  open_palm
    like                    →  thumbs_up
    all other classes       →  unknown  (capped at MAX_UNKNOWN total)

The unknown class teaches the model to output "I don't know" instead of
forcing every input into one of the 4 drone commands. Images are saved as
hagrid_<original_class>_<n>.jpg so individual classes can be promoted to
a real command later just by moving those files to a new folder and retraining.

Run from WSL2:
    cd /home/kyo/Projects/MLxDrone_larp
    source venv/bin/activate
    python gesture_drone/scripts/prepare_hagrid.py
"""

import io
import random
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm

random.seed(42)

ROOT        = Path(__file__).resolve().parent.parent
PARQUET_DIR = ROOT / "external" / "hagrid_512p" / "data"
TRAIN_DIR   = ROOT / "dataset_cropped" / "train"
VAL_DIR     = ROOT / "dataset_cropped" / "val"

VAL_RATIO   = 0.20
MAX_UNKNOWN = 45_000   # cap unknown class to match the size of two_fingers

HAGRID_NAMES = [
    "call", "dislike", "fist", "four", "like", "mute", "ok", "one",
    "palm", "peace", "peace_inverted", "rock", "stop", "stop_inverted",
    "three", "three2", "two_up", "two_up_inverted",
]

# Classes that map to one of our 4 drone commands
TARGET_MAP = {
    HAGRID_NAMES.index("fist"):           "fist",
    HAGRID_NAMES.index("like"):           "thumbs_up",
    HAGRID_NAMES.index("palm"):           "open_palm",
    HAGRID_NAMES.index("peace"):          "two_fingers",
    HAGRID_NAMES.index("peace_inverted"): "two_fingers",
}

# Everything else goes to unknown (we build this dynamically)
UNKNOWN_IDS = {i for i in range(len(HAGRID_NAMES)) if i not in TARGET_MAP}

ALL_CLASSES = ("fist", "thumbs_up", "open_palm", "two_fingers", "unknown")

# Ensure output dirs exist
for split_dir in (TRAIN_DIR, VAL_DIR):
    for cls in ALL_CLASSES:
        (split_dir / cls).mkdir(parents=True, exist_ok=True)

parquet_files = sorted(PARQUET_DIR.glob("*.parquet"))
print(f"Found {len(parquet_files)} parquet shards")
print(f"Unknown distractor classes ({len(UNKNOWN_IDS)}): "
      f"{[HAGRID_NAMES[i] for i in sorted(UNKNOWN_IDS)]}")

# Count existing hagrid files to support resuming without overwriting
counters = defaultdict(int)
for cls in ALL_CLASSES:
    for split_dir in (TRAIN_DIR, VAL_DIR):
        existing = list((split_dir / cls).glob("hagrid_*.jpg"))
        counters[cls] += len(existing)

if any(counters.values()):
    print("Resuming — existing HaGRID files per class:", dict(counters))

saved   = defaultdict(int)
skipped = 0

for pq_path in parquet_files:
    table  = pq.read_table(pq_path, columns=["image", "label"])
    labels = table["label"].to_pylist()
    imgcol = table["image"]

    for i in tqdm(range(table.num_rows), desc=pq_path.name, leave=False):
        lid = labels[i]

        if lid in TARGET_MAP:
            cls          = TARGET_MAP[lid]
            hagrid_class = HAGRID_NAMES[lid]
        elif lid in UNKNOWN_IDS:
            if counters["unknown"] >= MAX_UNKNOWN:
                skipped += 1
                continue
            cls          = "unknown"
            hagrid_class = HAGRID_NAMES[lid]
        else:
            skipped += 1
            continue

        idx = counters[cls]
        counters[cls] += 1

        split_dir = VAL_DIR if random.random() < VAL_RATIO else TRAIN_DIR
        # Filename encodes the original HaGRID class — makes it easy to
        # promote a specific gesture to its own class later.
        dest = split_dir / cls / f"hagrid_{hagrid_class}_{idx:06d}.jpg"

        if dest.exists():
            saved[cls] += 1
            continue

        struct = imgcol[i].as_py()
        img    = Image.open(io.BytesIO(struct["bytes"])).convert("RGB")
        img.save(dest, quality=90)
        saved[cls] += 1

print("\nExtraction complete.")
print(f"{'Class':<15} {'Saved':>8}")
print("-" * 25)
for cls in ALL_CLASSES:
    print(f"{cls:<15} {saved[cls]:>8,}")
print("-" * 25)
print(f"{'TOTAL':<15} {sum(saved.values()):>8,}")
print(f"\nSkipped (already existed or unknown cap reached): {skipped:,}")
print(f"\nTo promote a distractor class later (e.g. 'one_finger'):")
print(f"  mkdir dataset_cropped/train/one_finger dataset_cropped/val/one_finger")
print(f"  mv dataset_cropped/train/unknown/hagrid_one_*.jpg dataset_cropped/train/one_finger/")
print(f"  mv dataset_cropped/val/unknown/hagrid_one_*.jpg   dataset_cropped/val/one_finger/")
print(f"  # then retrain")
