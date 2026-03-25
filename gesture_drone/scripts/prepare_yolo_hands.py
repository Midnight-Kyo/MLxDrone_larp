"""
Converts the HaGRID-250k dataset into YOLO hand-detection format.

Input layout (after unzip):
    hagrid_detection/extracted/hagrid-sample-250k-384p/
        hagrid_250k/train_val_<gesture>/<uuid>.jpg
        ann_train_val/<gesture>.json

Output layout:
    hagrid_detection/yolo_hands/
        images/train/<uuid>.jpg
        images/val/<uuid>.jpg
        labels/train/<uuid>.txt
        labels/val/<uuid>.txt
        dataset.yaml

YOLO label format (one line per hand in the image):
    0 <x_center> <y_center> <width> <height>   (all normalized 0-1)

HaGRID annotation format:
    [x_topleft, y_topleft, width, height]       (normalized 0-1)

Conversion:
    x_center = x_topleft + width  / 2
    y_center = y_topleft + height / 2
"""

import json
import random
import shutil
from pathlib import Path

SEED       = 42
VAL_SPLIT  = 0.15      # 15% validation

BASE       = Path(__file__).resolve().parents[1] / "hagrid_detection" / "extracted" / "hagrid-sample-250k-384p"
IMG_ROOT   = BASE / "hagrid_250k"
ANN_ROOT   = BASE / "ann_train_val"
OUT_ROOT   = Path(__file__).resolve().parents[1] / "hagrid_detection" / "yolo_hands"

random.seed(SEED)


def hagrid_to_yolo(bbox):
    """[x_tl, y_tl, w, h] → [x_center, y_center, w, h] (all 0-1)."""
    x, y, w, h = bbox
    return x + w / 2, y + h / 2, w, h


def main():
    for split in ("train", "val"):
        (OUT_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

    ann_files = sorted(ANN_ROOT.glob("*.json"))
    if not ann_files:
        raise FileNotFoundError(f"No annotation JSONs found in {ANN_ROOT}")

    print(f"Found {len(ann_files)} gesture classes")

    total_train = total_val = 0

    for ann_path in ann_files:
        gesture = ann_path.stem           # e.g. "fist"
        img_dir = IMG_ROOT / f"train_val_{gesture}"
        if not img_dir.exists():
            print(f"  [skip] no image folder for {gesture}")
            continue

        with open(ann_path) as f:
            annotations = json.load(f)

        uuids = list(annotations.keys())
        random.shuffle(uuids)
        n_val = max(1, int(len(uuids) * VAL_SPLIT))
        val_set = set(uuids[:n_val])

        n_written = 0
        for uuid in uuids:
            img_path = img_dir / f"{uuid}.jpg"
            if not img_path.exists():
                continue

            entry  = annotations[uuid]
            bboxes = entry.get("bboxes", [])
            if not bboxes:
                continue

            split = "val" if uuid in val_set else "train"

            # Copy image
            dst_img = OUT_ROOT / "images" / split / f"{uuid}.jpg"
            shutil.copy2(img_path, dst_img)

            # Write label file (class 0 = hand for every bbox)
            dst_lbl = OUT_ROOT / "labels" / split / f"{uuid}.txt"
            lines = []
            for bbox in bboxes:
                if len(bbox) != 4:
                    continue
                xc, yc, w, h = hagrid_to_yolo(bbox)
                # Clamp to [0, 1] just in case of annotation noise
                xc = max(0.0, min(1.0, xc))
                yc = max(0.0, min(1.0, yc))
                w  = max(0.001, min(1.0, w))
                h  = max(0.001, min(1.0, h))
                lines.append(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

            if lines:
                dst_lbl.write_text("\n".join(lines))
                n_written += 1

        n_train = n_written - (sum(1 for u in uuids if u in val_set and (img_dir / f"{u}.jpg").exists()))
        if split == "val":
            total_val += n_val
        total_train += n_written
        print(f"  {gesture:<20} {n_written:>6} images processed")

    # Write dataset.yaml
    yaml_path = OUT_ROOT / "dataset.yaml"
    yaml_path.write_text(f"""\
path: {OUT_ROOT}
train: images/train
val:   images/val

nc: 1
names:
  0: hand
""")

    print(f"\nDataset ready at: {OUT_ROOT}")
    print(f"  Train images: {sum(1 for _ in (OUT_ROOT / 'images' / 'train').iterdir())}")
    print(f"  Val   images: {sum(1 for _ in (OUT_ROOT / 'images' / 'val').iterdir())}")
    print(f"  dataset.yaml: {yaml_path}")


if __name__ == "__main__":
    main()
