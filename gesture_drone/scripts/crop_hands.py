"""
Batch-crop hands from the existing gesture dataset using MediaPipe HandLandmarker.

Creates a new dataset_cropped/ directory with the same structure as dataset/,
but each image contains only the hand region (with padding).

Run from WSL2:
    cd /home/kyo/Projects/MLxDrone_larp
    source venv/bin/activate
    python gesture_drone/scripts/crop_hands.py
"""

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "dataset"
CROPPED_DIR = ROOT / "dataset_cropped"
MODEL_PATH = ROOT / "models" / "hand_landmarker.task"

PADDING_RATIO = 0.25  # expand bounding box by 25% on each side


def create_landmarker():
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.25,
        min_hand_presence_confidence=0.25,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(options)


def crop_hand(img, landmarks, padding=PADDING_RATIO):
    """Crop hand region from image using landmark bounding box + padding."""
    h, w = img.shape[:2]
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]

    x1, y1 = min(xs), min(ys)
    x2, y2 = max(xs), max(ys)

    bw = x2 - x1
    bh = y2 - y1
    pad_x = bw * padding
    pad_y = bh * padding

    x1 = max(0, int(x1 - pad_x))
    y1 = max(0, int(y1 - pad_y))
    x2 = min(w, int(x2 + pad_x))
    y2 = min(h, int(y2 + pad_y))

    return img[y1:y2, x1:x2]


def main():
    landmarker = create_landmarker()

    total = 0
    detected = 0
    skipped = 0

    for split in ("train", "val"):
        for gesture_dir in sorted((DATASET_DIR / split).iterdir()):
            if not gesture_dir.is_dir():
                continue
            gesture = gesture_dir.name
            out_dir = CROPPED_DIR / split / gesture
            out_dir.mkdir(parents=True, exist_ok=True)

            imgs = sorted(gesture_dir.glob("*.jpg"))
            class_detected = 0

            for img_path in imgs:
                total += 1
                img = cv2.imread(str(img_path))
                if img is None:
                    skipped += 1
                    continue

                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect(mp_image)

                if result.hand_landmarks:
                    cropped = crop_hand(img, result.hand_landmarks[0])
                    if cropped.shape[0] >= 20 and cropped.shape[1] >= 20:
                        cv2.imwrite(str(out_dir / img_path.name), cropped)
                        detected += 1
                        class_detected += 1
                    else:
                        skipped += 1
                else:
                    skipped += 1

            print(f"  {split}/{gesture}: {class_detected}/{len(imgs)} hands detected")

    landmarker.close()

    print(f"\n{'=' * 50}")
    print(f"  Total images processed: {total}")
    print(f"  Hands detected & saved: {detected}")
    print(f"  Skipped (no hand/bad):  {skipped}")
    print(f"  Output: {CROPPED_DIR}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
