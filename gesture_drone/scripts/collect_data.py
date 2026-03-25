r"""
Hand gesture dataset collector for YOLOv8 classification training.

Run from Windows PowerShell (for webcam access):
    python "\\wsl$\Ubuntu-22.04\home\kyo\Projects\MLxDrone_larp\gesture_drone\scripts\collect_data.py"

Controls:
    1/2/3/4  - Select gesture class
    SPACE    - Start/stop capturing frames
    Q        - Quit and print summary
"""

import cv2
import sys
import time
import random
from pathlib import Path

GESTURES = {
    ord("1"): "two_fingers",
    ord("2"): "fist",
    ord("3"): "open_palm",
    ord("4"): "thumbs_up",
}

GESTURE_LIST = ["two_fingers", "fist", "open_palm", "thumbs_up"]

CAPTURE_INTERVAL = 0.1  # seconds between saved frames
TRAIN_RATIO = 0.8

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR.parent / "dataset"


def setup_directories():
    for split in ("train", "val"):
        for gesture in GESTURE_LIST:
            (DATASET_DIR / split / gesture).mkdir(parents=True, exist_ok=True)


def count_existing():
    """Return dict of {gesture: {train: n, val: n}} for images already on disk."""
    counts = {}
    for gesture in GESTURE_LIST:
        counts[gesture] = {
            "train": len(list((DATASET_DIR / "train" / gesture).glob("*.jpg"))),
            "val": len(list((DATASET_DIR / "val" / gesture).glob("*.jpg"))),
        }
    return counts


def draw_hud(frame, gesture, capturing, counts):
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Dark banner at top
    cv2.rectangle(overlay, (0, 0), (w, 110), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # Gesture label
    color = (0, 255, 0) if capturing else (255, 255, 255)
    status = "RECORDING" if capturing else "PAUSED"
    cv2.putText(frame, f"Gesture: {gesture}", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, status, (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Frame counts
    train_n = counts[gesture]["train"]
    val_n = counts[gesture]["val"]
    total = train_n + val_n
    cv2.putText(frame, f"This gesture: {total}  (train={train_n} val={val_n})",
                (15, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    # Key hints at bottom
    cv2.putText(frame, "[1] two_fingers  [2] fist  [3] open_palm  [4] thumbs_up",
                (15, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    cv2.putText(frame, "[SPACE] start/stop   [Q] quit",
                (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    # Red recording dot
    if capturing:
        radius = 12
        cx, cy = w - 30, 30
        pulse = int(abs(time.time() % 1 - 0.5) * 2 * 255)
        cv2.circle(frame, (cx, cy), radius, (0, 0, max(150, pulse)), -1)


def print_summary(counts):
    print("\n" + "=" * 55)
    print("  DATASET COLLECTION SUMMARY")
    print("=" * 55)
    print(f"  {'Gesture':<15} {'Train':>7} {'Val':>7} {'Total':>7}")
    print("-" * 55)
    grand_train, grand_val = 0, 0
    for g in GESTURE_LIST:
        t, v = counts[g]["train"], counts[g]["val"]
        grand_train += t
        grand_val += v
        print(f"  {g:<15} {t:>7} {v:>7} {t + v:>7}")
    print("-" * 55)
    print(f"  {'TOTAL':<15} {grand_train:>7} {grand_val:>7} {grand_train + grand_val:>7}")
    print("=" * 55)


def find_camera():
    """Open camera by index using MSMF backend (best for Insta360 + virtual cams on Windows)."""
    cam_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    cap = cv2.VideoCapture(cam_index, cv2.CAP_MSMF)
    if cap.isOpened():
        ret, _ = cap.read()
        if ret:
            print(f"Using camera index {cam_index} (MSMF)")
            return cap
        cap.release()

    cap = cv2.VideoCapture(cam_index)
    if cap.isOpened():
        print(f"Using camera index {cam_index}")
        return cap

    print(f"ERROR: Camera index {cam_index} could not be opened.")
    print("Try a different index:  python collect_data.py 1")
    return None


def main():
    setup_directories()
    counts = count_existing()

    print(__doc__)

    cap = find_camera()
    if cap is None or not cap.isOpened():
        print("ERROR: Could not open webcam. Make sure no other app is using it.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    current_gesture = GESTURE_LIST[0]
    capturing = False
    last_save_time = 0.0

    print("Webcam opened. Controls are shown on the video feed.\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("ERROR: Failed to read frame from webcam.")
                break

            # Capture logic
            now = time.time()
            if capturing and (now - last_save_time) >= CAPTURE_INTERVAL:
                split = "train" if random.random() < TRAIN_RATIO else "val"
                timestamp = int(now * 1000)
                filename = f"{current_gesture}_{timestamp}.jpg"
                save_path = DATASET_DIR / split / current_gesture / filename
                cv2.imwrite(str(save_path), frame)
                counts[current_gesture][split] += 1
                last_save_time = now

            # Draw HUD and show
            draw_hud(frame, current_gesture, capturing, counts)
            cv2.imshow("Gesture Collector", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == ord("Q"):
                break
            elif key == ord(" "):
                capturing = not capturing
                state = "RECORDING" if capturing else "PAUSED"
                print(f"  [{state}] {current_gesture}")
            elif key in GESTURES:
                capturing = False
                current_gesture = GESTURES[key]
                print(f"  Selected: {current_gesture}")

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        counts = count_existing()
        print_summary(counts)


if __name__ == "__main__":
    main()
