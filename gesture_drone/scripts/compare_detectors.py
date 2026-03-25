"""
Side-by-side comparison of the old Bingsu hand detector vs the new HaGRID-trained detector.

Left panel  : old model (hand_yolov8n.pt, mAP50=0.767)
Right panel : new model (yolo_hands/weights/best.pt, mAP50=0.995)

Run on Windows:
    python "\\wsl$\Ubuntu-22.04\home\kyo\Projects\MLxDrone_larp\gesture_drone\scripts\compare_detectors.py" 2
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR  = SCRIPT_DIR.parent / "models"

CONF       = 0.50
IMGSZ      = 320


def load_models():
    # Old model
    old_path = MODEL_DIR / "hand_yolov8n.pt"
    if not old_path.exists():
        import shutil
        src = hf_hub_download("Bingsu/adetailer", "hand_yolov8n.pt")
        shutil.copy(src, old_path)
    old = YOLO(str(old_path))

    # New model
    new_path = MODEL_DIR / "yolo_hands" / "weights" / "best.pt"
    if not new_path.exists():
        raise FileNotFoundError(f"New model not found: {new_path}")
    new = YOLO(str(new_path))

    print(f"Old model : {old_path.name}  (mAP50=0.767 on adetailer test set)")
    print(f"New model : {new_path.name}  (mAP50=0.995 on HaGRID val set)")
    return old, new


def run_detector(model, frame_rgb):
    t0 = time.perf_counter()
    results = model(frame_rgb, verbose=False, conf=CONF, imgsz=IMGSZ)
    ms = (time.perf_counter() - t0) * 1000
    boxes = []
    if results and len(results[0].boxes) > 0:
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            boxes.append((int(x1), int(y1), int(x2), int(y2), conf))
    return boxes, ms


def draw_panel(frame_bgr, boxes, label, ms, color):
    panel = frame_bgr.copy()
    h, w = panel.shape[:2]

    for x1, y1, x2, y2, conf in boxes:
        cv2.rectangle(panel, (x1, y1), (x2, y2), color, 2)
        cv2.putText(panel, f"{conf:.2f}", (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Header bar
    cv2.rectangle(panel, (0, 0), (w, 50), (20, 20, 20), -1)
    cv2.putText(panel, label, (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    det_text = f"{len(boxes)} hand{'s' if len(boxes) != 1 else ''}  |  {ms:.0f}ms"
    cv2.putText(panel, det_text, (10, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    return panel


def main():
    cam_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    cap = cv2.VideoCapture(cam_index, cv2.CAP_MSMF)
    if not cap.isOpened():
        cap = cv2.VideoCapture(cam_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    cap.set(cv2.CAP_PROP_FPS, 30)

    old_model, new_model = load_models()
    print("\nPress Q to quit.")

    fps_smooth = 0.0
    prev_time  = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()
        dt = now - prev_time
        fps_smooth = 0.9 * fps_smooth + 0.1 * (1.0 / max(dt, 1e-6))
        prev_time = now

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        old_boxes, old_ms = run_detector(old_model, frame_rgb)
        new_boxes, new_ms = run_detector(new_model, frame_rgb)

        old_panel = draw_panel(frame, old_boxes,
                               "OLD  Bingsu  mAP50=76.7%", old_ms,
                               (80, 80, 255))
        new_panel = draw_panel(frame, new_boxes,
                               "NEW  HaGRID  mAP50=99.5%", new_ms,
                               (80, 220, 80))

        sep = np.full((frame.shape[0], 4, 3), 60, dtype=np.uint8)
        combined = np.hstack([old_panel, sep, new_panel])

        # FPS overlay at bottom
        cv2.putText(combined, f"FPS: {fps_smooth:.0f}",
                    (combined.shape[1] - 100, combined.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        cv2.imshow("Detector Comparison  |  Old (left) vs New (right)", combined)
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
