"""
Fine-tunes YOLOv8n on the prepared HaGRID hand-detection dataset.

Run AFTER prepare_yolo_hands.py has finished.

Output:
    gesture_drone/models/yolo_hands/weights/best.pt   ← use this in inference
    gesture_drone/models/yolo_hands/weights/last.pt
    gesture_drone/models/yolo_hands/results.png        ← training curves
"""

from pathlib import Path
import torch
from ultralytics import YOLO

SCRIPT_DIR   = Path(__file__).resolve().parent
DATASET_YAML = SCRIPT_DIR.parent / "hagrid_detection" / "yolo_hands" / "dataset.yaml"
OUTPUT_DIR   = SCRIPT_DIR.parent / "models" / "yolo_hands"

# ── Training hyperparameters ────────────────────────────────────────────────
EPOCHS         = 50
IMGSZ          = 384   # native HaGRID-250k resolution — no upscaling waste
BATCH          = 32    # RTX 3070 8GB; safe headroom alongside caching
WORKERS        = 8     # confirmed optimal in benchmark_training.py
LR0            = 0.01  # standard YOLO starting LR
PATIENCE       = 10    # early stop if mAP50 doesn't improve for 10 epochs
WARMUP_EPOCHS  = 3     # gradual LR ramp-up avoids early instability


def main():
    if not DATASET_YAML.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: {DATASET_YAML}\n"
            "Run prepare_yolo_hands.py first."
        )

    device = "0" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if device == "0" else "CPU"

    print("=" * 60)
    print("  YOLO HAND DETECTOR TRAINING")
    print("=" * 60)
    print(f"  Device  : {gpu_name}")
    print(f"  Dataset : {DATASET_YAML}")
    print(f"  Epochs  : {EPOCHS}  (early stop patience={PATIENCE})")
    print(f"  Img size: {IMGSZ}px  |  Batch: {BATCH}  |  Workers: {WORKERS}")
    print(f"  Output  : {OUTPUT_DIR}")
    print()

    model = YOLO("yolov8n.pt")   # ~6MB base weights, downloads once

    model.train(
        data           = str(DATASET_YAML),
        epochs         = EPOCHS,
        imgsz          = IMGSZ,
        batch          = BATCH,
        workers        = WORKERS,
        device         = device,
        lr0            = LR0,
        cos_lr         = True,          # cosine LR decay — smoother convergence than linear
        warmup_epochs  = WARMUP_EPOCHS,
        patience       = PATIENCE,
        amp            = True,          # FP16 mixed precision — same as EfficientNet trainer
        cache          = False,          # disk cache caused RAM OOM at 216k images; not worth the risk
        project        = str(OUTPUT_DIR.parent),
        name           = OUTPUT_DIR.name,
        exist_ok       = True,
        verbose        = True,
        plots          = True,
    )

    best = OUTPUT_DIR / "weights" / "best.pt"
    print(f"\nTraining complete.")
    print(f"  Best weights : {best}")
    print(f"  Swap this path into load_hand_detector() in the three inference scripts.")


if __name__ == "__main__":
    main()
