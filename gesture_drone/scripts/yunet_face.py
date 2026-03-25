"""
OpenCV YuNet face detection (FaceDetectorYN) + proximity proxy for follow preview.

Larger face bbox in the image → closer (higher proximity). Not metric depth.

Requires OpenCV >= ~4.8 for ``face_detection_yunet_2023mar.onnx`` with FaceDetectorYN
(project pins ``opencv-python`` 4.13.x).
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import cv2
import numpy as np

# 2023mar replaces 2022mar in opencv_zoo; GitHub raw is Git LFS (pointer only).
YUNET_FILENAME = "face_detection_yunet_2023mar.onnx"
YUNET_URL = (
    "https://huggingface.co/opencv/face_detection_yunet/resolve/main/"
    f"{YUNET_FILENAME}"
)
_MIN_ONNX_BYTES = 10_000


def ensure_yunet_onnx(model_dir: Path) -> Path:
    """Download YuNet ONNX to model_dir if missing or corrupt (LFS pointer)."""
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / YUNET_FILENAME
    if path.exists() and path.stat().st_size >= _MIN_ONNX_BYTES:
        return path
    if path.exists():
        path.unlink()
    print(f"  Downloading YuNet ({YUNET_FILENAME})...")
    urllib.request.urlretrieve(YUNET_URL, path)
    if path.stat().st_size < _MIN_ONNX_BYTES:
        path.unlink()
        raise RuntimeError(
            f"YuNet download looks invalid (< {_MIN_ONNX_BYTES} bytes). "
            f"Place {YUNET_FILENAME} manually under {model_dir}."
        )
    print(f"  Saved to {path}")
    return path


def load_face_detector(model_dir: Path, score_threshold: float = 0.75):
    """
    Returns cv2.FaceDetectorYN instance. Call setInputSize + detect each frame.
    """
    onnx = ensure_yunet_onnx(model_dir)
    # Placeholder size; must call setInputSize((w,h)) before detect.
    detector = cv2.FaceDetectorYN.create(
        str(onnx),
        "",
        (320, 320),
        score_threshold=score_threshold,
        nms_threshold=0.3,
        top_k=5000,
    )
    return detector


def _largest_face_from_detections(faces, cw: int, ch: int):
    if faces is None or len(faces) == 0:
        return None, 0.0
    best = None
    best_area = 0.0
    best_score = 0.0
    for row in faces:
        x, y, bw, bh = float(row[0]), float(row[1]), float(row[2]), float(row[3])
        sc = float(row[-1]) if len(row) else 1.0
        area = bw * bh
        if area > best_area:
            best_area = area
            best_score = sc
            x1 = int(max(0, x))
            y1 = int(max(0, y))
            x2 = int(min(cw - 1, x + bw))
            y2 = int(min(ch - 1, y + bh))
            best = (x1, y1, x2, y2)
    return best, best_score


def detect_largest_face(
    detector,
    frame_bgr: np.ndarray,
    *,
    max_infer_side: int | None = None,
):
    """
    Returns (x1, y1, x2, y2) in **full-frame** coordinates (or None), and score.

    If ``max_infer_side`` is set and the frame is larger on the long edge, YuNet runs on a
    downscaled BGR copy and the box is mapped back — much faster at 720p+ on CPU.
    """
    h, w = frame_bgr.shape[:2]
    if max_infer_side is None or max(w, h) <= max_infer_side:
        detector.setInputSize((w, h))
        _, faces = detector.detect(frame_bgr)
        return _largest_face_from_detections(faces, w, h)

    sc = max_infer_side / max(w, h)
    nw, nh = int(round(w * sc)), int(round(h * sc))
    small = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    detector.setInputSize((nw, nh))
    _, faces = detector.detect(small)
    best, score = _largest_face_from_detections(faces, nw, nh)
    if best is None:
        return None, 0.0
    sx = w / float(nw)
    sy = h / float(nh)
    x1, y1, x2, y2 = best
    return (
        int(round(x1 * sx)),
        int(round(y1 * sy)),
        int(round(x2 * sx)),
        int(round(y2 * sy)),
    ), score


def bbox_iou(
    a: tuple[float, ...] | tuple[int, ...],
    b: tuple[float, ...] | tuple[int, ...],
) -> float:
    """Intersection-over-union for axis-aligned boxes (x1, y1, x2, y2)."""
    ax1, ay1, ax2, ay2 = (float(a[i]) for i in range(4))
    bx1, by1, bx2, by2 = (float(b[i]) for i in range(4))
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def proximity_from_bbox(
    bbox: tuple[int, int, int, int] | None, frame_w: int, frame_h: int
) -> float | None:
    """
    Unitless 0–100: larger normalized area → closer. None if no bbox.
    """
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    area = max(0, (x2 - x1) * (y2 - y1))
    area_norm = area / float(frame_w * frame_h)
    # sqrt compresses dynamic range; scale so typical face fills a useful band
    return float(min(100.0, 100.0 * np.sqrt(area_norm) * 6.28))


class ProximitySmoother:
    """EMA on scalar proximity."""

    def __init__(self, alpha: float = 0.35):
        self.alpha = alpha
        self._value: float | None = None

    def update(self, raw: float | None) -> float | None:
        if raw is None:
            self._value = None
            return None
        if self._value is None:
            self._value = raw
        else:
            self._value = self.alpha * raw + (1.0 - self.alpha) * self._value
        return self._value

    def reset(self):
        self._value = None
