"""
Shared YOLO hand crop + diagnostics (used by gesture_bridge.py and simulate_drone.py).

Filters false positives (faces, furniture, large blobs) via YuNet face IoU + geometry + YOLO conf.

Performance: YOLO runs on a downscaled RGB copy (long edge capped) and maps boxes back to full
frame for smoothing + display; imgsz is reduced vs full-HD native inference.
"""

from __future__ import annotations

import cv2
import numpy as np

from yunet_face import bbox_iou

# Inference: YOLO minimum score to emit a candidate (low so diagnostics see top conf).
YOLO_INFER_CONF = 0.40
# Accept a candidate only if its score is at least this.
YOLO_MIN_CONF = 0.58
# Reject very elongated / very flat boxes (typical false positives: chair rails, monitors).
HAND_ASPECT_RATIO_MAX = 2.75
# Normalized area = box_area / frame_area.
HAND_AREA_MIN_FRAC = 0.0015
HAND_AREA_MAX_FRAC = 0.40

# Longest side of the image passed to YOLO (smaller → faster; bbox mapped back to full res).
MAX_INFER_SIDE_DEFAULT = 640
# Ultralytics letterbox size (320 default; 256 is noticeably faster on CPU).
YOLO_IMGSZ = 256


def _passes_geometry(x1, y1, x2, y2, fw: int, fh: int) -> tuple[bool, str]:
    bw = max(1e-6, float(x2 - x1))
    bh = max(1e-6, float(y2 - y1))
    ar = max(bw, bh) / min(bw, bh)
    if ar > HAND_ASPECT_RATIO_MAX:
        return False, "aspect"
    area_frac = (bw * bh) / float(fw * fh)
    if area_frac < HAND_AREA_MIN_FRAC:
        return False, "area_small"
    if area_frac > HAND_AREA_MAX_FRAC:
        return False, "area_large"
    return True, "ok"


def detect_hand(
    detector,
    frame_rgb: np.ndarray,
    smoother,
    face_xyxy=None,
    *,
    face_hand_iou_max: float = 0.22,
    padding_ratio: float = 0.08,
    max_infer_side: int = MAX_INFER_SIDE_DEFAULT,
    yolo_imgsz: int = YOLO_IMGSZ,
):
    """
    Returns (crop_rgb, bbox_xyxy | None, diagnostics dict).

    Runs YOLO on a resized copy when the frame exceeds ``max_infer_side`` on the long edge;
    hand bbox smoothing and returned crop use **full-resolution** coordinates / pixels.

    diagnostics keys:
      yolo_n, yolo_top_conf, yolo_pick_conf, face_iou (str),
      reject_stage (ok | no_yolo | no_passing_box | aspect | area_small | area_large |
                    smooth_none | crop_small)
    """
    fh, fw = frame_rgb.shape[:2]
    diag: dict = {
        "yolo_n": 0,
        "yolo_top_conf": 0.0,
        "yolo_pick_conf": 0.0,
        "face_iou": "",
        "reject_stage": "no_yolo",
    }

    sx = sy = 1.0
    infer_rgb = frame_rgb
    ifw, ifh = fw, fh
    if max(fw, fh) > max_infer_side:
        sc = max_infer_side / max(fw, fh)
        ifw, ifh = int(round(fw * sc)), int(round(fh * sc))
        infer_rgb = cv2.resize(
            frame_rgb, (ifw, ifh), interpolation=cv2.INTER_AREA
        )
        sx = fw / float(ifw)
        sy = fh / float(ifh)

    face_infer = None
    if face_xyxy is not None:
        fx1, fy1, fx2, fy2 = face_xyxy
        face_infer = (fx1 / sx, fy1 / sy, fx2 / sx, fy2 / sy)

    results = detector(
        infer_rgb, verbose=False, conf=YOLO_INFER_CONF, imgsz=yolo_imgsz
    )
    if not results or len(results[0].boxes) == 0:
        diag["reject_stage"] = "no_yolo"
        bbox = smoother.update(None)
        if bbox is None:
            diag["reject_stage"] = "smooth_none"
            return None, None, diag
        return _finish_crop(frame_rgb, bbox, diag, fw, fh)

    boxes = results[0].boxes
    confs = boxes.conf.cpu().numpy()
    diag["yolo_n"] = int(len(confs))
    diag["yolo_top_conf"] = float(np.max(confs))

    order = np.argsort(-confs)
    raw_bbox = None
    chosen_conf = None
    last_geom = "no_passing_box"

    for bi in order:
        cf = float(confs[int(bi)])
        if cf < YOLO_MIN_CONF:
            continue
        x1, y1, x2, y2 = boxes.xyxy[int(bi)].cpu().numpy()
        if face_infer is not None:
            if bbox_iou((x1, y1, x2, y2), face_infer) > face_hand_iou_max:
                continue
        ok_geo, reason = _passes_geometry(x1, y1, x2, y2, ifw, ifh)
        if not ok_geo:
            last_geom = reason
            continue
        bw, bh = x2 - x1, y2 - y1
        pad_x, pad_y = bw * padding_ratio, bh * padding_ratio
        raw_bbox_infer = (
            max(0.0, x1 - pad_x),
            max(0.0, y1 - pad_y),
            min(float(ifw), x2 + pad_x),
            min(float(ifh), y2 + pad_y),
        )
        raw_bbox = (
            raw_bbox_infer[0] * sx,
            raw_bbox_infer[1] * sy,
            raw_bbox_infer[2] * sx,
            raw_bbox_infer[3] * sy,
        )
        chosen_conf = cf
        diag["reject_stage"] = "ok"
        if face_infer is not None:
            diag["face_iou"] = f"{bbox_iou((x1, y1, x2, y2), face_infer):.3f}"
        break
    else:
        if raw_bbox is None:
            if diag["yolo_top_conf"] < YOLO_MIN_CONF:
                diag["reject_stage"] = "conf"
            else:
                diag["reject_stage"] = last_geom

    diag["yolo_pick_conf"] = float(chosen_conf or 0.0)

    bbox = smoother.update(raw_bbox)
    if bbox is None:
        diag["reject_stage"] = "smooth_none"
        return None, None, diag

    return _finish_crop(frame_rgb, bbox, diag, fw, fh)


def _finish_crop(frame_rgb, bbox, diag, fw, fh):
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    half = max(x2 - x1, y2 - y1) / 2
    x1 = max(0, int(cx - half))
    y1 = max(0, int(cy - half))
    x2 = min(fw, int(cx + half))
    y2 = min(fh, int(cy + half))
    cropped = frame_rgb[y1:y2, x1:x2]
    if cropped.shape[0] < 20 or cropped.shape[1] < 20:
        diag["reject_stage"] = "crop_small"
        return None, None, diag
    return cropped, (x1, y1, x2, y2), diag
