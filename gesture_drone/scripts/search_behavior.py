"""
SEARCH / FACE_LOCK autonomy — shared constants and face validity helpers.

Behavioral state machine runs in gesture_ros2_node.py; Windows bridge uses
the same helpers for face_ok / face_x_norm in TCP payloads and HUD.
"""

from __future__ import annotations

# --- v1 defaults (approved policy) ---
M_ACQUIRE = 8
M_LOSS = 18
OMEGA_SEARCH = 0.15
KP_LOCK = 0.35
EPS_X = 0.06
H_HOLD = 6
MAX_ANG_LOCK = 0.22
# Min face area as a fraction of frame pixels (rejects tiny false positives).
TAU_FACE_AREA_FRAC = 0.015
# Extra floor on YuNet score (detector already applies score_threshold).
TAU_FACE_SCORE = 0.72


def face_ok_and_x_norm(
    face_bbox,
    score: float,
    frame_w: int,
    frame_h: int,
) -> tuple[bool, float]:
    """
    Returns (face_ok, face_x_norm). x_norm = (cx - w/2) / (w/2), ~[-1, 1].
    """
    if face_bbox is None or frame_w < 1:
        return False, 0.0
    x1, y1, x2, y2 = face_bbox
    area = max(0, x2 - x1) * max(0, y2 - y1)
    if area < TAU_FACE_AREA_FRAC * float(frame_w * frame_h):
        return False, 0.0
    if score < TAU_FACE_SCORE:
        return False, 0.0
    cx = 0.5 * (x1 + x2)
    half_w = max(float(frame_w) * 0.5, 1.0)
    x_norm = (cx - frame_w * 0.5) / half_w
    return True, float(max(-1.5, min(1.5, x_norm)))
