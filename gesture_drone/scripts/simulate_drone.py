r"""
Drone flight simulator driven by real-time hand gesture recognition.

Left panel:  live webcam with MediaPipe hand detection + EfficientNet classification
Right panel: top-down 2D drone simulation responding to gesture commands

Gesture → Command mapping:
    two_fingers → FOLLOW_ARM (follow preview / no forward; YuNet when enabled)
    fist        → STOP / HOVER
    open_palm   → LAND
    thumbs_up   → MOVE UP

Run from Windows PowerShell:
    python "...\\simulate_drone.py" 2
    python "...\\simulate_drone.py" --source tello

  --source webcam|tello   Perception from PC webcam (default) or Tello camera (Wi‑Fi to drone).
  --world-width-m N       Fictional width of the top‑down panel in meters (default 10). Sets px→m scale.

Controls:
    Q  - Quit
    R  - Reset drone position
"""

import argparse
import csv
import datetime
import json
import math
import sys
import time
from pathlib import Path

import cv2
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import efficientnet_b0

import hand_detection
from hand_detection import detect_hand
from perception_gating import (
    TrustedHandConfig,
    TrustedHandGate,
    describe_gate_load_failure,
    format_trust_hud_line,
    load_hand_landmarker,
    perception_gate_wanted,
)
from yunet_face import (
    ProximitySmoother,
    detect_largest_face,
    load_face_detector,
    proximity_from_bbox,
)
from search_behavior import (
    EPS_X,
    H_HOLD,
    KP_LOCK,
    MAX_ANG_LOCK,
    M_ACQUIRE,
    M_LOSS,
    OMEGA_SEARCH,
    face_ok_and_x_norm,
)

# ── paths ────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR.parent / "models"
LOGS_DIR  = SCRIPT_DIR.parent / "logs"
GESTURE_MODEL_PATH = MODEL_DIR / "gesture_model.pt"
HAND_LANDMARK_PATH = MODEL_DIR / "hand_landmarker.task"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PADDING_RATIO = 0.08   # YOLO boxes already contain the full hand; less padding needed

CONFIDENCE_THRESHOLD = 0.85

# Conservative defaults: slow motion + fewer command edges (tune up after Gazebo/real tests).
COMMAND_COOLDOWN    = 1.2    # seconds between accepted command changes
GESTURE_LOCK_FRAMES = 8      # more frames before a gesture locks (was 6)
GESTURE_UNLOCK_FRAMES = 12  # harder to switch away from a locked gesture (was 10)

# Physical `tello_real_autonomy_v1.py` and optional `tello_view.py --autonomy-preview`
AUTONOMY_GESTURE_LOCK_FRAMES = 19
AUTONOMY_GESTURE_UNLOCK_FRAMES = 25

# 2D sim speeds (horizontal uses pixels; see SIM_WORLD_WIDTH_M for m/s readout)
SIM_MOVE_SPEED_PX_S = 28.0   # px/s across the top-down panel
SIM_ALT_SPEED_M_S   = 0.45   # m/s for simulated altitude (vertical axis)

# Fictional horizontal scale: treat the panel width SIM_W as this many meters.
# Forward speed in m/s ≈ SIM_MOVE_SPEED_PX_S * (SIM_WORLD_WIDTH_M / SIM_W).
SIM_WORLD_WIDTH_M = 10.0

# Webcam capture + YuNet scheduling (fewer pixels + half-rate face = higher FPS on CPU).
PERF_CAPTURE_W = 960
PERF_CAPTURE_H = 540
YUNET_MAX_INFER_SIDE = 480
# YuNet runs every frame for sim SEARCH/FACE_LOCK timing (matches design spec).
YUNET_FRAME_STRIDE = 1

# After peace sign locks, keep FOLLOW_ARM until explicit high-confidence exit or long no-hand.
FOLLOW_LATCH_ENABLE = True
FOLLOW_NO_HAND_FRAMES_TO_DROP = 45
FOLLOW_EXIT_CONF = 0.93
FOLLOW_EXIT_MARGIN = 0.18

GESTURE_TO_COMMAND = {
    "two_fingers": "FOLLOW_ARM",
    "fist":        "STOP",
    "open_palm":   "LAND",
    "thumbs_up":   "MOVE UP",
}

COMMAND_COLORS = {
    "FOLLOW_ARM":   (180, 220, 100),
    "STOP":         (0, 200, 255),
    "LAND":         (0, 0, 220),
    "MOVE UP":      (220, 180, 0),
    "IDLE":         (100, 100, 100),
    "LANDED":       (80, 80, 80),
}

# ── sim panel size ───────────────────────────────────────────────────────
SIM_W, SIM_H = 500, 720

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# ── perception (reused from test_model.py) ───────────────────────────────

def load_gesture_model():
    checkpoint = torch.load(str(GESTURE_MODEL_PATH), map_location=DEVICE, weights_only=False)
    class_names = checkpoint["class_names"]
    model = efficientnet_b0()
    model.classifier = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(1280, len(class_names)))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    print(f"Gesture model loaded ({len(class_names)} classes)")
    return model, class_names


def load_hand_detector():
    # Custom model trained on HaGRID-250k (mAP50=99.5%) — falls back to Bingsu if not found
    custom = MODEL_DIR / "yolo_hands" / "weights" / "best.pt"
    fallback = MODEL_DIR / "hand_yolov8n.pt"
    if custom.exists():
        model_path = custom
        print(f"  Using custom HaGRID hand detector ({model_path.name})")
    else:
        model_path = fallback
        if not model_path.exists():
            print("  Downloading fallback YOLOv8 hand detector...")
            import shutil
            src = hf_hub_download("Bingsu/adetailer", "hand_yolov8n.pt")
            shutil.copy(src, model_path)
        print(f"  Using fallback Bingsu hand detector ({model_path.name})")
    detector = YOLO(str(model_path))
    print("  YOLO hand detector ready")
    return detector


def find_camera(cam_index=0):
    cap = cv2.VideoCapture(cam_index, cv2.CAP_MSMF)
    if cap.isOpened():
        ret, _ = cap.read()
        if ret:
            print(f"Camera index {cam_index} (MSMF)")
            return cap
        cap.release()
    cap = cv2.VideoCapture(cam_index)
    if cap.isOpened():
        return cap
    print(f"ERROR: Camera {cam_index} not found.")
    return None


class GestureFilter:
    """
    Confidence-weighted, recency-biased sliding-window gesture filter with
    dead-band and asymmetric hysteresis.

    Each frame's vote is weighted by its classifier confidence AND by how
    recent it is (newest frame = 1.0, oldest = 0.5).  A gesture only
    becomes "confirmed" once it wins the weighted vote by a clear margin
    (min_vote_share) AND has led for a sufficient streak of frames.
    Switching away from an already-confirmed gesture requires a longer
    streak (unlock_frames) than initial lock-in (lock_frames), giving the
    system deliberate stickiness.
    """
    def __init__(self, window=10, lock_frames=GESTURE_LOCK_FRAMES,
                 unlock_frames=GESTURE_UNLOCK_FRAMES, min_vote_share=0.60):
        self._window         = window
        self._lock_frames    = lock_frames
        self._unlock_frames  = unlock_frames
        self._min_vote_share = min_vote_share
        self._history        = []   # list of (label, confidence)
        self._streak         = 0
        self._streak_cand    = None
        self._confirmed      = None
        self._lock_target    = lock_frames

    def update(self, gesture, confidence=1.0):
        """Feed one frame's result (gesture=None means no hand). Returns confirmed gesture."""
        label = gesture if gesture is not None else "__none__"
        conf  = float(confidence) if confidence is not None else 0.0
        self._history.append((label, conf))
        if len(self._history) > self._window:
            self._history.pop(0)

        n = len(self._history)
        # Weighted votes: recency 0.5 (oldest) → 1.0 (newest), scaled by confidence
        vote_totals  = {}
        total_weight = 0.0
        for i, (g, c) in enumerate(self._history):
            recency = 0.5 + 0.5 * (i / max(1, n - 1))
            w = c * recency
            vote_totals[g]  = vote_totals.get(g, 0.0) + w
            total_weight   += w

        winner       = max(vote_totals, key=vote_totals.get)
        winner_share = vote_totals[winner] / total_weight if total_weight > 0 else 0.0
        winner_gest  = winner if winner != "__none__" else None

        # Dead band: if the leading gesture doesn't hold a clear majority, hold still
        if winner_share < self._min_vote_share:
            self._streak      = 0
            self._streak_cand = None
            return self._confirmed

        # Hysteresis: switching away from a confirmed gesture is harder
        if self._confirmed is not None and winner_gest != self._confirmed:
            self._lock_target = self._unlock_frames
        else:
            self._lock_target = self._lock_frames

        # Streak counting
        if winner_gest == self._streak_cand:
            self._streak += 1
        else:
            self._streak_cand = winner_gest
            self._streak      = 1

        if self._streak >= self._lock_target:
            self._confirmed = winner_gest

        return self._confirmed

    @property
    def streak_ratio(self):
        """0.0–1.0 progress toward confirming the current streak candidate."""
        if self._lock_target == 0:
            return 0.0
        return min(1.0, self._streak / self._lock_target)

    @property
    def lock_target(self):
        return self._lock_target


class SessionLogger:
    """
    Writes one CSV row per frame to gesture_drone/logs/session_YYYYMMDD_HHMMSS.csv.
    Call close() (or use in a try/finally) to flush the file and print a summary.

    Extended columns support per-frame diagnosis (Excel / pandas): YOLO stats, reject path,
    classifier margin, full softmax JSON.
    """
    HEADER = [
        "timestamp_ms",
        "raw_gesture",
        "raw_confidence",
        "confirmed_gesture",
        "command_fired",
        "streak_ratio",
        "active_command",
        "yolo_n",
        "yolo_top_conf",
        "yolo_pick_conf",
        "face_iou",
        "reject_stage",
        "classifier_margin",
        "classifier_probs_json",
        "cmd_gesture",
    ]

    def __init__(self):
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self._path   = LOGS_DIR / f"session_{ts}.csv"
        self._file   = open(self._path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.HEADER)
        self._start          = time.time()
        self._frame_count    = 0
        self._flip_count     = 0
        self._prev_confirmed = None
        self._command_counts: dict = {}

    def log(
        self,
        raw_gesture: str,
        raw_confidence: float,
        confirmed_gesture,
        command_fired: str,
        streak_ratio: float,
        active_command: str = "",
        diag=None,
        classifier_margin: float | None = None,
        classifier_probs_json: str = "",
        cmd_gesture: str = "",
    ):
        ts_ms = int((time.time() - self._start) * 1000)
        d = diag or {}
        self._writer.writerow([
            ts_ms,
            raw_gesture,
            f"{raw_confidence:.3f}",
            confirmed_gesture or "",
            command_fired or "",
            f"{streak_ratio:.3f}",
            active_command,
            d.get("yolo_n", ""),
            f"{float(d.get('yolo_top_conf', 0)):.3f}",
            f"{float(d.get('yolo_pick_conf', 0)):.3f}",
            d.get("face_iou", ""),
            d.get("reject_stage", ""),
            f"{classifier_margin:.4f}" if classifier_margin is not None else "",
            classifier_probs_json,
            cmd_gesture,
        ])
        self._frame_count += 1

        if confirmed_gesture and confirmed_gesture != self._prev_confirmed:
            self._flip_count += 1
        self._prev_confirmed = confirmed_gesture

        if command_fired:
            self._command_counts[command_fired] = \
                self._command_counts.get(command_fired, 0) + 1

    def close(self):
        self._file.close()
        duration  = time.time() - self._start
        flip_rate = self._flip_count / duration if duration > 0 else 0.0
        cmds_str  = "  ".join(
            f"{k}={v}" for k, v in self._command_counts.items()
        ) or "none"
        print(f"\n--- Session summary ({self._frame_count} frames, {duration:.0f}s) ---")
        print(f"Confirmed flip rate : {flip_rate:.2f} flips/sec")
        print(f"Commands fired      : {cmds_str}")
        print(f"Log saved to        : {self._path}")


class BboxSmoother:
    """EMA-smooth the hand bbox and hold the last position for a few missed frames."""
    def __init__(self, alpha=0.4, max_miss_frames=8):
        self.alpha = alpha
        self.max_miss = max_miss_frames
        self._smooth = None
        self._misses = 0

    def update(self, raw_bbox):
        if raw_bbox is not None:
            self._misses = 0
            if self._smooth is None:
                self._smooth = tuple(float(v) for v in raw_bbox)
            else:
                self._smooth = tuple(
                    self.alpha * r + (1.0 - self.alpha) * s
                    for r, s in zip(raw_bbox, self._smooth)
                )
        else:
            self._misses += 1
            if self._misses > self.max_miss:
                self._smooth = None

        if self._smooth is None:
            return None
        return tuple(int(v) for v in self._smooth)


@torch.no_grad()
def classify_hand(model, hand_rgb, class_names: tuple | list):
    """Returns (class_index, top_confidence, margin, probs_json)."""
    tensor = preprocess(hand_rgb).unsqueeze(0).to(DEVICE)
    logits = model(tensor)
    probs = F.softmax(logits, dim=1)
    p = probs[0].detach().cpu().numpy()
    idx = int(np.argmax(p))
    conf = float(p[idx])
    sp = np.sort(p)[::-1]
    margin = float(sp[0] - sp[1]) if len(sp) > 1 else conf
    probs_dict = {class_names[i]: round(float(p[i]), 4) for i in range(len(class_names))}
    return idx, conf, margin, json.dumps(probs_dict, separators=(",", ":"))


# ── drone simulation state ───────────────────────────────────────────────

class DroneState:
    def __init__(self, world_width_m=SIM_WORLD_WIDTH_M):
        self.world_width_m = world_width_m
        self.reset()

    def reset(self):
        self.x = SIM_W // 2      # horizontal position (pixels)
        self.y = SIM_H // 2      # vertical position (top-down Y)
        self.altitude = 0.0      # meters
        self.heading = -90       # degrees, -90 = facing up on screen
        self.m_per_px = self.world_width_m / float(SIM_W)
        self.forward_speed_m_s = 0.0   # horizontal, m/s (derived from px motion)
        self.climb_speed_m_s = 0.0     # vertical, m/s
        self.is_flying = False
        self.trail = []
        self.command_log = []

    def update(self, command, dt):
        move_speed = SIM_MOVE_SPEED_PX_S
        alt_speed = SIM_ALT_SPEED_M_S

        if command == "MOVE FORWARD" and self.is_flying:
            rad = math.radians(self.heading)
            self.x += math.cos(rad) * move_speed * dt
            self.y += math.sin(rad) * move_speed * dt
            self.x = max(30, min(SIM_W - 30, self.x))
            self.y = max(80, min(SIM_H - 40, self.y))
            self.forward_speed_m_s = move_speed * self.m_per_px
            self.climb_speed_m_s = 0.0
            self.trail.append((int(self.x), int(self.y), time.time()))

        elif command == "FOLLOW_ARM" and self.is_flying:
            self.forward_speed_m_s = 0.0
            self.climb_speed_m_s = 0.0

        elif command == "MOVE UP" and self.is_flying:
            self.altitude = min(10.0, self.altitude + alt_speed * dt)
            self.forward_speed_m_s = 0.0
            self.climb_speed_m_s = alt_speed

        elif command == "STOP" and self.is_flying:
            self.forward_speed_m_s = 0.0
            self.climb_speed_m_s = 0.0

        elif command == "LAND":
            if self.is_flying:
                self.altitude = max(0.0, self.altitude - alt_speed * dt * 0.5)
                self.forward_speed_m_s = 0.0
                self.climb_speed_m_s = -alt_speed * 0.5
                if self.altitude <= 0.05:
                    self.altitude = 0.0
                    self.is_flying = False
                    self.climb_speed_m_s = 0.0

        # Auto-takeoff: if not flying and a flight command is received
        if not self.is_flying and command in ("MOVE FORWARD", "MOVE UP"):
            self.is_flying = True
            self.altitude = max(self.altitude, 0.5)
            self.log_command("AUTO TAKEOFF")

        if not self.is_flying:
            self.forward_speed_m_s = 0.0
            self.climb_speed_m_s = 0.0

    def log_command(self, cmd):
        self.command_log.append((time.time(), cmd))
        if len(self.command_log) > 8:
            self.command_log.pop(0)


def _norm_heading_deg(deg: float) -> float:
    """Map heading to [-180, 180) degrees."""
    x = (float(deg) + 180.0) % 360.0 - 180.0
    return x


# ── drawing functions ────────────────────────────────────────────────────

def draw_cam_panel(frame, raw_gesture, gesture, confidence, bbox, command, fps, gfilter=None,
                   source_label="", battery=None, temp=None,
                   follow_preview=False, face_proximity=None, face_tracked=False,
                   yunet_error=None, trust_line=None, beh_line=None):
    h, w = frame.shape[:2]
    hud_extra = (28 if trust_line else 0) + (28 if beh_line else 0)
    hud_h = (138 if follow_preview else 128) + hud_extra
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, hud_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    if source_label:
        cv2.putText(frame, source_label, (w - min(280, w - 10), 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 200, 160), 1)
    if battery is not None:
        bc = (0, 200, 0) if battery > 30 else (0, 80, 220)
        cv2.putText(frame, f"BAT {battery}%", (w - 100, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, bc, 1)
    if temp is not None:
        cv2.putText(frame, f"TMP {int(temp)}C", (w - 100, 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 140), 1)

    if gesture != "No hand":
        color = COMMAND_COLORS.get(command, (200, 200, 200))
        cv2.putText(frame, gesture, (15, 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(frame, f"{confidence * 100:.0f}%", (15, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, command, (15, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        if raw_gesture != gesture and raw_gesture != "No hand":
            cv2.putText(frame, f"raw: {raw_gesture}", (15, 112),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (130, 130, 130), 1)
    else:
        cv2.putText(frame, "No hand", (15, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 100, 100), 2)
        cv2.putText(frame, "IDLE", (15, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)

    cv2.putText(frame, f"FPS: {fps:.0f}", (w - 100, 96),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    if follow_preview:
        if yunet_error:
            fp_msg = f"FOLLOW preview  |  YuNet: {yunet_error}"
            fp_color = (0, 140, 255)
        elif face_tracked and face_proximity is not None:
            fp_msg = f"FOLLOW preview  |  PROX {face_proximity:.0f}/100"
            fp_color = (180, 220, 100)
        else:
            fp_msg = "FOLLOW preview  |  No face"
            fp_color = (160, 200, 255)
        cv2.putText(frame, fp_msg, (12, 132),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, fp_color, 1)

    if trust_line:
        cv2.putText(
            frame,
            trust_line[: min(95, len(trust_line))],
            (12, hud_h - (48 if beh_line else 26)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (160, 200, 160),
            1,
        )
    if beh_line:
        cv2.putText(
            frame,
            beh_line[: min(145, len(beh_line))],
            (12, hud_h - 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (200, 180, 255),
            1,
        )

    # Confirmation progress bar — fills as the gesture streak builds toward lock-in
    bar_y, bar_h = hud_h - 8, 8
    cv2.rectangle(frame, (0, bar_y), (w, bar_y + bar_h), (40, 40, 40), -1)
    if gfilter is not None and gesture != "No hand":
        fill = int(gfilter.streak_ratio * w)
        if fill > 0:
            bar_color = COMMAND_COLORS.get(command, (100, 100, 100))
            cv2.rectangle(frame, (0, bar_y), (fill, bar_y + bar_h), bar_color, -1)

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        color = COMMAND_COLORS.get(command, (0, 200, 0))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    return frame


def draw_sim_panel(drone, active_command, world_width_m):
    panel = np.zeros((SIM_H, SIM_W, 3), dtype=np.uint8)
    panel[:] = (30, 30, 30)

    # Grid
    for gx in range(0, SIM_W, 50):
        cv2.line(panel, (gx, 70), (gx, SIM_H), (50, 50, 50), 1)
    for gy in range(70, SIM_H, 50):
        cv2.line(panel, (0, gy), (SIM_W, gy), (50, 50, 50), 1)

    # Header
    cv2.rectangle(panel, (0, 0), (SIM_W, 65), (20, 20, 20), -1)
    cv2.putText(panel, "DRONE SIMULATOR", (15, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    status = "FLYING" if drone.is_flying else "LANDED"
    s_color = (0, 220, 0) if drone.is_flying else (100, 100, 100)
    cv2.putText(panel, status, (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, s_color, 2)

    cv2.putText(panel, f"ALT: {drone.altitude:.1f}m", (200, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    cv2.putText(panel, f"HVEL: {drone.forward_speed_m_s:.2f} m/s", (200, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 220, 180), 1)
    cv2.putText(panel, f"VVEL: {drone.climb_speed_m_s:+.2f} m/s", (360, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 200, 220), 1)

    cmd_color = COMMAND_COLORS.get(active_command, (100, 100, 100))
    cv2.putText(panel, active_command, (340, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, cmd_color, 2)

    # Trail (fades over 5 seconds)
    now = time.time()
    for tx, ty, t in drone.trail:
        age = now - t
        if age > 5.0:
            continue
        alpha = max(0, 1.0 - age / 5.0)
        color = (int(80 * alpha), int(200 * alpha), int(80 * alpha))
        cv2.circle(panel, (tx, ty), 2, color, -1)

    # Drone body
    dx, dy = int(drone.x), int(drone.y)
    size = int(12 + drone.altitude * 2)  # bigger when higher

    if drone.is_flying:
        body_color = (0, 200, 255)
        # Drone cross shape
        cv2.line(panel, (dx - size, dy - size), (dx + size, dy + size), body_color, 2)
        cv2.line(panel, (dx + size, dy - size), (dx - size, dy + size), body_color, 2)
        # Propeller circles
        for ox, oy in [(-size, -size), (size, -size), (-size, size), (size, size)]:
            prop_phase = int(time.time() * 20) % 6
            prop_r = 6 + prop_phase % 3
            cv2.circle(panel, (dx + ox, dy + oy), prop_r, (100, 200, 255), 1)
        # Center dot
        cv2.circle(panel, (dx, dy), 4, body_color, -1)
        # Heading indicator
        rad = math.radians(drone.heading)
        hx = int(dx + math.cos(rad) * (size + 8))
        hy = int(dy + math.sin(rad) * (size + 8))
        cv2.arrowedLine(panel, (dx, dy), (hx, hy), (0, 255, 255), 2, tipLength=0.4)
    else:
        cv2.circle(panel, (dx, dy), 8, (80, 80, 80), -1)
        cv2.circle(panel, (dx, dy), 8, (120, 120, 120), 1)
        cv2.putText(panel, "LANDED", (dx - 28, dy + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)

    # Shadow (grows with altitude)
    if drone.is_flying:
        shadow_offset = int(drone.altitude * 3)
        shadow_size = int(size * 0.6)
        cv2.ellipse(panel, (dx + shadow_offset, dy + shadow_offset),
                    (shadow_size, shadow_size // 2), 0, 0, 360, (15, 15, 15), -1)

    # Command log at bottom
    log_y = SIM_H - 15
    for _, cmd in reversed(drone.command_log):
        c = COMMAND_COLORS.get(cmd, (150, 150, 150))
        cv2.putText(panel, cmd, (15, log_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, c, 1)
        log_y -= 18
        if log_y < SIM_H - 150:
            break

    # Scale reminder (horizontal motion only; altitude uses SIM_ALT_SPEED_M_S directly)
    cv2.putText(panel, f"scale: {world_width_m:.0f}m / {SIM_W}px", (8, SIM_H - 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (90, 90, 110), 1)

    # Controls hint
    cv2.putText(panel, "[Q] quit  [R] reset", (SIM_W - 170, SIM_H - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 80), 1)

    return panel


# ── main loop ────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="2D gesture drone simulator (webcam or Tello preview → virtual drone)."
    )
    p.add_argument(
        "camera",
        nargs="?",
        type=int,
        default=0,
        help="Webcam index in --source webcam mode (default: 0).",
    )
    p.add_argument(
        "--source",
        choices=["webcam", "tello"],
        default="webcam",
        help="Video source: PC webcam or Tello camera (Wi‑Fi to drone, no real flight).",
    )
    p.add_argument(
        "--world-width-m",
        type=float,
        default=SIM_WORLD_WIDTH_M,
        help=f"Map the sim panel width ({SIM_W} px) to this many meters for HVEL readout (default: {SIM_WORLD_WIDTH_M}).",
    )
    p.add_argument(
        "--no-perception-gate",
        action="store_true",
        help="Disable YOLO+MediaPipe trusted-hand gate (on by default). Env: MLX_GESTURE_PERCEPTION_GATE=0.",
    )
    p.add_argument(
        "--k-create",
        type=int,
        default=4,
        help="Trusted-hand: consecutive YOLO+MP passes to create trust (default=4).",
    )
    p.add_argument(
        "--mp-miss-drop",
        type=int,
        default=5,
        help="MP miss streak on YOLO crop to drop trust (default=5).",
    )
    p.add_argument(
        "--no-box-drop",
        type=int,
        default=5,
        help="No YOLO box streak to drop trust (default=5).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    world_width_m = args.world_width_m
    m_per_px = world_width_m / float(SIM_W)
    nominal_forward_m_s = SIM_MOVE_SPEED_PX_S * m_per_px

    model, class_names = load_gesture_model()
    detector = load_hand_detector()
    tgate = None
    if perception_gate_wanted(args.no_perception_gate):
        tcfg = TrustedHandConfig(
            k_create=args.k_create,
            mp_miss_drop=args.mp_miss_drop,
            no_box_drop=args.no_box_drop,
        )
        lm = load_hand_landmarker(MODEL_DIR, tcfg)
        if lm is not None:
            tgate = TrustedHandGate(lm, tcfg)
            print("  Trusted-hand gate: ON (YOLO + MediaPipe; behavior gated)")
            print(
                f"    K_CREATE={tcfg.k_create} MP_MISS_DROP={tcfg.mp_miss_drop} "
                f"NO_BOX_DROP={tcfg.no_box_drop}"
            )
        else:
            print(f"  Trusted-hand gate: OFF — {describe_gate_load_failure(MODEL_DIR)}")
    else:
        print(
            "  Trusted-hand gate: OFF (--no-perception-gate or MLX_GESTURE_PERCEPTION_GATE=0)"
        )
    face_detector = None
    yunet_load_error = None
    try:
        face_detector = load_face_detector(MODEL_DIR)
        print("  YuNet face detector (hand/face overlap filter + follow preview on cam)")
    except Exception as e:
        yunet_load_error = str(e)[:96]
        print(f"  YuNet unavailable ({yunet_load_error}); face-as-hand filter off")

    proximity_smoother = ProximitySmoother(alpha=0.35)

    cap = None
    tello = None
    frame_reader = None
    battery = None
    temp = None
    telem_timer = 0.0

    if args.source == "webcam":
        cap = find_camera(args.camera)
        if cap is None:
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, PERF_CAPTURE_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, PERF_CAPTURE_H)
        cap.set(cv2.CAP_PROP_FPS, 30)
        source_label = f"WEBCAM:{args.camera}"
    else:
        try:
            from djitellopy import Tello
        except ImportError:
            print("ERROR: djitellopy not installed. pip install djitellopy")
            return
        print("Connecting to Tello (Wi‑Fi to drone)...")
        tello = Tello()
        tello.connect()
        battery = tello.get_battery()
        temp = tello.get_temperature()
        print(f"  Battery {battery}%  Temp {temp}C")
        tello.streamon()
        frame_reader = tello.get_frame_read()
        print("Waiting for video...", end="", flush=True)
        deadline = time.time() + 15.0
        while time.time() < deadline:
            f = frame_reader.frame
            if f is not None and f.size > 0:
                break
            print(".", end="", flush=True)
            time.sleep(0.15)
        else:
            print("\nERROR: No Tello video. Check Wi‑Fi and try again.")
            tello.streamoff()
            tello.end()
            return
        print(" OK\n")
        source_label = "TELLO_CAM"
        telem_timer = time.time()

    smoother = BboxSmoother(alpha=0.4, max_miss_frames=8)
    gfilter = GestureFilter(
        window=10,
        lock_frames=GESTURE_LOCK_FRAMES,
        unlock_frames=GESTURE_UNLOCK_FRAMES,
        min_vote_share=0.60,
    )
    drone = DroneState(world_width_m=world_width_m)
    logger = SessionLogger()
    active_command = "IDLE"
    last_command_time = 0.0
    prev_time = time.time()
    fps = 0.0
    follow_latched = False
    follow_no_hand_streak = 0
    sim_beh_state = "MANUAL"
    acq_streak = 0
    loss_streak = 0
    hold_streak = 0
    prev_confirmed_for_fist = None

    print("\n" + "=" * 50)
    print("  GESTURE DRONE SIMULATOR")
    print("=" * 50)
    print(f"Video: {args.source}   Sim horizontal scale: {world_width_m:.1f} m / {SIM_W} px")
    print(f"  → nominal HVEL ≈ {nominal_forward_m_s:.3f} m/s if MOVE FORWARD were used (SIM_MOVE_SPEED_PX_S={SIM_MOVE_SPEED_PX_S}); peace=FOLLOW_ARM")
    print(f"  → climb / descend rate: {SIM_ALT_SPEED_M_S:.2f} m/s (sim altitude axis)")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD * 100:.0f}%")
    print(f"Gesture lock frames : {GESTURE_LOCK_FRAMES}  unlock: {GESTURE_UNLOCK_FRAMES}")
    print(f"Dead-band threshold : 60% weighted vote share")
    print(
        f"Follow latch         : {'ON' if FOLLOW_LATCH_ENABLE else 'OFF'} "
        f"(hold FOLLOW_ARM until fist/palm/thumbs @ conf≥{FOLLOW_EXIT_CONF} "
        f"& margin≥{FOLLOW_EXIT_MARGIN}, or {FOLLOW_NO_HAND_FRAMES_TO_DROP} no-hand frames)"
    )
    if tgate is not None:
        print("  Trusted-hand gate  : active (HUD trust line)")
    print(f"Session log         : {logger._path}")
    print("  CSV columns: YOLO counts/conf, reject_stage, classifier margin + full softmax JSON")
    if args.source == "webcam":
        print(
            f"  Perf: capture {PERF_CAPTURE_W}x{PERF_CAPTURE_H}, "
            f"YuNet side≤{YUNET_MAX_INFER_SIDE} every {YUNET_FRAME_STRIDE} fr, "
            f"YOLO long edge≤{hand_detection.MAX_INFER_SIDE_DEFAULT} imgsz={hand_detection.YOLO_IMGSZ}"
        )
    else:
        print(
            f"  Perf: YuNet side≤{YUNET_MAX_INFER_SIDE} every {YUNET_FRAME_STRIDE} fr, "
            f"YOLO long edge≤{hand_detection.MAX_INFER_SIDE_DEFAULT} imgsz={hand_detection.YOLO_IMGSZ}"
        )
    print()

    try:
        while True:
            if cap is not None:
                ret, frame = cap.read()
                if not ret:
                    break
            else:
                frame = frame_reader.frame
                if frame is None or frame.size == 0:
                    time.sleep(0.01)
                    continue

            if args.source == "tello" and time.time() - telem_timer > 3.0:
                try:
                    battery = tello.get_battery()
                    temp = tello.get_temperature()
                except Exception:
                    pass
                telem_timer = time.time()

            now = time.time()
            dt = now - prev_time
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)
            prev_time = now

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_fb = None
            face_score = 0.0
            if face_detector is not None:
                face_fb, face_score = detect_largest_face(
                    face_detector,
                    frame,
                    max_infer_side=YUNET_MAX_INFER_SIDE,
                )
                face_score = float(face_score or 0.0)
            fw, fh = frame.shape[1], frame.shape[0]
            face_ok = False
            face_x_norm = 0.0
            if face_fb is not None:
                face_ok, face_x_norm = face_ok_and_x_norm(face_fb, face_score, fw, fh)
            hand_crop, bbox, frame_diag = detect_hand(
                detector,
                frame_rgb,
                smoother,
                face_xyxy=face_fb,
                face_hand_iou_max=0.22,
                padding_ratio=PADDING_RATIO,
            )
            if tgate is not None:
                frame_diag = tgate.update(hand_crop, frame_diag)
            else:
                frame_diag = dict(frame_diag or {})
                frame_diag["trust_enabled"] = False
                frame_diag["behavior_allow"] = hand_crop is not None
                frame_diag["trust_phase"] = "gate_off"

            behavior_allow = bool(frame_diag.get("behavior_allow", hand_crop is not None))

            raw_gesture = "No hand"
            confidence = 0.0
            new_command = None
            cls_margin = 0.0
            cls_probs_json = ""

            if hand_crop is not None:
                idx, confidence, cls_margin, cls_probs_json = classify_hand(
                    model, hand_crop, class_names
                )
                raw_gesture = class_names[idx]

            feed = (
                raw_gesture
                if (
                    behavior_allow
                    and hand_crop is not None
                    and confidence >= CONFIDENCE_THRESHOLD
                )
                else None
            )
            gesture = (
                gfilter.update(feed, confidence if behavior_allow else 0.0) or "No hand"
            )
            cmd_gesture = gesture

            if FOLLOW_LATCH_ENABLE and gfilter._confirmed == "two_fingers":
                follow_latched = True

            if FOLLOW_LATCH_ENABLE and follow_latched:
                if gesture == "No hand":
                    follow_no_hand_streak += 1
                else:
                    follow_no_hand_streak = 0
                exit_explicit = (
                    behavior_allow
                    and hand_crop is not None
                    and raw_gesture in ("fist", "open_palm", "thumbs_up")
                    and confidence >= FOLLOW_EXIT_CONF
                    and cls_margin >= FOLLOW_EXIT_MARGIN
                )
                exit_timeout = follow_no_hand_streak >= FOLLOW_NO_HAND_FRAMES_TO_DROP
                if exit_explicit or exit_timeout:
                    follow_latched = False
                    follow_no_hand_streak = 0
                else:
                    cmd_gesture = "two_fingers"

            if cmd_gesture != "No hand":
                new_command = GESTURE_TO_COMMAND.get(cmd_gesture)

            command_fired = None
            if new_command and new_command != active_command:
                if (now - last_command_time) >= COMMAND_COOLDOWN:
                    active_command = new_command
                    last_command_time = now
                    command_fired = active_command
                    drone.log_command(active_command)
                    print(f"  >>> {active_command}  ({cmd_gesture} @ {confidence*100:.0f}%)")

            if cmd_gesture == "No hand":
                if drone.is_flying:
                    active_command = "STOP"
                else:
                    active_command = "IDLE"

            logger.log(
                raw_gesture,
                confidence,
                gfilter._confirmed,
                command_fired,
                gfilter.streak_ratio,
                active_command=active_command,
                diag=frame_diag,
                classifier_margin=cls_margin,
                classifier_probs_json=cls_probs_json,
                cmd_gesture=cmd_gesture,
            )

            drone.update(active_command, dt)

            if not drone.is_flying and active_command == "LAND" and drone.altitude <= 0:
                active_command = "LANDED"

            confirmed = gfilter._confirmed
            fist_edge = (
                drone.is_flying
                and confirmed == "fist"
                and prev_confirmed_for_fist != "fist"
            )
            prev_confirmed_for_fist = confirmed

            sim_yaw_rate_deg_s = 0.0
            dt_run = max(dt, 1e-6)

            if not drone.is_flying or active_command == "LAND":
                sim_beh_state = "MANUAL"
                acq_streak = loss_streak = hold_streak = 0
            else:
                if fist_edge:
                    if sim_beh_state == "MANUAL":
                        sim_beh_state = "SEARCH"
                        acq_streak = loss_streak = hold_streak = 0
                    elif sim_beh_state in ("SEARCH", "FACE_LOCK"):
                        sim_beh_state = "MANUAL"
                        acq_streak = loss_streak = hold_streak = 0

                if sim_beh_state == "SEARCH":
                    d_deg = math.degrees(OMEGA_SEARCH * dt_run)
                    drone.heading += d_deg
                    sim_yaw_rate_deg_s = math.degrees(OMEGA_SEARCH)
                    if face_ok:
                        acq_streak += 1
                        if acq_streak >= M_ACQUIRE:
                            sim_beh_state = "FACE_LOCK"
                            acq_streak = loss_streak = hold_streak = 0
                    else:
                        acq_streak = 0

                elif sim_beh_state == "FACE_LOCK":
                    if face_ok:
                        loss_streak = 0
                        ex = face_x_norm
                        if abs(ex) > EPS_X:
                            rate_rad_s = -KP_LOCK * ex
                            rate_rad_s = max(
                                -MAX_ANG_LOCK, min(MAX_ANG_LOCK, rate_rad_s)
                            )
                            d_deg = math.degrees(rate_rad_s * dt_run)
                            drone.heading += d_deg
                            sim_yaw_rate_deg_s = math.degrees(rate_rad_s)
                        else:
                            sim_yaw_rate_deg_s = 0.0
                    else:
                        loss_streak += 1
                        sim_yaw_rate_deg_s = 0.0
                        if loss_streak >= M_LOSS:
                            sim_beh_state = "SEARCH"
                            acq_streak = loss_streak = hold_streak = 0

                    if face_ok and abs(face_x_norm) <= EPS_X:
                        hold_streak += 1
                    else:
                        hold_streak = 0

            drone.heading = _norm_heading_deg(drone.heading)

            in_deadband = bool(
                sim_beh_state == "FACE_LOCK"
                and face_ok
                and abs(face_x_norm) <= EPS_X
            )
            settled = hold_streak >= H_HOLD

            follow_arm = cmd_gesture == "two_fingers"
            # Fist → STOP → hover: face overlay only in MANUAL (autonomous modes use SEARCH/FACE_LOCK HUD rules).
            hover_face_scan = (
                drone.is_flying
                and active_command == "STOP"
                and sim_beh_state == "MANUAL"
            )
            face_tracked = False
            face_prox_smoothed = None
            if follow_arm and face_detector is not None:
                raw_p = proximity_from_bbox(face_fb, fw, fh)
                face_prox_smoothed = proximity_smoother.update(raw_p)
                if face_fb is not None:
                    face_tracked = True
            elif hover_face_scan and face_detector is not None:
                raw_p = proximity_from_bbox(face_fb, fw, fh)
                face_prox_smoothed = proximity_smoother.update(raw_p)
                if face_fb is not None:
                    face_tracked = True
            elif not follow_arm and not hover_face_scan:
                proximity_smoother.reset()

            cam_h = SIM_H
            cam_w = int(frame.shape[1] * cam_h / frame.shape[0])
            frame_resized = cv2.resize(frame, (cam_w, cam_h))

            trust_hud = format_trust_hud_line(frame_diag) if tgate is not None else None
            beh_hud = (
                f"beh={sim_beh_state} hdg={drone.heading:.1f} "
                f"yaw_cmd={sim_yaw_rate_deg_s:+.1f}deg/s "
                f"db={int(in_deadband)} hold={hold_streak}/{H_HOLD} settled={int(settled)} "
                f"fly={int(drone.is_flying)} "
                f"acq={acq_streak}/{M_ACQUIRE} loss={loss_streak}/{M_LOSS} face_ok={int(face_ok)}"
            )
            cam_panel = draw_cam_panel(
                frame_resized,
                raw_gesture,
                cmd_gesture,
                confidence,
                None,
                active_command,
                fps,
                gfilter,
                source_label=source_label,
                battery=battery,
                temp=temp,
                follow_preview=follow_arm,
                face_proximity=face_prox_smoothed,
                face_tracked=face_tracked,
                yunet_error=yunet_load_error,
                trust_line=trust_hud,
                beh_line=beh_hud,
            )

            scale_x = cam_w / frame.shape[1]
            scale_y = cam_h / frame.shape[0]
            # Face box: FACE_LOCK > SEARCH (face_ok) > follow / hover STOP scan
            if sim_beh_state == "FACE_LOCK" and face_ok and face_fb is not None:
                fx1, fy1, fx2, fy2 = face_fb
                cv2.rectangle(
                    cam_panel,
                    (int(fx1 * scale_x), int(fy1 * scale_y)),
                    (int(fx2 * scale_x), int(fy2 * scale_y)),
                    (0, 220, 255),
                    3,
                )
                cv2.putText(
                    cam_panel,
                    "LOCK",
                    (int(fx1 * scale_x), max(24, int(fy1 * scale_y) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 220, 255),
                    2,
                )
            elif sim_beh_state == "SEARCH" and face_ok and face_fb is not None:
                fx1, fy1, fx2, fy2 = face_fb
                cv2.rectangle(
                    cam_panel,
                    (int(fx1 * scale_x), int(fy1 * scale_y)),
                    (int(fx2 * scale_x), int(fy2 * scale_y)),
                    (80, 200, 255),
                    2,
                )
            elif (follow_arm or hover_face_scan) and face_fb is not None:
                fx1, fy1, fx2, fy2 = face_fb
                cv2.rectangle(
                    cam_panel,
                    (int(fx1 * scale_x), int(fy1 * scale_y)),
                    (int(fx2 * scale_x), int(fy2 * scale_y)),
                    (80, 200, 255),
                    2,
                )

            if bbox is not None:
                bx1, by1, bx2, by2 = bbox
                scaled_bbox = (
                    int(bx1 * scale_x),
                    int(by1 * scale_y),
                    int(bx2 * scale_x),
                    int(by2 * scale_y),
                )
                color = COMMAND_COLORS.get(active_command, (0, 200, 0))
                cv2.rectangle(
                    cam_panel,
                    (scaled_bbox[0], scaled_bbox[1]),
                    (scaled_bbox[2], scaled_bbox[3]),
                    color,
                    2,
                )

            sim_panel = draw_sim_panel(drone, active_command, world_width_m)

            sep = np.zeros((SIM_H, 3, 3), dtype=np.uint8)
            sep[:] = (80, 80, 80)

            combined = np.hstack([cam_panel, sep, sim_panel])
            cv2.imshow("Gesture Drone Simulator", combined)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            if key in (ord("r"), ord("R")):
                drone.reset()
                active_command = "IDLE"
                sim_beh_state = "MANUAL"
                acq_streak = loss_streak = hold_streak = 0
                prev_confirmed_for_fist = None
                print("  [RESET] Drone position reset")

    except KeyboardInterrupt:
        pass
    finally:
        logger.close()
        if cap is not None:
            cap.release()
        if tello is not None:
            tello.streamoff()
            tello.end()
        cv2.destroyAllWindows()
        print("Simulation ended.")


if __name__ == "__main__":
    main()
