r"""
Gesture-to-ROS2 bridge (Windows side).

Runs YOLO + EfficientNet gesture recognition and sends commands over TCP to the
ROS2 node (Gazebo) on WSL2.

Run from Windows PowerShell:
    python "...\\gesture_bridge.py" 2
    python "...\\gesture_bridge.py" --source tello

Arguments:
    [camera_index]  Webcam index when --source webcam (default: 0)
    --source webcam|tello   Perception from PC webcam or Tello camera (Wi‑Fi to drone)
    --host HOST     WSL2 IP (default: auto-detect)
    --port PORT     TCP port (default: 9090)
"""

import argparse
import json
import select
import socket
import subprocess
import sys
import time
from pathlib import Path

import cv2
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
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
from search_behavior import M_ACQUIRE, M_LOSS, face_ok_and_x_norm
from yunet_face import (
    ProximitySmoother,
    detect_largest_face,
    load_face_detector,
    proximity_from_bbox,
)

# ── paths ────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR.parent / "models"
GESTURE_MODEL_PATH = MODEL_DIR / "gesture_model.pt"
HAND_LANDMARK_PATH = MODEL_DIR / "hand_landmarker.task"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PADDING_RATIO = 0.08   # YOLO boxes already contain the full hand; less padding needed

CONFIDENCE_THRESHOLD = 0.85
# Keep in sync with simulate_drone.py (conservative command spacing).
COMMAND_COOLDOWN       = 1.2
GESTURE_LOCK_FRAMES    = 8
GESTURE_UNLOCK_FRAMES = 12

# Reject YOLO "hand" boxes that overlap the YuNet face box (face mis-detected as hand).
FACE_HAND_IOU_MAX = 0.22

# Webcam capture + perception (Tello resolution is fixed by the drone; stride still helps).
PERF_CAPTURE_W = 960
PERF_CAPTURE_H = 540
YUNET_MAX_INFER_SIDE = 480
YUNET_FRAME_STRIDE = 2

# After peace sign locks, keep FOLLOW_ARM until explicit high-confidence exit or long no-hand.
FOLLOW_LATCH_ENABLE = True
FOLLOW_NO_HAND_FRAMES_TO_DROP = 45
FOLLOW_EXIT_CONF = 0.93
FOLLOW_EXIT_MARGIN = 0.18

# two_fingers = follow-stack hook (YuNet preview); does NOT command forward flight.
GESTURE_TO_COMMAND = {
    "two_fingers": "FOLLOW_ARM",
    "fist":        "STOP",
    "open_palm":   "LAND",
    "thumbs_up":   "MOVE_UP",
}

COMMAND_COLORS = {
    "FOLLOW_ARM":   (180, 220, 100),
    "STOP":         (0, 200, 255),
    "LAND":         (0, 0, 220),
    "MOVE_UP":      (220, 180, 0),
    "IDLE":         (100, 100, 100),
    "NO_HAND":      (80, 80, 80),
}

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def load_gesture_model():
    checkpoint = torch.load(str(GESTURE_MODEL_PATH), map_location=DEVICE, weights_only=False)
    class_names = checkpoint["class_names"]
    model = efficientnet_b0()
    model.classifier = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(1280, len(class_names)))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    print(f"Gesture model loaded ({len(class_names)} classes, device={DEVICE})")
    return model, class_names


def load_hand_detector():
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


def find_camera(cam_index):
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


class BboxSmoother:
    """
    Smooths the hand bounding box with EMA and holds the last known position for
    a few frames after MediaPipe drops detection. This eliminates bbox jitter and
    prevents single-frame dropout from breaking gesture classification.

    alpha         : EMA responsiveness (0–1). Higher = snappier, lower = smoother.
    max_miss_frames: how many consecutive missed frames to tolerate before giving up.
                    At 30 fps, 8 frames = ~267 ms grace window.
    """
    def __init__(self, alpha=0.4, max_miss_frames=8):
        self.alpha = alpha
        self.max_miss = max_miss_frames
        self._smooth = None   # (x1, y1, x2, y2) as floats
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

    def reset(self):
        self._smooth = None
        self._misses = 0


class GestureFilter:
    """
    Confidence-weighted, recency-biased sliding-window gesture filter with
    dead-band and asymmetric hysteresis. Identical to the version in simulate_drone.py.

    Prevents rapid-fire commands to the drone during gesture transitions.
    """
    def __init__(self, window=10, lock_frames=GESTURE_LOCK_FRAMES,
                 unlock_frames=GESTURE_UNLOCK_FRAMES, min_vote_share=0.60):
        self._window         = window
        self._lock_frames    = lock_frames
        self._unlock_frames  = unlock_frames
        self._min_vote_share = min_vote_share
        self._history        = []
        self._streak         = 0
        self._streak_cand    = None
        self._confirmed      = None
        self._lock_target    = lock_frames

    def update(self, gesture, confidence=1.0):
        label = gesture if gesture is not None else "__none__"
        conf  = float(confidence) if confidence is not None else 0.0
        self._history.append((label, conf))
        if len(self._history) > self._window:
            self._history.pop(0)

        n = len(self._history)
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

        if winner_share < self._min_vote_share:
            self._streak      = 0
            self._streak_cand = None
            return self._confirmed

        if self._confirmed is not None and winner_gest != self._confirmed:
            self._lock_target = self._unlock_frames
        else:
            self._lock_target = self._lock_frames

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
        if self._lock_target == 0:
            return 0.0
        return min(1.0, self._streak / self._lock_target)


@torch.no_grad()
def classify_hand(model, hand_rgb):
    tensor = preprocess(hand_rgb).unsqueeze(0).to(DEVICE)
    logits = model(tensor)
    probs = F.softmax(logits, dim=1)
    p = probs[0]
    confidence, idx = probs.max(1)
    top2 = torch.topk(p, min(2, p.numel()))
    margin = (
        (top2.values[0] - top2.values[1]).item()
        if top2.values.numel() > 1
        else confidence.item()
    )
    return idx.item(), confidence.item(), margin


def get_wsl_ip():
    """Auto-detect WSL2 IP by querying hostname -I inside WSL."""
    try:
        result = subprocess.run(
            ["wsl", "-d", "Ubuntu-22.04", "--", "hostname", "-I"],
            capture_output=True, text=True, timeout=5,
        )
        ip = result.stdout.strip().split()[0]
        print(f"Auto-detected WSL2 IP: {ip}")
        return ip
    except Exception:
        return "localhost"


def connect_to_ros2(host, port, max_retries=10):
    """Connect to the ROS2 bridge node with retries."""
    for attempt in range(max_retries):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2.0)
            sock.connect((host, port))
            sock.settimeout(0.5)
            print(f"Connected to ROS2 bridge at {host}:{port}")
            return sock
        except (ConnectionRefusedError, socket.timeout, OSError) as e:
            if attempt < max_retries - 1:
                print(f"  Connection attempt {attempt+1}/{max_retries} failed ({e}), retrying...")
                time.sleep(2)
            else:
                print(f"ERROR: Could not connect to {host}:{port} after {max_retries} attempts")
                return None


def send_command(sock, command, gesture, confidence, extra=None):
    """Send a JSON command over TCP. Returns False if connection is lost."""
    payload = {
        "command": command,
        "gesture": gesture,
        "confidence": round(confidence, 3),
        "timestamp": time.time(),
    }
    if extra:
        payload.update(extra)
    msg = json.dumps(payload) + "\n"
    try:
        sock.sendall(msg.encode("utf-8"))
        return True
    except (BrokenPipeError, ConnectionResetError, OSError):
        return False


def _parse_beh_replies(recv_buf, data: bytes, latest_beh: dict) -> None:
    recv_buf.extend(data)
    while True:
        i = recv_buf.find(b"\n")
        if i < 0:
            break
        line = bytes(recv_buf[:i])
        del recv_buf[: i + 1]
        if not line.strip():
            continue
        try:
            j = json.loads(line.decode("utf-8"))
            if j.get("type") == "beh_debug":
                latest_beh.clear()
                latest_beh.update(j)
        except json.JSONDecodeError:
            pass


def poll_beh_replies(sock, recv_buf: bytearray, latest_beh: dict) -> bool:
    """Drain non-blocking beh_debug lines from ROS. Returns False if peer closed."""
    try:
        while True:
            r, _, _ = select.select([sock], [], [], 0)
            if not r:
                return True
            chunk = sock.recv(65536)
            if not chunk:
                return False
            _parse_beh_replies(recv_buf, chunk, latest_beh)
    except (OSError, ValueError):
        return True


def draw_hud(frame, raw_gesture, gesture, confidence, command, fps, connected, gfilter=None,
             video_label="", battery=None, temp=None,
             follow_preview=False, face_proximity=None, face_tracked=False,
             yunet_error=None, trust_line=None, beh_line=None):
    h, w = frame.shape[:2]
    hud_extra = (28 if trust_line else 0) + (22 if beh_line else 0)
    hud_h = (138 if follow_preview else 128) + hud_extra
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, hud_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    if video_label:
        cv2.putText(frame, video_label, (w - min(260, w - 8), 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 200, 160), 1)
    if battery is not None:
        bc = (0, 200, 0) if battery > 30 else (0, 80, 220)
        cv2.putText(frame, f"BAT {battery}%", (w - 100, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, bc, 1)
    if temp is not None:
        cv2.putText(frame, f"TMP {int(temp)}C", (w - 100, 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 140), 1)

    conn_color = (0, 200, 0) if connected else (0, 0, 200)
    conn_text  = "ROS2 CONNECTED" if connected else "DISCONNECTED"
    cv2.putText(frame, conn_text, (w - 200, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, conn_color, 1)

    if gesture != "No hand":
        color = COMMAND_COLORS.get(command, (200, 200, 200))
        cv2.putText(frame, gesture, (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(frame, f"{confidence * 100:.0f}%", (15, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"CMD: {command}", (15, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        if raw_gesture != gesture and raw_gesture != "No hand":
            cv2.putText(frame, f"raw: {raw_gesture}", (15, 118),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (130, 130, 130), 1)
    else:
        cv2.putText(frame, "No hand", (15, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 100, 100), 2)
        cv2.putText(frame, "CMD: IDLE", (15, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)

    cv2.putText(frame, f"FPS: {fps:.0f}", (w - 100, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    cv2.putText(frame, "[Q] quit", (w - 100, 118),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)

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
            frame, trust_line[: min(95, len(trust_line))], (12, hud_h - 48),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 200, 160), 1,
        )

    if beh_line:
        cv2.putText(
            frame, beh_line[: min(120, len(beh_line))], (12, hud_h - 26),
            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 100), 1,
        )

    # Confirmation progress bar
    bar_y, bar_h = hud_h - 8, 8
    cv2.rectangle(frame, (0, bar_y), (w, bar_y + bar_h), (40, 40, 40), -1)
    if gfilter is not None and gesture != "No hand":
        fill = int(gfilter.streak_ratio * w)
        if fill > 0:
            cv2.rectangle(frame, (0, bar_y), (fill, bar_y + bar_h),
                          COMMAND_COLORS.get(command, (100, 100, 100)), -1)


def main():
    parser = argparse.ArgumentParser(description="Gesture bridge to ROS2 (Gazebo)")
    parser.add_argument("camera", nargs="?", type=int, default=0,
                        help="Webcam index when --source webcam (default: 0)")
    parser.add_argument(
        "--source",
        choices=["webcam", "tello"],
        default="webcam",
        help="Video: PC webcam or Tello camera (join TELLO-XXXX Wi‑Fi for tello).",
    )
    parser.add_argument("--host", default=None,
                        help="WSL2 IP (default: auto-detect)")
    parser.add_argument("--port", type=int, default=9090,
                        help="TCP port (default: 9090)")
    parser.add_argument(
        "--no-perception-gate",
        action="store_true",
        help="Disable YOLO+MediaPipe trusted-hand gate (on by default). Env: MLX_GESTURE_PERCEPTION_GATE=0.",
    )
    parser.add_argument(
        "--k-create",
        type=int,
        default=4,
        help="Consecutive YOLO+MP passes to create trust (default=4).",
    )
    parser.add_argument(
        "--mp-miss-drop",
        type=int,
        default=5,
        help="MP miss streak on YOLO crop to drop trust (default=5).",
    )
    parser.add_argument(
        "--no-box-drop",
        type=int,
        default=5,
        help="No YOLO box streak to drop trust (default=5).",
    )
    args = parser.parse_args()

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
            print("  Trusted-hand gate: ON (YOLO + MediaPipe)")
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

    cap = None
    tello = None
    frame_reader = None
    battery = None
    temp = None
    telem_timer = 0.0
    video_label = ""

    if args.source == "webcam":
        cap = find_camera(args.camera)
        if cap is None:
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, PERF_CAPTURE_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, PERF_CAPTURE_H)
        cap.set(cv2.CAP_PROP_FPS, 30)
        video_label = f"WEBCAM:{args.camera}"
    else:
        try:
            from djitellopy import Tello
        except ImportError:
            print("ERROR: pip install djitellopy")
            return
        print("Connecting Tello (Wi‑Fi to drone) for Gazebo bridge...")
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
            print("\nERROR: No Tello video.")
            tello.streamoff()
            tello.end()
            return
        print(" OK\n")
        video_label = "TELLO + GAZEBO"
        telem_timer = time.time()

    host = args.host or get_wsl_ip()
    sock = connect_to_ros2(host, args.port)
    connected = sock is not None
    if sock is not None:
        sock.settimeout(None)

    tcp_recv_buf = bytearray()
    latest_beh_dbg = {}

    smoother = BboxSmoother(alpha=0.4, max_miss_frames=8)
    gfilter  = GestureFilter(window=10, lock_frames=GESTURE_LOCK_FRAMES,
                             unlock_frames=GESTURE_UNLOCK_FRAMES, min_vote_share=0.60)
    face_detector = None
    proximity_smoother = ProximitySmoother(alpha=0.35)
    yunet_load_error = None
    try:
        face_detector = load_face_detector(MODEL_DIR)
        print("  YuNet face detector loaded (hand/face overlap filter + follow preview)")
    except Exception as e:
        yunet_load_error = str(e)[:96]
        print(f"  YuNet unavailable ({yunet_load_error}); face-as-hand filter off")
    yunet_frame_i = 0
    cached_face_pack = None  # (bbox, score)
    prev_confirmed_for_fist = None
    follow_latched = False
    follow_no_hand_streak = 0
    active_command    = "IDLE"
    last_command_time = 0.0
    prev_time         = time.time()
    fps               = 0.0

    print("\n" + "=" * 50)
    print("  GESTURE → ROS2 BRIDGE")
    print("=" * 50)
    print(f"Video: {args.source}" + (f" (index {args.camera})" if args.source == "webcam" else ""))
    print(f"Sending commands to {host}:{args.port}")
    print(f"Confidence threshold : {CONFIDENCE_THRESHOLD * 100:.0f}%")
    print(f"Gesture lock frames  : {GESTURE_LOCK_FRAMES}  unlock: {GESTURE_UNLOCK_FRAMES}")
    print(f"Dead-band threshold  : 60% weighted vote share")
    if args.source == "webcam":
        print(
            f"Perf (webcam)        : capture {PERF_CAPTURE_W}x{PERF_CAPTURE_H}, "
            f"YuNet side≤{YUNET_MAX_INFER_SIDE} every {YUNET_FRAME_STRIDE} fr, "
            f"YOLO long edge≤{hand_detection.MAX_INFER_SIDE_DEFAULT} imgsz={hand_detection.YOLO_IMGSZ}"
        )
    else:
        print(
            f"Perf (Tello)         : YuNet side≤{YUNET_MAX_INFER_SIDE} every {YUNET_FRAME_STRIDE} fr, "
            f"YOLO long edge≤{hand_detection.MAX_INFER_SIDE_DEFAULT} imgsz={hand_detection.YOLO_IMGSZ}"
        )
    print(
        f"Follow latch         : {'ON' if FOLLOW_LATCH_ENABLE else 'OFF'} "
        f"(hold FOLLOW_ARM until fist/palm/thumbs @ conf≥{FOLLOW_EXIT_CONF} "
        f"& margin≥{FOLLOW_EXIT_MARGIN}, or {FOLLOW_NO_HAND_FRAMES_TO_DROP} no-hand frames)"
    )
    if tgate is not None:
        print("  Trusted-hand gate  : active (HUD trust line)")
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

            if tello is not None and time.time() - telem_timer > 3.0:
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
                yunet_frame_i += 1
                if yunet_frame_i % YUNET_FRAME_STRIDE == 0 or cached_face_pack is None:
                    fb, sc = detect_largest_face(
                        face_detector,
                        frame,
                        max_infer_side=YUNET_MAX_INFER_SIDE,
                    )
                    cached_face_pack = (fb, float(sc))
                face_fb, face_score = cached_face_pack

            hand_crop, bbox, frame_diag = detect_hand(
                detector,
                frame_rgb,
                smoother,
                face_xyxy=face_fb,
                face_hand_iou_max=FACE_HAND_IOU_MAX,
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
            confidence  = 0.0
            new_command = None

            cls_margin = 0.0
            if hand_crop is not None:
                idx, confidence, cls_margin = classify_hand(model, hand_crop)
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

            confirmed = gfilter._confirmed
            fist_edge = confirmed == "fist" and prev_confirmed_for_fist != "fist"
            prev_confirmed_for_fist = confirmed

            if cmd_gesture != "No hand":
                new_command = GESTURE_TO_COMMAND.get(cmd_gesture)

            if new_command and new_command != active_command:
                if (now - last_command_time) >= COMMAND_COOLDOWN:
                    active_command    = new_command
                    last_command_time = now
                    print(f"  >>> {active_command}  ({cmd_gesture} @ {confidence*100:.0f}%)")

            if cmd_gesture == "No hand":
                active_command = "IDLE"

            fh, fw = frame.shape[:2]
            follow_arm = (cmd_gesture == "two_fingers")
            face_tracked = False
            face_prox_smoothed = None
            if follow_arm and face_detector is not None:
                raw_p = proximity_from_bbox(face_fb, fw, fh)
                face_prox_smoothed = proximity_smoother.update(raw_p)
                if face_fb is not None:
                    face_tracked = True
            elif not follow_arm:
                proximity_smoother.reset()

            face_ok, face_x_norm = face_ok_and_x_norm(face_fb, face_score, fw, fh)

            tcp_extra = {
                "fist_edge": fist_edge,
                "face_ok": face_ok,
                "face_x_norm": round(face_x_norm, 4),
            }

            if connected:
                ok = send_command(
                    sock, active_command, cmd_gesture, confidence, extra=tcp_extra
                )
                if ok:
                    poll_ok = poll_beh_replies(sock, tcp_recv_buf, latest_beh_dbg)
                    if not poll_ok:
                        ok = False
                if not ok:
                    print("  [!] Connection lost, attempting reconnect...")
                    try:
                        sock.close()
                    except OSError:
                        pass
                    tcp_recv_buf.clear()
                    latest_beh_dbg.clear()
                    sock = connect_to_ros2(host, args.port, max_retries=3)
                    connected = sock is not None
                    if sock is not None:
                        sock.settimeout(None)

            # Face box: same style as follow preview, plus SEARCH / FACE_LOCK (fist autonomy).
            beh_state = latest_beh_dbg.get("beh_state", "MANUAL") if latest_beh_dbg else "MANUAL"
            scan_modes = beh_state in ("SEARCH", "FACE_LOCK")
            if face_fb is not None and (follow_arm or scan_modes):
                fx1, fy1, fx2, fy2 = face_fb
                cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (80, 200, 255), 2)
                if scan_modes and not follow_arm:
                    face_tracked = True

            if bbox is not None:
                color = COMMAND_COLORS.get(active_command, (0, 200, 0))
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            trust_hud = format_trust_hud_line(frame_diag) if tgate is not None else None
            if latest_beh_dbg:
                beh_hud = (
                    f"BEH {latest_beh_dbg.get('beh_state', '?')} "
                    f"fly={int(latest_beh_dbg.get('flying', 0))} | "
                    f"acq {latest_beh_dbg.get('acq_streak', 0)}/"
                    f"{latest_beh_dbg.get('M_acquire', M_ACQUIRE)} | "
                    f"loss {latest_beh_dbg.get('loss_streak', 0)}/"
                    f"{latest_beh_dbg.get('M_loss', M_LOSS)} | "
                    f"f_ok={int(latest_beh_dbg.get('face_ok', 0))} "
                    f"yaw={latest_beh_dbg.get('autonomy_yaw', 0)}"
                )
            else:
                beh_hud = "BEH (waiting for ROS debug…)" if connected else None
            draw_hud(
                frame, raw_gesture, cmd_gesture, confidence, active_command, fps, connected, gfilter,
                video_label=video_label, battery=battery, temp=temp,
                follow_preview=follow_arm,
                face_proximity=face_prox_smoothed,
                face_tracked=face_tracked,
                yunet_error=yunet_load_error,
                trust_line=trust_hud,
                beh_line=beh_hud,
            )
            cv2.imshow("Gesture -> ROS2 Bridge", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break

    except KeyboardInterrupt:
        pass
    finally:
        if sock:
            try:
                send_command(sock, "STOP", "shutdown", 0.0)
                sock.close()
            except Exception:
                pass
        pass  # YOLO detectors need no explicit close
        if cap is not None:
            cap.release()
        if tello is not None:
            tello.streamoff()
            tello.end()
        cv2.destroyAllWindows()
        print("\nBridge stopped.")


if __name__ == "__main__":
    main()
