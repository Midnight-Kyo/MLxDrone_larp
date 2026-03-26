r"""
Live gesture recognition using the **Tello camera** — **same perception stack as**
``simulate_drone.py`` / ``gesture_bridge.py``: ``hand_detection`` + **TrustedHandGate** (MediaPipe)
+ **YuNet** + **GestureFilter** + follow latch. **Trusted-hand is ON by default** with
**Tello-tuned** MediaPipe thresholds / crop upscale (``trusted_hand_config_tello_camera`` —
softer on compressed drone video than bridge/webcam). Use ``--no-perception-gate`` or
``MLX_GESTURE_PERCEPTION_GATE=0`` only to disable.

After **connect**, requests **720p / 30 fps / 5 Mbps** before **streamon**. Optional
``--enhance-stream``: cheap bilateral denoise + unsharp on each frame.

**No flight commands** in this script (no ROS / TCP).

Windows:
  1. Join Tello hotspot (``TELLO-XXXX``).
  2. ``python tello_view.py``  or  ``tello`` (after aliases).

Controls:
  Q  - quit
  S  - screenshot to Desktop

``tello_real_flight_test.py --onboard`` can show this HUD **after** takeoff (motors only for
that script’s climb/land). For **zero motors** with the same stack as ``simulate_drone``,
use **this** script (Tello on a stand is fine).
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from djitellopy import Tello

import hand_detection
from hand_detection import detect_hand
from perception_gating import (
    TrustedHandGate,
    describe_gate_load_failure,
    format_trust_hud_line,
    load_hand_landmarker,
    perception_gate_wanted,
    trusted_hand_config_tello_camera,
)
from simulate_drone import (
    BboxSmoother,
    COMMAND_COOLDOWN,
    CONFIDENCE_THRESHOLD,
    FOLLOW_EXIT_CONF,
    FOLLOW_EXIT_MARGIN,
    FOLLOW_LATCH_ENABLE,
    FOLLOW_NO_HAND_FRAMES_TO_DROP,
    GESTURE_LOCK_FRAMES,
    GESTURE_TO_COMMAND,
    GESTURE_UNLOCK_FRAMES,
    YUNET_FRAME_STRIDE,
    YUNET_MAX_INFER_SIDE,
    PADDING_RATIO,
    GestureFilter,
    classify_hand,
    draw_cam_panel,
    load_gesture_model,
    load_hand_detector,
)
from yunet_face import (
    ProximitySmoother,
    detect_largest_face,
    load_face_detector,
    proximity_from_bbox,
)

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR.parent / "models"

FACE_HAND_IOU_MAX = 0.22


def enhance_tello_frame_bgr(frame_bgr: np.ndarray) -> np.ndarray:
    """Mild bilateral denoise + unsharp for compressed Tello video (cheap, real-time)."""
    if frame_bgr is None or frame_bgr.size == 0:
        return frame_bgr
    bgr = cv2.bilateralFilter(frame_bgr, d=5, sigmaColor=40, sigmaSpace=40)
    blur = cv2.GaussianBlur(bgr, (0, 0), 1.0)
    return cv2.addWeighted(bgr, 1.25, blur, -0.25, 0)


def add_preview_arguments(p: argparse.ArgumentParser) -> None:
    """Same perception CLI as ``simulate_drone`` / onboard flight test."""
    p.add_argument(
        "--no-perception-gate",
        action="store_true",
        help="Disable TrustedHand (MediaPipe). Default: gate ON (Tello-tuned thresholds).",
    )
    p.add_argument(
        "--k-create",
        type=int,
        default=3,
        help="Trusted-hand K_CREATE (default matches Tello-tuned preset).",
    )
    p.add_argument(
        "--mp-miss-drop",
        type=int,
        default=7,
        help="Trusted-hand MP miss drop (default matches Tello-tuned preset).",
    )
    p.add_argument("--no-box-drop", type=int, default=5, help="Trusted-hand no-box drop.")
    p.add_argument(
        "--enhance-stream",
        action="store_true",
        help="Cheap preprocessing: bilateral denoise + unsharp on each frame before perception.",
    )


def parse_args():
    p = argparse.ArgumentParser(
        description="Tello camera + gesture HUD (same stack as simulate_drone; no flight)."
    )
    add_preview_arguments(p)
    return p.parse_args()


def init_perception(args: Any) -> dict[str, Any]:
    """Load YOLO, gesture model, optional TrustedHand gate, YuNet. No drone connection."""
    print("Loading models...")
    model, class_names = load_gesture_model()
    detector = load_hand_detector()
    tgate = None
    if perception_gate_wanted(args.no_perception_gate):
        tcfg = replace(
            trusted_hand_config_tello_camera(),
            k_create=args.k_create,
            mp_miss_drop=args.mp_miss_drop,
            no_box_drop=args.no_box_drop,
        )
        lm = load_hand_landmarker(MODEL_DIR, tcfg)
        if lm is not None:
            tgate = TrustedHandGate(lm, tcfg)
            print(
                "  Trusted-hand gate: ON — Tello-tuned MediaPipe (low bitrate / H.264)"
            )
            print(
                f"    K_CREATE={tcfg.k_create} MP_MISS_DROP={tcfg.mp_miss_drop} "
                f"NO_BOX_DROP={tcfg.no_box_drop}"
            )
            print(
                f"    MP_crop_min_side={tcfg.mp_infer_min_side} "
                f"mp_det={tcfg.mp_min_hand_detection_confidence:.2f} "
                f"mp_pres={tcfg.mp_min_hand_presence_confidence:.2f} "
                f"min_lm={tcfg.mp_min_landmarks}"
            )
        else:
            print(f"  Trusted-hand gate: OFF — {describe_gate_load_failure(MODEL_DIR)}")
    else:
        print("  Trusted-hand gate: OFF (--no-perception-gate or MLX_GESTURE_PERCEPTION_GATE=0)")

    face_detector = None
    yunet_load_error = None
    try:
        face_detector = load_face_detector(MODEL_DIR)
        print("  YuNet (hand/face gate + two_fingers follow preview)")
    except Exception as e:
        yunet_load_error = str(e)[:96]
        print(f"  YuNet unavailable ({yunet_load_error})")

    return {
        "model": model,
        "class_names": class_names,
        "detector": detector,
        "tgate": tgate,
        "face_detector": face_detector,
        "yunet_load_error": yunet_load_error,
    }


def run_preview_loop(
    tello: Tello,
    frame_reader: Any,
    args: Any,
    perception: dict[str, Any],
    battery: int,
    temp: int,
    *,
    source_label: str,
    footer_line: str,
) -> None:
    """OpenCV loop: same logic as simulate_drone Tello branch (HUD only)."""
    model = perception["model"]
    class_names = perception["class_names"]
    detector = perception["detector"]
    tgate = perception["tgate"]
    face_detector = perception["face_detector"]
    yunet_load_error = perception["yunet_load_error"]

    smoother = BboxSmoother(alpha=0.4, max_miss_frames=8)
    gfilter = GestureFilter(
        window=10,
        lock_frames=GESTURE_LOCK_FRAMES,
        unlock_frames=GESTURE_UNLOCK_FRAMES,
        min_vote_share=0.60,
    )
    proximity_smoother = ProximitySmoother(alpha=0.35)
    fps = 0.0
    prev_time = time.time()
    screenshot_n = 0
    telem_timer = time.time()
    yunet_frame_i = 0
    cached_face_fb = None
    follow_latched = False
    follow_no_hand_streak = 0
    active_command = "IDLE"
    last_command_time = 0.0

    print("HUD running. Q = quit this window.")
    print(
        f"GestureFilter: lock={GESTURE_LOCK_FRAMES} unlock={GESTURE_UNLOCK_FRAMES}; "
        f"conf {CONFIDENCE_THRESHOLD:.0%}; Trusted-hand: "
        f"{'ON' if tgate is not None else 'OFF'}"
    )

    try:
        while True:
            frame_bgr = frame_reader.frame
            if frame_bgr is None or frame_bgr.size == 0:
                time.sleep(0.01)
                continue

            if args.enhance_stream:
                frame_bgr = enhance_tello_frame_bgr(frame_bgr)

            if time.time() - telem_timer > 3.0:
                try:
                    battery = tello.get_battery()
                    temp = tello.get_temperature()
                except Exception:
                    pass
                telem_timer = time.time()

            now = time.time()
            dt = now - prev_time
            if dt > 0:
                fps = 0.85 * fps + 0.15 * (1.0 / dt)
            prev_time = now

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            face_fb = None
            if face_detector is not None:
                yunet_frame_i += 1
                if yunet_frame_i % YUNET_FRAME_STRIDE == 0 or cached_face_fb is None:
                    cached_face_fb, _ = detect_largest_face(
                        face_detector,
                        frame_bgr,
                        max_infer_side=YUNET_MAX_INFER_SIDE,
                    )
                face_fb = cached_face_fb

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
            confidence = 0.0
            cls_margin = 0.0

            if hand_crop is not None:
                idx, confidence, cls_margin, _ = classify_hand(
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

            new_command = None
            if cmd_gesture != "No hand":
                new_command = GESTURE_TO_COMMAND.get(cmd_gesture)

            if new_command and new_command != active_command:
                if (now - last_command_time) >= COMMAND_COOLDOWN:
                    active_command = new_command
                    last_command_time = now
                    print(f"  >>> [HUD] {active_command}  ({cmd_gesture} @ {confidence*100:.0f}%)")

            if cmd_gesture == "No hand":
                active_command = "IDLE"

            fh, fw = frame_bgr.shape[0], frame_bgr.shape[1]
            follow_arm = cmd_gesture == "two_fingers"
            # Match simulate: fist → STOP / hover → show face overlay when YuNet sees you.
            hover_face_scan = active_command == "STOP"
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

            trust_hud = format_trust_hud_line(frame_diag) if tgate is not None else None
            draw_cam_panel(
                frame_bgr,
                raw_gesture,
                cmd_gesture,
                confidence,
                bbox,
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
            )
            if (follow_arm or hover_face_scan) and face_fb is not None:
                fx1, fy1, fx2, fy2 = face_fb
                cv2.rectangle(frame_bgr, (fx1, fy1), (fx2, fy2), (80, 200, 255), 2)

            cv2.putText(
                frame_bgr,
                footer_line,
                (20, fh - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (100, 100, 100),
                1,
            )

            cv2.imshow("Tello Gesture View", frame_bgr)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            if key in (ord("s"), ord("S")):
                desktop = Path.home() / "Desktop"
                desktop.mkdir(exist_ok=True)
                fname = desktop / f"tello_gesture_{screenshot_n:03d}.png"
                cv2.imwrite(str(fname), frame_bgr)
                print(f"Screenshot saved: {fname}")
                screenshot_n += 1

    except KeyboardInterrupt:
        pass


def main():
    args = parse_args()

    print("=" * 50)
    print("  TELLO GESTURE VIEWER  (same perception stack as simulate_drone)")
    print("=" * 50)
    print()

    perception = init_perception(args)

    print()
    print("Connecting to Tello drone…")
    tello = Tello()
    tello.connect()

    battery = tello.get_battery()
    temp = tello.get_temperature()
    print(f"  Connected!  Battery: {battery}%  Temp: {temp}C")

    if battery < 15:
        print("  WARNING: Low battery.")

    print()
    print("  Requesting video: 720p, 30 fps, 5 Mbps…")
    try:
        tello.set_video_resolution(Tello.RESOLUTION_720P)
        tello.set_video_fps(Tello.FPS_30)
        tello.set_video_bitrate(Tello.BITRATE_5MBPS)
        print("  Video settings applied.")
    except Exception as e:
        print(f"  Video settings skipped ({e})")

    time.sleep(0.35)

    print()
    print("Starting video stream…")
    tello.streamon()
    frame_reader = tello.get_frame_read()
    print("Waiting for first valid frame", end="", flush=True)
    deadline = time.time() + 15
    while time.time() < deadline:
        f = frame_reader.frame
        if f is not None and f.size > 0:
            break
        print(".", end="", flush=True)
        time.sleep(0.15)
    else:
        print("\n  ERROR: No video.")
        tello.streamoff()
        tello.end()
        return
    print(" ready!\n")

    try:
        run_preview_loop(
            tello,
            frame_reader,
            args,
            perception,
            battery,
            temp,
            source_label="TELLO_CAM (preview only)",
            footer_line="[Q] quit   [S] screenshot   (no flight commands)",
        )
    finally:
        print()
        print("Shutting down…")
        tello.streamoff()
        tello.end()
        cv2.destroyAllWindows()
        print("Done.")


if __name__ == "__main__":
    main()
