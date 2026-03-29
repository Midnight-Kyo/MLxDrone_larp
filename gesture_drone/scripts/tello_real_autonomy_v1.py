r"""
Physical Tello — v1 autonomy (no ROS): **take off** → **IDLE** (zero sticks) until you gesture.
Confirmed **fist** starts **SEARCH** (slow yaw for a face) → **FACE_LOCK**.
**Thumbs up → climb** when ``GestureFilter`` promotes ``thumbs_up`` to ``_confirmed`` (same hysteresis as
other gestures), then one ``move_up(``\ ``--thumbs-up-cm``\ ``)``; **COMMAND_COOLDOWN** between climbs.
Confirmed **open_palm** **lands**. **Q** / Ctrl+C lands.

- Reuses ``tello_view.init_perception`` (YOLO + YuNet + GestureFilter + classifier).
- **TrustedHandGate is off** (``perception["tgate"] = None`` after init); **GestureFilter** uses
  ``AUTONOMY_GESTURE_*`` from ``simulate_drone`` (19 / 25).
- SEARCH default: **continuous RC yaw**; optional **--search-mode cw** for SDK rotate steps.
- After **FACE_LOCK**, if the face is lost long enough, behavior returns to **SEARCH** (not IDLE).

Pre-flight: preview **[T]** → optional climb **(cm)** → console **Enter** to take off.

Windows: join TELLO-XXXX Wi‑Fi, clear space, then ``python gesture_drone/scripts/tello_real_autonomy_v1.py``.

Preview CLI flags match ``tello_view.add_preview_arguments`` (see ``--help``); TrustedHand args are ignored.
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from typing import Any

import cv2
import tello_view
from djitellopy import Tello
from hand_detection import detect_hand
from search_behavior import (
    EPS_X,
    M_ACQUIRE,
    M_LOSS,
    MAX_ANG_LOCK,
    KP_LOCK,
    face_ok_and_x_norm,
)
from simulate_drone import (
    AUTONOMY_GESTURE_LOCK_FRAMES,
    AUTONOMY_GESTURE_UNLOCK_FRAMES,
    BboxSmoother,
    COMMAND_COOLDOWN,
    CONFIDENCE_THRESHOLD,
    GESTURE_TO_COMMAND,
    YUNET_FRAME_STRIDE,
    YUNET_MAX_INFER_SIDE,
    PADDING_RATIO,
    GestureFilter,
    classify_hand,
    draw_cam_panel,
)
from yunet_face import detect_largest_face

FACE_HAND_IOU_MAX = 0.22

# Conservative RC yaw magnitude (-100..100). Tune via CLI.
DEFAULT_SEARCH_YAW_RC = 16
DEFAULT_LOCK_YAW_MAX_RC = 28

# After takeoff + settle, optional ``move_up`` before IDLE (cm); capped for safety.
MAX_CLIMB_AFTER_TAKEOFF_CM = 200
MAX_THUMBS_UP_CM = 200

# Tello SDK: ``move_*`` distance must be 20–500 cm. Below 20 is rejected ("out of range").
MIN_SDK_MOVE_CM = 20
MAX_SDK_MOVE_CM = 500

# Pause RC updates before ``move_*`` so the firmware is not in joystick mode.
RC_GAP_BEFORE_MOVE_UP_S = 0.45
RC_GAP_AFTER_MOVE_UP_S = 0.12

_stop_requested = False


def _on_sigint(signum, frame) -> None:  # noqa: ARG001
    global _stop_requested
    _stop_requested = True
    print("\n  [!] Interrupt — will zero RC and land if airborne.", flush=True)


def _clamp_rc(v: int) -> int:
    return max(-100, min(100, int(v)))


def _send_yaw_only(tello: Tello, yaw_rc: int) -> None:
    """Hover: all translation axes zero; yaw-only per v1 spec."""
    tello.send_rc_control(0, 0, 0, _clamp_rc(yaw_rc))


def _sdk_move_dist_cm(dist: int, cap_cm: int) -> int:
    """Clamp distance for ``move_up`` / ``move_*``. Zero skips; else in [MIN_SDK, cap]."""
    if dist <= 0:
        return 0
    upper = min(cap_cm, MAX_SDK_MOVE_CM)
    return max(MIN_SDK_MOVE_CM, min(upper, dist))


def _move_up_with_rc_gap(tello: Tello, dist_cm: int) -> None:
    """Do not call ``send_rc_control`` immediately before ``move_up`` — sleep first, then resume RC after."""
    if dist_cm <= 0:
        return
    time.sleep(RC_GAP_BEFORE_MOVE_UP_S)
    tello.move_up(int(dist_cm))
    time.sleep(RC_GAP_AFTER_MOVE_UP_S)


def _hud_command_label(state: str, cmd_gesture: str) -> str:
    """Third line in ``draw_cam_panel`` (below gesture)."""
    if state == "TAKEOFF":
        return "SETTLE"
    if cmd_gesture == "thumbs_up":
        return GESTURE_TO_COMMAND.get("thumbs_up", "MOVE UP")
    if cmd_gesture == "open_palm":
        return GESTURE_TO_COMMAND.get("open_palm", "LAND")
    if state == "IDLE" and cmd_gesture == "fist":
        return "START SCAN"
    if state == "SEARCH":
        return "SCANNING"
    if state == "FACE_LOCK":
        return "FACE LOCK"
    if state == "IDLE":
        return "HOVER"
    return state


def _prompt_climb_after_takeoff_cm(default: int) -> int:
    """Console prompt after preview [T]. Empty line keeps ``default``."""
    print(
        "\n--- Climb after takeoff (optional, before IDLE) ---\n"
        "  Enter centimeters for one ``move_up`` after takeoff (head height), or 0 to skip.\n"
        f"  Tello requires {MIN_SDK_MOVE_CM}-{MAX_SDK_MOVE_CM} cm; values below {MIN_SDK_MOVE_CM} "
        f"are raised to {MIN_SDK_MOVE_CM}.\n"
        f"  Flag default (--climb-after-takeoff-cm): {default}"
    )
    raw = input(f"  Climb cm [default {default}]: ").strip()
    if not raw:
        v = default
    else:
        try:
            v = int(raw, 10)
        except ValueError:
            print(f"  Invalid number — using default {default}.")
            v = default
    return max(0, min(MAX_CLIMB_AFTER_TAKEOFF_CM, v))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Tello v1: takeoff -> IDLE; fist=SEARCH; thumbs_up=move_up; palm=land. No ROS."
        )
    )
    tello_view.add_preview_arguments(p)
    p.add_argument(
        "--min-battery",
        type=int,
        default=20,
        help="Abort before takeoff if battery below this (default 20).",
    )
    p.add_argument(
        "--settle-s",
        type=float,
        default=2.5,
        help="Seconds after takeoff with zero yaw RC before climb/SEARCH (default 2.5).",
    )
    p.add_argument(
        "--search-mode",
        choices=("rc", "cw"),
        default="rc",
        help=(
            "SEARCH spin: 'rc' = continuous yaw via send_rc_control (default, smooth). "
            "'cw' = discrete rotate_clockwise/ccw SDK commands (pauses feed each step)."
        ),
    )
    p.add_argument(
        "--search-cw-degrees",
        type=int,
        default=15,
        help="SEARCH cw-mode: degrees per step, 1-360 (default 15).",
    )
    p.add_argument(
        "--search-cw-interval",
        type=float,
        default=0.5,
        help="SEARCH cw-mode: minimum seconds between completed steps (default 0.5).",
    )
    p.add_argument(
        "--search-yaw-rc",
        type=int,
        default=DEFAULT_SEARCH_YAW_RC,
        help=(
            f"SEARCH rc-mode only: yaw stick -100..100 (default {DEFAULT_SEARCH_YAW_RC}). "
            "Sign = direction. For cw-mode, sign chooses cw vs ccw only."
        ),
    )
    p.add_argument(
        "--lock-yaw-max-rc",
        type=int,
        default=DEFAULT_LOCK_YAW_MAX_RC,
        help="FACE_LOCK: max |yaw| RC when correcting (default 28).",
    )
    p.add_argument(
        "--climb-after-takeoff-cm",
        type=int,
        default=0,
        help=(
            "After takeoff + settle, move_up this many cm before IDLE (0 = skip). "
            "Console prompt after [T] can override."
        ),
    )
    p.add_argument(
        "--thumbs-up-cm",
        type=int,
        default=20,
        help=(
            "After confirmed thumbs_up: one move_up by this many cm (default 20; "
            f"Tello requires {MIN_SDK_MOVE_CM}..{MAX_SDK_MOVE_CM} cm; "
            f"clamped {MIN_SDK_MOVE_CM}..{MAX_THUMBS_UP_CM})."
        ),
    )
    args = p.parse_args()
    args.search_yaw_rc = max(-100, min(100, args.search_yaw_rc))
    args.lock_yaw_max_rc = max(1, min(100, args.lock_yaw_max_rc))
    args.search_cw_degrees = max(1, min(360, args.search_cw_degrees))
    args.search_cw_interval = max(0.15, float(args.search_cw_interval))
    args.climb_after_takeoff_cm = max(
        0, min(MAX_CLIMB_AFTER_TAKEOFF_CM, int(args.climb_after_takeoff_cm))
    )
    args.thumbs_up_cm = max(
        MIN_SDK_MOVE_CM, min(MAX_THUMBS_UP_CM, int(args.thumbs_up_cm))
    )
    return args


def run_pre_takeoff_preview(
    tello: Tello,
    frame_reader: Any,
    perception: dict[str, Any],
    args: argparse.Namespace,
    initial_bat: int,
    initial_tmp: int,
) -> bool:
    """
    Ground-safe loop: same perception + HUD as flight, no RC / no takeoff.
    [T] in the video window = operator ready; then Enter in console to take off.
    [Q] = disconnect without flying. Ctrl+C = same as Q for preview purposes.
    Returns True to continue to takeoff prompt, False to abort.
    """
    global _stop_requested
    model = perception["model"]
    class_names = perception["class_names"]
    detector = perception["detector"]
    face_detector = perception["face_detector"]
    yunet_load_error = perception["yunet_load_error"]

    smoother = BboxSmoother(alpha=0.4, max_miss_frames=8)
    gfilter = GestureFilter(
        window=10,
        lock_frames=AUTONOMY_GESTURE_LOCK_FRAMES,
        unlock_frames=AUTONOMY_GESTURE_UNLOCK_FRAMES,
        min_vote_share=0.60,
    )
    telem_timer = time.time()
    prev_time = time.time()
    fps = 0.0
    yunet_frame_i = 0
    cached_face_pack: tuple[Any, float] | None = None
    battery = initial_bat
    temp = initial_tmp

    print("\n--- Pre-flight preview (drone on ground, no motor commands) ---")
    print("  In the video window: [T] = feed OK -> console for climb + takeoff")
    print("                       [Q] = quit without flying")
    print("  Ctrl+C also aborts preview.")

    try:
        while True:
            if _stop_requested:
                _stop_requested = False
                print("  Abort (interrupt) — disconnecting.")
                return False

            frame_bgr = frame_reader.frame
            if frame_bgr is None or frame_bgr.size == 0:
                time.sleep(0.01)
                continue

            now = time.time()
            dt = now - prev_time
            if dt > 0:
                fps = 0.85 * fps + 0.15 * (1.0 / dt)
            prev_time = now

            if now - telem_timer > 3.0:
                try:
                    battery = tello.get_battery()
                    temp = tello.get_temperature()
                except Exception:
                    pass
                telem_timer = now

            fh, fw = frame_bgr.shape[0], frame_bgr.shape[1]
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            face_fb = None
            face_score = 0.0
            if face_detector is not None:
                yunet_frame_i += 1
                if yunet_frame_i % YUNET_FRAME_STRIDE == 0 or cached_face_pack is None:
                    fb, sc = detect_largest_face(
                        face_detector,
                        frame_bgr,
                        max_infer_side=YUNET_MAX_INFER_SIDE,
                    )
                    cached_face_pack = (fb, float(sc or 0.0))
                face_fb, face_score = cached_face_pack

            hand_crop, bbox, frame_diag = detect_hand(
                detector,
                frame_rgb,
                smoother,
                face_xyxy=face_fb,
                face_hand_iou_max=FACE_HAND_IOU_MAX,
                padding_ratio=PADDING_RATIO,
            )
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

            face_ok, face_x_norm = face_ok_and_x_norm(face_fb, face_score, fw, fh)

            trust_hud = None
            beh_line = (
                "PREVIEW | [T] ready -> Enter in console to take off | "
                f"f_ok={int(face_ok)} x={face_x_norm:+.2f} | "
                f"classifier>={CONFIDENCE_THRESHOLD:.0%} | bat={battery}%"
            )
            draw_cam_panel(
                frame_bgr,
                raw_gesture,
                cmd_gesture,
                confidence,
                bbox,
                "IDLE",
                fps,
                gfilter,
                source_label="TELLO autonomy v1 (preview)",
                battery=battery,
                temp=temp,
                follow_preview=False,
                face_proximity=None,
                face_tracked=bool(face_ok and face_fb is not None),
                yunet_error=yunet_load_error,
                trust_line=trust_hud,
                beh_line=beh_line,
            )
            cv2.putText(
                frame_bgr,
                "[T] console: climb cm + takeoff   [Q] quit",
                (12, fh - 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (120, 220, 255),
                1,
            )

            cv2.imshow("Tello Autonomy v1", frame_bgr)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                print("  [Q] — quit preview, no takeoff.")
                return False
            if key in (ord("t"), ord("T")):
                print("  [T] — preview OK. Use the console for the takeoff step.")
                return True
    except KeyboardInterrupt:
        print("\n  Preview aborted (KeyboardInterrupt).")
        return False


def main() -> int:
    global _stop_requested
    args = parse_args()
    signal.signal(signal.SIGINT, _on_sigint)

    try:
        perception = tello_view.init_perception(args)
    except Exception as e:
        print(f"ERROR: perception init failed: {e}", file=sys.stderr)
        return 1
    perception["tgate"] = None
    if perception["face_detector"] is None:
        print(
            "ERROR: YuNet face detector required for SEARCH / FACE_LOCK. Fix ONNX under models/.",
            file=sys.stderr,
        )
        return 1

    print("\n=== Tello real autonomy v1 (yaw-only) ===")
    print("Join TELLO Wi-Fi, clear volume around drone, keep hands clear during takeoff.")
    print(
        "Flow: preview [T] -> climb -> takeoff -> IDLE -> fist=SEARCH; thumbs_up=move_up; palm=land "
        "(thumbs/FACE_LOCK too)."
    )
    print("In flight: OPEN PALM (confirmed) to land; Q / Ctrl+C lands.")
    input("Press Enter to CONNECT (no takeoff yet)...")

    tello = Tello()
    tello.connect()
    bat = tello.get_battery()
    tmp = tello.get_temperature()
    print(f"Battery: {bat}%   Temp: {tmp}°C")
    if bat < args.min_battery:
        print(f"Aborting: battery < {args.min_battery}%")
        tello.end()
        return 2

    print("\nStarting video stream...")
    tello.streamon()
    frame_reader = tello.get_frame_read()
    deadline = time.time() + 15.0
    while time.time() < deadline:
        f = frame_reader.frame
        if f is not None and f.size > 0:
            break
        time.sleep(0.1)
    else:
        print("ERROR: no video stream.")
        tello.streamoff()
        tello.end()
        return 3

    model = perception["model"]
    class_names = perception["class_names"]
    detector = perception["detector"]
    face_detector = perception["face_detector"]
    yunet_load_error = perception["yunet_load_error"]

    smoother = BboxSmoother(alpha=0.4, max_miss_frames=8)
    gfilter = GestureFilter(
        window=10,
        lock_frames=AUTONOMY_GESTURE_LOCK_FRAMES,
        unlock_frames=AUTONOMY_GESTURE_UNLOCK_FRAMES,
        min_vote_share=0.60,
    )

    airborne = False
    acq_streak = 0
    loss_streak = 0
    settle_deadline = 0.0
    last_land_fire = 0.0
    last_start_search_time = 0.0
    last_move_up_time = 0.0
    telem_timer = time.time()
    prev_time = time.time()
    fps = 0.0
    yunet_frame_i = 0
    cached_face_pack: tuple[Any, float] | None = None
    battery = bat
    temp = tmp
    last_yaw_rc = 0

    if not run_pre_takeoff_preview(
        tello, frame_reader, perception, args, bat, tmp
    ):
        try:
            tello.streamoff()
        except Exception:
            pass
        try:
            tello.end()
        except Exception:
            pass
        cv2.destroyAllWindows()
        print("Disconnected (no takeoff).")
        return 0

    cv2.destroyAllWindows()
    climb_pending_cm = _prompt_climb_after_takeoff_cm(args.climb_after_takeoff_cm)
    if climb_pending_cm > 0:
        print(f"  Will climb {climb_pending_cm} cm after takeoff settle, then IDLE (zero sticks).")

    print(
        "\nPreview closed. Takeoff is next: put the video window aside and use this console.\n"
        "Press Enter to TAKEOFF — **IDLE** until **fist** starts SEARCH; **thumbs_up** climbs."
    )
    input()

    try:
        tello.takeoff()
        airborne = True
        settle_deadline = time.time() + max(0.0, args.settle_s)
        state = "TAKEOFF"
        print(
            f"  Takeoff issued — settling {args.settle_s:.1f}s with zero RC, then **IDLE** "
            f"(fist->SEARCH, thumbs_up->move_up {args.thumbs_up_cm}cm)."
        )
        print(
            f"  When SEARCH runs: {args.search_mode}"
            + (
                f" ({args.search_cw_degrees}° / {args.search_cw_interval:.2f}s, sign={'cw' if args.search_yaw_rc >= 0 else 'ccw'})"
                if args.search_mode == "cw"
                else f" (yaw_rc={args.search_yaw_rc})"
            )
        )

        last_search_cw_time = 0.0

        while True:
            frame_bgr = frame_reader.frame
            if frame_bgr is None or frame_bgr.size == 0:
                time.sleep(0.01)
                continue

            now = time.time()
            dt = now - prev_time
            if dt > 0:
                fps = 0.85 * fps + 0.15 * (1.0 / dt)
            prev_time = now

            if now - telem_timer > 3.0:
                try:
                    battery = tello.get_battery()
                    temp = tello.get_temperature()
                except Exception:
                    pass
                telem_timer = now
                if airborne and battery < max(10, args.min_battery - 5):
                    print(f"  [!] Low battery ({battery}%) — landing.")
                    state = "LAND"

            fh, fw = frame_bgr.shape[0], frame_bgr.shape[1]
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            face_fb = None
            face_score = 0.0
            if face_detector is not None:
                yunet_frame_i += 1
                if yunet_frame_i % YUNET_FRAME_STRIDE == 0 or cached_face_pack is None:
                    fb, sc = detect_largest_face(
                        face_detector,
                        frame_bgr,
                        max_infer_side=YUNET_MAX_INFER_SIDE,
                    )
                    cached_face_pack = (fb, float(sc or 0.0))
                face_fb, face_score = cached_face_pack

            hand_crop, bbox, frame_diag = detect_hand(
                detector,
                frame_rgb,
                smoother,
                face_xyxy=face_fb,
                face_hand_iou_max=FACE_HAND_IOU_MAX,
                padding_ratio=PADDING_RATIO,
            )
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

            face_ok, face_x_norm = face_ok_and_x_norm(face_fb, face_score, fw, fh)

            if _stop_requested:
                state = "LAND"

            if state in ("IDLE", "SEARCH", "FACE_LOCK"):
                if gfilter._confirmed == "open_palm" and (now - last_land_fire) >= COMMAND_COOLDOWN:
                    print("  Land trigger: open_palm (confirmed)")
                    state = "LAND"
                    last_land_fire = now

            if state in ("IDLE", "SEARCH", "FACE_LOCK", "TAKEOFF"):
                if gfilter._confirmed == "thumbs_up" and (
                    now - last_move_up_time
                ) >= COMMAND_COOLDOWN:
                    up_cm = _sdk_move_dist_cm(args.thumbs_up_cm, MAX_THUMBS_UP_CM)
                    print(
                        f"  Climb trigger: thumbs_up (confirmed) -> move_up({up_cm} cm)",
                        flush=True,
                    )
                    try:
                        _move_up_with_rc_gap(tello, up_cm)
                    except Exception as e:
                        print(f"  [!] move_up (thumbs_up) failed: {e}", flush=True)
                    last_move_up_time = now

            if state == "IDLE":
                if gfilter._confirmed == "fist" and (now - last_start_search_time) >= COMMAND_COOLDOWN:
                    print("  Start SEARCH: fist (confirmed)", flush=True)
                    state = "SEARCH"
                    acq_streak = 0
                    loss_streak = 0
                    last_search_cw_time = now - args.search_cw_interval
                    last_start_search_time = now

            if state == "LAND":
                try:
                    _send_yaw_only(tello, 0)
                except Exception:
                    pass
                print("  Landing...")
                tello.land()
                airborne = False
                break

            yaw_rc = 0
            if state == "TAKEOFF":
                yaw_rc = 0
                if now >= settle_deadline:
                    if climb_pending_cm > 0:
                        up_cm = _sdk_move_dist_cm(
                            climb_pending_cm, MAX_CLIMB_AFTER_TAKEOFF_CM
                        )
                        if up_cm > 0:
                            print(
                                f"  Climb: move_up({up_cm} cm), then IDLE…",
                                flush=True,
                            )
                            try:
                                _move_up_with_rc_gap(tello, up_cm)
                            except Exception as e:
                                print(f"  [!] move_up failed: {e}", flush=True)
                        climb_pending_cm = 0
                        time.sleep(1.2)
                    state = "IDLE"
                    acq_streak = 0
                    loss_streak = 0
                    print("  State -> IDLE (fist->SEARCH; thumbs_up->up; palm->land)")
                else:
                    _send_yaw_only(tello, 0)

            elif state == "IDLE":
                yaw_rc = 0
                _send_yaw_only(tello, 0)

            elif state == "SEARCH":
                yaw_rc = 0
                if args.search_mode == "cw":
                    _send_yaw_only(tello, 0)
                    if now - last_search_cw_time >= args.search_cw_interval:
                        deg = args.search_cw_degrees
                        try:
                            if args.search_yaw_rc >= 0:
                                tello.rotate_clockwise(deg)
                            else:
                                tello.rotate_counter_clockwise(deg)
                        except Exception as e:
                            print(f"  [!] SEARCH cw/ccw failed: {e}", flush=True)
                        last_search_cw_time = time.time()
                else:
                    yaw_rc = _clamp_rc(args.search_yaw_rc)
                    _send_yaw_only(tello, yaw_rc)
                if face_ok:
                    acq_streak += 1
                    if acq_streak >= M_ACQUIRE:
                        state = "FACE_LOCK"
                        acq_streak = 0
                        loss_streak = 0
                        print("  State -> FACE_LOCK")
                else:
                    acq_streak = 0

            elif state == "FACE_LOCK":
                if face_ok:
                    loss_streak = 0
                    ex = face_x_norm
                    if abs(ex) > EPS_X:
                        rate_rad = -KP_LOCK * ex
                        rate_rad = max(-MAX_ANG_LOCK, min(MAX_ANG_LOCK, rate_rad))
                        yaw_rc = int(
                            round((rate_rad / MAX_ANG_LOCK) * float(args.lock_yaw_max_rc))
                        )
                        yaw_rc = _clamp_rc(yaw_rc)
                    else:
                        yaw_rc = 0
                else:
                    loss_streak += 1
                    yaw_rc = 0
                    if loss_streak >= M_LOSS:
                        state = "SEARCH"
                        acq_streak = 0
                        loss_streak = 0
                        last_search_cw_time = now - args.search_cw_interval
                        print("  State -> SEARCH (face loss)")

                _send_yaw_only(tello, yaw_rc)

            last_yaw_rc = yaw_rc

            trust_hud = None
            if state == "IDLE":
                search_note = (
                    f"fist->SEARCH thumbs->{args.thumbs_up_cm}cm palm->LAND (confirmed)"
                )
            elif state == "SEARCH" and args.search_mode == "cw":
                search_note = f"SEARCH=cw {args.search_cw_degrees}deg/{args.search_cw_interval:.2f}s"
            elif state == "SEARCH":
                search_note = f"SEARCH=rc yaw={last_yaw_rc}"
            else:
                search_note = f"yaw_rc={last_yaw_rc}"
            beh_line = (
                f"{state}: {search_note} | "
                f"acq={acq_streak}/{M_ACQUIRE} loss={loss_streak}/{M_LOSS} | "
                f"f_ok={int(face_ok)} x={face_x_norm:+.2f} | bat={battery}%"
            )
            draw_cam_panel(
                frame_bgr,
                raw_gesture,
                cmd_gesture,
                confidence,
                bbox,
                _hud_command_label(state, cmd_gesture),
                fps,
                gfilter,
                source_label="TELLO autonomy v1",
                battery=battery,
                temp=temp,
                follow_preview=False,
                face_proximity=None,
                face_tracked=bool(face_ok and face_fb is not None),
                yunet_error=yunet_load_error,
                trust_line=trust_hud,
                beh_line=beh_line,
            )
            if state == "IDLE":
                foot = "[Q] land  fist->SEARCH  thumbs->up  palm=LAND"
            elif state in ("SEARCH", "FACE_LOCK"):
                foot = f"[Q] land  thumbs->up {args.thumbs_up_cm}cm  palm=LAND"
            else:
                foot = "[Q] land   palm (confirmed)=land"
            cv2.putText(
                frame_bgr,
                foot,
                (12, fh - 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (180, 180, 255),
                1,
            )

            cv2.imshow("Tello Autonomy v1", frame_bgr)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                print("  Q pressed — landing.")
                state = "LAND"

    except KeyboardInterrupt:
        print("\n  KeyboardInterrupt — landing if airborne.")
        state = "LAND"
    finally:
        try:
            _send_yaw_only(tello, 0)
        except Exception:
            pass
        if airborne:
            try:
                tello.land()
            except Exception as e:
                print(f"  Land in finally failed: {e}", file=sys.stderr)
        try:
            tello.streamoff()
        except Exception:
            pass
        try:
            tello.end()
        except Exception:
            pass
        cv2.destroyAllWindows()

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
