r"""
**Physical Tello** flight test (djitellopy): takeoff → hover → you land.

**Default:** minimal — no ML, no OpenCV, just ``takeoff()`` / ``land()``.

**Same perception as ``simulate_drone`` but Tello camera and *no motors at all*:**
use ``tello_view.py`` (drone on ground / desk). That is the “dummy” camera check.

**``--onboard``** (optional): after ``takeoff``, run the simulate-style HUD on the
drone camera; gestures still do **not** command motors. Press **Q**, then type
``land`` in the terminal.

Later you can script: fly up → slow yaw search → land — keep that separate and
simple.

Trusted-hand flags: ``--no-perception-gate``, ``--k-create``, … (with ``--onboard``).
"""

from __future__ import annotations

import argparse
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Real Tello takeoff/hover/land. Default = minimal. Use tello_view.py for camera-only simulate stack."
    )
    parser.add_argument(
        "--onboard",
        action="store_true",
        help="After takeoff, open simulate-style gesture HUD (still no gesture→motor commands).",
    )
    parser.add_argument(
        "--min-battery",
        type=int,
        default=20,
        help="Refuse takeoff if battery below this (default: 20).",
    )
    parser.add_argument(
        "--after-takeoff-s",
        type=float,
        default=2.0,
        help="Seconds to settle after takeoff before HUD or land prompt (default: 2).",
    )

    try:
        import tello_view
    except ImportError:
        tello_view = None  # type: ignore

    if tello_view is not None:
        tello_view.add_preview_arguments(parser)

    args = parser.parse_args()

    try:
        from djitellopy import Tello
    except ImportError:
        print("ERROR: pip install djitellopy", file=sys.stderr)
        return 1

    print("=== Real Tello flight test (djitellopy) ===")
    if args.onboard:
        print("Mode: --onboard (simulate-style HUD after takeoff; gestures = preview only)")
    else:
        print("Mode: minimal (takeoff / land only). For camera+simulate stack on the ground: tello_view.py")
    print("Join Tello Wi‑Fi, clear space above, then Enter to CONNECT.")
    input()

    tello = Tello()
    tello.connect()
    bat = tello.get_battery()
    temp = tello.get_temperature()
    print(f"Battery: {bat}%   Temp: {temp}°C")

    if bat < args.min_battery:
        print(f"Aborting: battery < {args.min_battery}%")
        try:
            tello.end()
        except Exception:
            pass
        return 2

    perception = None
    if args.onboard:
        if tello_view is None:
            print("ERROR: cannot import tello_view (onboard mode).", file=sys.stderr)
            tello.end()
            return 3
        print()
        perception = tello_view.init_perception(args)

    print()
    print("Enter to TAKEOFF (DJI auto climb, then hover).")
    input()

    try:
        tello.takeoff()
        time.sleep(max(0.0, args.after_takeoff_s))

        if not args.onboard:
            print()
            print("Hovering. Type land (or l) + Enter. Ctrl+C tries land.")
            while True:
                try:
                    line = input("> ").strip().lower()
                except EOFError:
                    break
                if line in ("land", "l", "q", "quit"):
                    break
        elif args.onboard:
            print()
            print("Starting camera + onboard HUD (TrustedHand ON unless you disabled it).")
            tello.streamon()
            frame_reader = tello.get_frame_read()
            print("Waiting for video…", end="", flush=True)
            deadline = time.time() + 15
            while time.time() < deadline:
                f = frame_reader.frame
                if f is not None and f.size > 0:
                    print(" OK")
                    break
                print(".", end="", flush=True)
                time.sleep(0.15)
            else:
                print("\nERROR: no video. Landing.")
                tello.land()
                tello.streamoff()
                tello.end()
                return 4

            bat2 = tello.get_battery()
            tmp2 = tello.get_temperature()
            try:
                tello_view.run_preview_loop(
                    tello,
                    frame_reader,
                    args,
                    perception,
                    bat2,
                    tmp2,
                    source_label="TELLO onboard (flying; HUD only)",
                    footer_line="[Q] close HUD   [S] screenshot   (gestures do not fly)",
                )
            finally:
                try:
                    tello.streamoff()
                except Exception:
                    pass

            print()
            print("HUD closed. Type land (or l) + Enter to LAND. Ctrl+C tries land.")
            while True:
                try:
                    line = input("> ").strip().lower()
                except EOFError:
                    break
                if line in ("land", "l", "q", "quit"):
                    break

        print("Landing …")
        tello.land()

    except KeyboardInterrupt:
        print("\nInterrupt — landing …")
        try:
            tello.land()
        except Exception as e:
            print(f"Land after interrupt failed: {e}", file=sys.stderr)
    finally:
        try:
            tello.end()
        except Exception:
            pass

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
