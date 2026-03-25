#!/usr/bin/env python3
"""
Minimal Tello hover stability test: connect → takeoff → hover 10s (no rc) → land.
No camera, no perception, no RC after takeoff. Join TELLO-XXXX Wi‑Fi first.
"""

import sys
import time


def main() -> int:
    try:
        from djitellopy import Tello
    except ImportError:
        print("Install: pip install djitellopy", file=sys.stderr)
        return 1

    tello = Tello()
    flying = False

    try:
        tello.connect()
        bat = tello.get_battery()
        print(f"Connected. Battery: {bat}%")
        input("Press Enter to take off...")

        tello.takeoff()
        flying = True
        print("Hovering...")
        time.sleep(10)
        print("Landing...")
        tello.land()
        flying = False
    except KeyboardInterrupt:
        print("\nCtrl+C — emergency land if airborne.")
        if flying:
            try:
                tello.land()
            except Exception:
                pass
            flying = False
    finally:
        try:
            tello.end()
        except Exception:
            pass

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
