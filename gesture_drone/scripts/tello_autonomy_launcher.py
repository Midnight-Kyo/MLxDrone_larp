"""Run ``tello_real_autonomy_v1`` after a visible banner.

The real script imports PyTorch, Ultralytics YOLO, OpenCV, etc. at module load — that
phase can take tens of seconds on Windows with **no** output if you execute the module
directly. This launcher prints first, then imports.
"""

from __future__ import annotations

import sys


def main() -> None:
    print(
        "\nTello autonomy — starting Python.\n"
        "Next: loading ML stack (PyTorch, Ultralytics, OpenCV, YuNet). "
        "First cold start is often 30–120 s with no further lines here — that is normal.\n",
        flush=True,
    )
    try:
        import tello_real_autonomy_v1 as autonomy  # noqa: PLC0415
    except ImportError as e:
        print(f"ERROR: failed to import tello_real_autonomy_v1: {e}", file=sys.stderr)
        raise SystemExit(1) from e
    raise SystemExit(autonomy.main())


if __name__ == "__main__":
    main()
