#!/usr/bin/env python3
"""
Summarize gesture_drone/logs/session_*.csv (from simulate_drone) to debug command flips.

Usage:
  python analyze_session_log.py /path/to/session_YYYYMMDD_HHMMSS.csv

Prints: reject_stage counts, raw vs confirmed mismatches, margin stats for noisy classes.
"""

from __future__ import annotations

import csv
import statistics
import sys
from collections import Counter
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python analyze_session_log.py <session_*.csv>")
        sys.exit(1)
    path = Path(sys.argv[1])
    if not path.is_file():
        print(f"Not found: {path}")
        sys.exit(1)

    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    n = len(rows)
    print(f"Frames: {n}  ({path.name})\n")

    # reject_stage (if column exists)
    rs_key = "reject_stage"
    if rows and rs_key in rows[0]:
        c = Counter(r.get(rs_key, "") for r in rows)
        print("reject_stage:")
        for k, v in c.most_common():
            print(f"  {k or '(empty)'}: {v}")
        print()

    raw_key, conf_key = "raw_gesture", "confirmed_gesture"
    if rows and raw_key in rows[0]:
        mismatch = sum(
            1
            for r in rows
            if r.get(raw_key) and r.get(conf_key)
            and r[raw_key] != r[conf_key]
            and r[conf_key] != ""
        )
        print(f"raw != confirmed (filter disagrees): {mismatch} frames")

        fist_raw = [
            r for r in rows
            if r.get(raw_key) == "fist"
        ]
        print(f"raw_gesture=fist (any): {len(fist_raw)} frames")
        if fist_raw and "classifier_margin" in fist_raw[0]:
            margins = []
            for r in fist_raw:
                try:
                    margins.append(float(r["classifier_margin"]))
                except (TypeError, ValueError):
                    pass
            if margins:
                print(
                    f"  margin when raw=fist: min={min(margins):.3f} "
                    f"median={statistics.median(margins):.3f} max={max(margins):.3f}"
                )
        print()

    ac_key = "active_command"
    if rows and ac_key in rows[0]:
        c = Counter(r.get(ac_key, "") for r in rows)
        print("active_command (sampled every frame):")
        for k, v in c.most_common(12):
            print(f"  {k}: {v}")
        print()

    if rows and "cmd_gesture" in rows[0] and "confirmed_gesture" in rows[0]:
        latch_hold = sum(
            1
            for r in rows
            if r.get("cmd_gesture") == "two_fingers"
            and r.get("confirmed_gesture")
            and r["confirmed_gesture"] != "two_fingers"
        )
        print(
            f"cmd_gesture=two_fingers while confirmed≠two_fingers (follow latch): {latch_hold} frames"
        )


if __name__ == "__main__":
    main()
