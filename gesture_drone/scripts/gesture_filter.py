"""
Time-based gesture confirmation for hand-command pipelines.

Vote window and lock/unlock are measured in seconds (``time.monotonic()``) so
confirmation latency is consistent across FPS and CPU load.
"""

from __future__ import annotations

import time

# Defaults: predictable real-flight feel (~2 s to confirm, ~2.5 s to switch away).
GESTURE_WINDOW_SECONDS = 0.5   # sliding vote uses roughly the last N seconds
GESTURE_LOCK_SECONDS = 2.0     # sustained winning gesture to confirm
GESTURE_UNLOCK_SECONDS = 2.5  # sustained different winner to change confirmation


class GestureFilter:
    """
    Confidence-weighted, recency-biased sliding **time** window with dead-band and
    asymmetric hysteresis (lock vs unlock duration in seconds).
    """

    def __init__(
        self,
        *,
        window_duration_s: float = GESTURE_WINDOW_SECONDS,
        lock_seconds: float = GESTURE_LOCK_SECONDS,
        unlock_seconds: float = GESTURE_UNLOCK_SECONDS,
        min_vote_share: float = 0.60,
    ):
        self._window_duration_s = max(1e-6, float(window_duration_s))
        self._lock_seconds = max(0.0, float(lock_seconds))
        self._unlock_seconds = max(0.0, float(unlock_seconds))
        self._min_vote_share = float(min_vote_share)
        self._history: list[tuple[float, str, float]] = []
        self._confirmed: str | None = None
        self._streak_cand: str | None = None
        self._streak_since: float | None = None
        self._active_threshold_sec = self._lock_seconds

    def update(self, gesture, confidence=1.0, t: float | None = None):
        """Feed one sample (gesture=None means no hand). Returns confirmed gesture name or None."""
        now = time.monotonic() if t is None else float(t)
        label = gesture if gesture is not None else "__none__"
        conf = float(confidence) if confidence is not None else 0.0

        self._history.append((now, label, conf))
        cutoff = now - self._window_duration_s
        self._history = [h for h in self._history if h[0] >= cutoff]
        if len(self._history) > 512:
            self._history = self._history[-512:]

        vote_totals: dict[str, float] = {}
        total_weight = 0.0
        for ts, g, c in self._history:
            age_ratio = min(
                1.0,
                max(0.0, (now - ts) / self._window_duration_s),
            )
            recency = 0.5 + 0.5 * (1.0 - age_ratio)
            w = c * recency
            vote_totals[g] = vote_totals.get(g, 0.0) + w
            total_weight += w

        if total_weight <= 0:
            self._streak_cand = None
            self._streak_since = None
            return self._confirmed

        winner = max(vote_totals, key=vote_totals.get)
        winner_share = vote_totals[winner] / total_weight
        winner_gest = winner if winner != "__none__" else None

        if winner_share < self._min_vote_share:
            self._streak_cand = None
            self._streak_since = None
            return self._confirmed

        if self._confirmed is not None and winner_gest != self._confirmed:
            need_sec = self._unlock_seconds
        else:
            need_sec = self._lock_seconds
        self._active_threshold_sec = need_sec

        if winner_gest != self._streak_cand or self._streak_since is None:
            self._streak_cand = winner_gest
            self._streak_since = now

        elapsed = now - (self._streak_since if self._streak_since is not None else now)
        if need_sec <= 0 or elapsed >= need_sec:
            self._confirmed = winner_gest

        return self._confirmed

    @property
    def confirmed(self) -> str | None:
        """Current locked-in gesture class name, or ``None`` for no-hand / not yet confirmed."""
        return self._confirmed

    def reset_after_climb(self) -> None:
        """Clear state so a fresh confirmation streak is required (e.g. after thumbs climb)."""
        self._history.clear()
        self._confirmed = None
        self._streak_cand = None
        self._streak_since = None
        self._active_threshold_sec = self._lock_seconds

    @property
    def streak_candidate(self) -> str | None:
        """Gesture class name currently being held for lock/unlock, or None."""
        return self._streak_cand

    @property
    def streak_ratio(self) -> float:
        """Progress 0..1 toward confirming the current streak under the active threshold."""
        if (
            self._streak_since is None
            or self._active_threshold_sec <= 0
        ):
            return 0.0
        elapsed = time.monotonic() - self._streak_since
        return min(1.0, elapsed / self._active_threshold_sec)

    @property
    def lock_target(self) -> float:
        """Active lock/unlock duration in seconds (for HUD / debug)."""
        return self._active_threshold_sec
