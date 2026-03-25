"""
Unified v1 trusted-hand policy: YOLO owns the crop; MediaPipe verifies a real hand
in that crop; temporal rules gate behavior (not debug classification).

Configurable: K_CREATE, MP_MISS_DROP, NO_BOX_DROP (see TrustedHandConfig).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore

try:
    import mediapipe as mp
except ImportError:
    mp = None  # type: ignore


@dataclass
class TrustedHandConfig:
    k_create: int = 4
    mp_miss_drop: int = 5
    no_box_drop: int = 5
    mp_min_landmarks: int = 15
    # Upscale YOLO crop before MP if max side is below this (hand tiny in box / far away).
    mp_infer_min_side: int = 192
    # HandLandmarker detection thresholds (verification path; can be lower than dataset tools).
    mp_min_hand_detection_confidence: float = 0.15
    mp_min_hand_presence_confidence: float = 0.15


@dataclass
class TrustedHandState:
    trusted: bool = False
    create_streak: int = 0
    mp_miss_streak: int = 0
    no_box_streak: int = 0


def describe_gate_load_failure(model_dir: Path) -> str:
    """Human-readable reason when ``load_hand_landmarker`` would return None."""
    if mp is None:
        return "mediapipe is not installed (pip install mediapipe)"
    path = model_dir / "hand_landmarker.task"
    if not path.is_file():
        return f"hand_landmarker.task not found at {path}"
    return "HandLandmarker could not be loaded (corrupt model?)"


def load_hand_landmarker(
    model_dir: Path, cfg: TrustedHandConfig | None = None,
):
    """Return HandLandmarker or None if mediapipe / hand_landmarker.task missing."""
    if mp is None:
        return None
    path = model_dir / "hand_landmarker.task"
    if not path.is_file():
        return None
    c = cfg or TrustedHandConfig()
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(path)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=c.mp_min_hand_detection_confidence,
        min_hand_presence_confidence=c.mp_min_hand_presence_confidence,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(options)


def _prepare_crop_for_mp_infer(crop_rgb: np.ndarray, min_side: int) -> np.ndarray:
    """Resize so max(h,w) >= min_side — helps when the hand is small inside YOLO's square."""
    if cv2 is None or crop_rgb is None or crop_rgb.size == 0:
        return crop_rgb
    h, w = crop_rgb.shape[:2]
    m = max(h, w)
    if m >= min_side:
        return np.ascontiguousarray(crop_rgb)
    scale = min_side / float(m)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    up = cv2.resize(crop_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(up)


def _mp_evaluate_crop(
    landmarker, crop_rgb: np.ndarray, cfg: TrustedHandConfig
) -> tuple[bool, int, str]:
    """
    Returns (pass, landmark_count_on_first_hand, reason_tag).
    reason_tag helps HUD when pass is False.
    """
    if landmarker is None or mp is None or crop_rgb is None or crop_rgb.size == 0:
        return False, 0, "no_input"
    if crop_rgb.shape[0] < 20 or crop_rgb.shape[1] < 20:
        return False, 0, "tiny_crop"
    prepared = _prepare_crop_for_mp_infer(crop_rgb, cfg.mp_infer_min_side)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=prepared)
    result = landmarker.detect(mp_image)
    if not result.hand_landmarks:
        return False, 0, "mp_no_hand"
    n = len(result.hand_landmarks[0])
    if n < cfg.mp_min_landmarks:
        return False, n, f"lm_{n}"
    return True, n, "ok"


def _mp_passes_crop(landmarker, crop_rgb: np.ndarray, cfg: TrustedHandConfig) -> bool:
    ok, _, _ = _mp_evaluate_crop(landmarker, crop_rgb, cfg)
    return ok


def format_trust_hud_line(diag: dict[str, Any]) -> str:
    """Single-line debug string for HUD when trust is enabled."""
    if not diag.get("trust_enabled"):
        return ""
    t = diag.get("trust_trusted", False)
    ph = diag.get("trust_phase", "")
    cr = diag.get("trust_create_streak", 0)
    ms = diag.get("trust_mp_miss_streak", 0)
    nb = diag.get("trust_no_box_streak", 0)
    mp_ok = diag.get("mp_pass", False)
    why = diag.get("mp_why", "")
    nlm = diag.get("mp_lm")
    extra = f" {why}" if why else ""
    lm_s = "" if nlm is None else f" lm:{int(nlm)}"
    return f"TRUST:{t} | {ph} | mp_ok:{mp_ok}{lm_s}{extra} | cr:{cr} mp_miss:{ms} nb:{nb}"


class TrustedHandGate:
    """
    Updates temporal trust from YOLO crop + MediaPipe verification each frame.
    Call ``update(hand_crop, base_diag)`` after ``detect_hand``; read ``behavior_allow``
    from the returned dict. Pre-trust: ``behavior_allow`` is False (classifier may still
    run upstream for debug only).
    """

    def __init__(self, landmarker, config: TrustedHandConfig | None = None):
        self.landmarker = landmarker
        self.config = config or TrustedHandConfig()
        self.state = TrustedHandState()

    def reset(self) -> None:
        self.state = TrustedHandState()

    def update(
        self, hand_crop: np.ndarray | None, base_diag: dict[str, Any] | None
    ) -> dict[str, Any]:
        diag: dict[str, Any] = dict(base_diag or {})

        if self.landmarker is None:
            diag["trust_enabled"] = False
            diag["behavior_allow"] = hand_crop is not None
            diag["trust_phase"] = "gate_off"
            return diag

        diag["trust_enabled"] = True
        yolo_valid = (
            hand_crop is not None
            and hand_crop.size > 0
            and hand_crop.shape[0] >= 20
            and hand_crop.shape[1] >= 20
        )

        mp_pass = False
        mp_lm = 0
        mp_why = ""
        if yolo_valid:
            mp_pass, mp_lm, mp_why = _mp_evaluate_crop(
                self.landmarker, hand_crop, self.config
            )
        else:
            mp_why = "no_yolo_crop"

        if not yolo_valid:
            self.state.no_box_streak += 1
            self.state.create_streak = 0
            self.state.mp_miss_streak = 0
            if self.state.trusted and self.state.no_box_streak >= self.config.no_box_drop:
                self.state.trusted = False
            phase = "no_hand"
            behavior_allow = False
        elif mp_pass:
            self.state.no_box_streak = 0
            self.state.create_streak += 1
            self.state.mp_miss_streak = 0
            if not self.state.trusted and self.state.create_streak >= self.config.k_create:
                self.state.trusted = True
            if self.state.trusted:
                phase = "trusted"
                behavior_allow = True
            else:
                phase = "pre_trust"
                behavior_allow = False
        else:
            # YOLO valid, MP fail
            self.state.no_box_streak = 0
            self.state.create_streak = 0
            if self.state.trusted:
                self.state.mp_miss_streak += 1
                if self.state.mp_miss_streak >= self.config.mp_miss_drop:
                    self.state.trusted = False
            if self.state.trusted:
                phase = "coast"
                behavior_allow = True
            else:
                phase = "pre_trust"
                behavior_allow = False

        diag["trust_trusted"] = self.state.trusted
        diag["trust_phase"] = phase
        diag["trust_create_streak"] = self.state.create_streak
        diag["trust_mp_miss_streak"] = self.state.mp_miss_streak
        diag["trust_no_box_streak"] = self.state.no_box_streak
        diag["mp_pass"] = mp_pass
        diag["mp_lm"] = mp_lm if yolo_valid else None
        diag["mp_why"] = mp_why
        diag["behavior_allow"] = behavior_allow
        return diag


def perception_gate_wanted(cli_no_perception_gate: bool) -> bool:
    """
    Trusted-hand gate is ON by default. Disable with ``--no-perception-gate`` or
    environment ``MLX_GESTURE_PERCEPTION_GATE=0`` (also: false, no, off).
    """
    import os

    if cli_no_perception_gate:
        return False
    v = os.environ.get("MLX_GESTURE_PERCEPTION_GATE", "").strip().lower()
    if v in ("0", "false", "no", "off"):
        return False
    return True
