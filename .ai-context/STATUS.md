# Project Status

Last updated: 2026-03-27 (tello_view stream enhance + `tello_real_autonomy_v1` policy; `.ai-context` synced)

## Self-contained clone

A **fresh clone** plus **Windows venv** + `pip install -r requirements.txt` gives **`gesture_model.pt`**, **HaGRID `yolo_hands/weights/best.pt`**, **YuNet ONNX**, and **`hand_landmarker.task`** (MediaPipe TrustedHand) in **`gesture_drone/models/`** — no manual model file copies for the default stack. **`.env`:** recreate locally if you rely on Hugging Face (e.g. **`HF_TOKEN`** for gated downloads / Ultralytics **`hand_yolov8n.pt`** fallback). ROS/Gazebo side on WSL unchanged (separate workspace setup).

## What Works

- **End-to-end Gazebo pipeline**: Webcam **or Tello camera** → YOLO → EfficientNet → GestureFilter → TCP → `gesture_ros2_node.py` → Gazebo Tello (`pretty.world`).
- **Takeoff / land / hover / up**: Mapped from gestures; **peace sign (`two_fingers`)** maps to **`FOLLOW_ARM`** (hover / zero forward — **not** forward flight). `gesture_ros2_node` mirrors plugin state machine (landed → taking_off → flying → landing).
- **cmd_vel (conservative)**: Default forward/up use **`VELOCITY_FORWARD = 0.12`**, **`VELOCITY_UP = 0.10`** m/s. When **`beh_state`** is **`SEARCH`** or **`FACE_LOCK`** (fist-edge + face TCP fields), the node publishes **`angular.z`** autonomy yaw and **zero linear** translation (see **`gesture_ros2_node.py`**).
- **TCP watchdog**: No line from Windows for **>3 s** → treat as **STOP** (zero twist when flying).
- **2D simulator** (`simulate_drone.py`): Same perception as bridge; **HVEL/VVEL** HUD; **`--source tello`**; SessionLogger; optional **MANUAL / SEARCH / FACE_LOCK** + **`H_HOLD` “settled”** line on HUD; fist toggles autonomy in sim.
- **Tello preview** (`tello_view.py`): **`hand_detection.detect_hand`** + TrustedHand + YuNet + GestureFilter + HUD; **no flight commands**. After **`connect()`**, requests **720p / 30 fps / 5 Mbps** via djitellopy (try/except). **TrustedHand** uses **`trusted_hand_config_tello_camera()`** — lower MP detection/presence thresholds, **288px** min crop upscale, **`min_landmarks=14`**, temporal **K_CREATE=3** / **MP_MISS_DROP=7** (CLI overrides). Optional **`-E` / `--enhance-stream`**: **bilateral** (**d=7**, σColor/Space **50**) → **unsharp** (same defaults as before: amount **1.35**, Gaussian blur σ **1.0**, blend **0.35**) on each frame; cyan **ENHANCED** badge. No exposure lift, no CLAHE, no **`--enhance-cuda`**. Preview FPS is usually bound by **Wi‑Fi / H.264 decode**, not this step.
- **Physical Tello (djitellopy)**: **`tello_real_autonomy_v1.py`** (preview **[T]** → Enter → takeoff → SEARCH default **cw/ccw** steps → FACE_LOCK **rc** yaw → **open_palm** / **Q** / Ctrl+C land); **`tello_hover_baseline.py`** (10 s hover **no rc**); **`tello_real_flight_test.py`** (minimal flight + optional **`--onboard`**). **Autonomy v1 only:** after **`init_perception`**, **`perception["tgate"] = None`** — **TrustedHandGate is off** in preview and flight; **`behavior_allow`** follows YOLO crop only. **GestureFilter** uses script-local **`AUTONOMY_GESTURE_LOCK_FRAMES = 19`** / **`AUTONOMY_GESTURE_UNLOCK_FRAMES = 25`** (bridge/sim/`tello_view` remain **8** / **12**). TrustedHand unchanged everywhere else.
- **Launchers**: `launch_all.ps1` with **`-Source webcam|tello`**; repo **`Load-PowerShellAliases.ps1`** — **`drone`**, **`tello-autonomy`**, **`tello-realtest`**, etc.; add **`tello-hover`** locally if desired (see ARCHITECTURE).
- **YOLO hands** (HaGRID `models/yolo_hands/weights/best.pt` if present, else HF **`hand_yolov8n.pt`**), **`hand_detection.detect_hand`** + **`TrustedHandGate`** (**MediaPipe** verifies YOLO crop; **on by default**; `--no-perception-gate` or `MLX_GESTURE_PERCEPTION_GATE=0` off) + **BboxSmoother** + **GestureFilter** on **`gesture_bridge.py`** and **`simulate_drone.py`**. Pre-trust: classifier may run for debug/session log **only**; `behavior_allow` gates `GestureFilter`. **Field-tested:** YOLO FPs rarely break behavior; MP structural check is conservative on non-hands; reduces need for room-specific YOLO hard negatives.
- **`tello_view.py`:** Uses **`hand_detection.detect_hand`** + **TrustedHandGate** (Tello-tuned config) + YuNet; bridge/sim still use generic **`TrustedHandConfig`** unless changed later.
- **YuNet** (`yunet_face.py`): On **bridge** and **sim**, used for **face-vs-hand overlap rejection** when loaded + **proximity HUD** while follow / `two_fingers` is active (see ARCHITECTURE). **No extra follow `cmd_vel`** — preview only. ONNX **`face_detection_yunet_2023mar.onnx`** is **tracked in repo** under `gesture_drone/models/`; **HF download** only if the file is missing/corrupt.
- **Training/data pipeline**, **GPU Gazebo** (WSLg d3d12), **URDF/visual** improvements — unchanged from prior status.

## Gesture Confirmation System

Shared constants (tune in `simulate_drone.py` / `gesture_bridge.py`):

| Parameter | Value | Effect |
|-----------|-------|--------|
| `window` | 10 | Sliding history |
| `GESTURE_LOCK_FRAMES` | **8** | Frames to lock a gesture |
| `GESTURE_UNLOCK_FRAMES` | **12** | Frames to switch away (hysteresis) |
| *`tello_real_autonomy_v1.py`* | **19** / **25** | Same filter class: **`AUTONOMY_GESTURE_LOCK_FRAMES`** / **`AUTONOMY_GESTURE_UNLOCK_FRAMES`** (TrustedHand off; stricter temporal gating for physical flight) |
| `min_vote_share` | 0.60 | Dead-band |
| `CONFIDENCE_THRESHOLD` | 0.85 | Below → no vote |
| `COMMAND_COOLDOWN` | **1.2 s** | Min time between accepted command changes |

Vote = `confidence × recency` (oldest 0.5× → newest 1.0×).

Older validated session stats (119s sim, pre–lock-frame bump) remain a useful reference: ~98% command suppression vs raw classifications; filter disagreed ~44% of frames.

## Hand detection & session diagnostics

- **Training:** `prepare_yolo_hands.py` → `train_yolo_hands.py` (Ultralytics `yolov8n.pt` → HaGRID). **Runtime weights:** see ARCHITECTURE table (`best.pt` vs Bingsu fallback).
- **`simulate_drone.py` `SessionLogger`:** CSV under `gesture_drone/logs/session_*.csv` with per-frame **`yolo_n`**, **`yolo_top_conf`**, **`yolo_pick_conf`**, **`face_iou`**, **`reject_stage`**, classifier margin + softmax JSON. **Not in repo** (gitignored) — generate locally by running the sim.
- **`analyze_session_log.py`** — quick **`reject_stage`** histograms and filter stats from a session file.
- **`reject_stage` values** (from `hand_detection.py`): `no_yolo`, `conf`, `aspect`, `area_small`, `area_large`, `smooth_none`, `crop_small`, `ok`. They describe **which gate fired**, not semantic object type (no “chair rail” / “torso” labels in logs).
- **Separating model vs policy issues:** Use a **controlled no-hand** recording and compute rates from CSV (e.g. fraction of frames with `yolo_n ≥ 1` or `reject_stage == ok`). High `ok` in true no-hand → detector/domain; mostly filtered but wrong commands → classifier + **GestureFilter** / temporal gating. **BBox xyxy is not logged** — add columns if analyzing N-frame spatial persistence.
- **Trusted-hand HUD** (`simulate_drone` / `gesture_bridge` when gate on): `trust_*`, `mp_pass`, `mp_lm`, `mp_why`, `behavior_allow`; MP uses upscaled crop for verify if small (`perception_gating.py`).

## Known Issues

- **BUG-2 (rc=3 BUSY)**: Still possible on rapid service calls during transitions; reduced with better gestures + slower cmd_vel.
- **BUG-3**: Gazebo cold start 15–30 s (shader compile).
- **Edge**: Single index finger → `two_fingers` at high confidence (closed-set); do not raise threshold above 0.85 for normal use.

## Troubleshooting (quick reference)

See ARCHITECTURE **Troubleshooting (minimal)** for a symptom → cause table (BUSY, Gazebo cold start, TCP watchdog, Tello H.264 startup, SEARCH drift **`cw` vs `rc`**).

- **Physical:** Prefer **`tello_hover_baseline.py`** before **`tello_real_autonomy_v1.py`**. If the preview never shows a stable image, do not proceed to takeoff — fix stream/Wi‑Fi first.
- **Two Pythons:** If imports or weights fail after “it works on the other side,” confirm whether the command ran **Windows venv** vs **WSL ROS** (ARCHITECTURE **Environment setup**).

## What Was Already Tried (Do NOT Repeat)

(See previous STATUS: WSLg `wsl --shutdown`, 5-class unknown model reverted, MediaPipe **detection** replaced by YOLO, `torch.compile` disabled on WSL, threshold-only fixes for wrong class.)

## Dataset / training corpora (local disk)

**2026-03-27:** The large gitignored trees **`gesture_drone/hagrid_detection/`**, **`gesture_drone/external/`**, **`gesture_drone/dataset_cropped/`**, and **`gesture_drone/dataset/`** were **removed** from this workspace to reclaim disk. **Inference is unchanged** — weights live under **`gesture_drone/models/`** (tracked in git).

**Historical training split** (for reference only, no longer on disk here): **114,060 / 28,707** train/val cropped images (4 classes + HaGRID merge). Full breakdown and narrative: **`JOURNEY.md`**.

**Retraining YOLO or gesture classifiers** requires re-downloading HaGRID (or your corpus), re-running **`prepare_yolo_hands.py`** / **`train_yolo_hands.py`** and the gesture **`collect_data.py` → `train_model.py`** style pipeline so those directories are repopulated.

## Key File Locations

| File | Purpose |
|------|---------|
| `gesture_drone/scripts/gesture_bridge.py` | Windows → TCP; `--source webcam\|tello` |
| `gesture_drone/scripts/gesture_ros2_node.py` | WSL2 TCP → ROS2 |
| `gesture_drone/scripts/launch_all.ps1` | `-CameraIndex`, `-Source` |
| `gesture_drone/scripts/simulate_drone.py` | 2D sim; `--source`, `--world-width-m` |
| `gesture_drone/scripts/tello_view.py` | Tello HUD only (shared `hand_detection`) |
| `gesture_drone/scripts/tello_real_autonomy_v1.py` | Physical autonomy v1 (djitellopy; no ROS) |
| `gesture_drone/scripts/tello_hover_baseline.py` | Physical hover 10 s, no rc |
| `gesture_drone/scripts/tello_real_flight_test.py` | Physical minimal flight / `--onboard` |
| `gesture_drone/scripts/search_behavior.py` | Face OK + streak constants (sim / ROS / bridge / physical) |
| `gesture_drone/scripts/launch_gazebo_bridge.sh` | WSL all-in-one; `--no-windows` with PS1 |

## Prioritized Next Steps

### P0 — GestureFilter everywhere
- **Done** on `gesture_bridge.py`, `simulate_drone.py`, `tello_view.py`.
- **Optional:** SessionLogger on bridge / Tello view for parity with sim.

### P1 — Control / safety
- **Done (initial):** Lower cmd_vel; longer cooldown; lock/unlock frames.
- **Done (sim / ROS path):** **`angular.z`** for Gazebo **SEARCH / FACE_LOCK** in **`gesture_ros2_node.py`**.
- **Todo:** Optional MOVE_DOWN; cap max velocities via config; physical FACE_LOCK drift / tuning.
- **Gazebo HUD** for active command (text overlay).

### P2 — Perception / behavior
- **Robustness (YOLO FP):** **Addressed** on **bridge, sim, `tello_view`** via **`TrustedHandGate`** (MP + time). Further tuning: `TrustedHandConfig` / CLI thresholds.
- **Other:** Controlled empty-scene session + `analyze_session_log.py` still useful for metrics; hard-negative YOLO retrain **de-prioritized** for room generalization.
- **Future “final phase” — STOP → slow 360° + face search:** State machine + **`angular.z`**. **Partial:** YuNet + proximity HUD when **`two_fingers`** confirmed; closed-loop follow not yet.
- **two_fingers → follow at distance:** Visual servoing (bbox center / size); heuristic depth + safety limits; separate from gesture CNN retrain. **Current:** `two_fingers` → **`FOLLOW_ARM`** (no forward); HUD preview only.
- **Identity (“only me”):** Optional later (embeddings / small classifier).

### P3 — Docs / ops
- Keep human docs (`SIMULATION_STATUS.md`, etc.) in sync when user asks.
- `.gitignore` — datasets and most weights excluded; **default inference** bundle (gesture + `best.pt` + YuNet ONNX + **`hand_landmarker.task`**) **tracked** (see ARCHITECTURE **Models & data**).

## Recent Changes (chronological summary)

### 2026-03-27 — `tello_view.py` stream enhance simplified; `tello_real_autonomy_v1.py` perception policy
- **`--enhance-stream` (`-E`):** Pipeline reverted to **bilateral + unsharp** only (defaults unchanged). Removed **conditional exposure**, **CLAHE (LAB L)**, **`--enhance-cuda`**, and related **`--enhance-*`** CLI flags.
- **`tello_real_autonomy_v1.py`:** **`perception["tgate"] = None`** immediately after **`init_perception`** so **TrustedHandGate never runs** in preview or flight; **`AUTONOMY_GESTURE_LOCK_FRAMES = 19`**, **`AUTONOMY_GESTURE_UNLOCK_FRAMES = 25`**. Q / Ctrl+C remain quick abort/land paths; open palm still uses standard GestureFilter hysteresis (including **25**-frame switch-away when already locked on another gesture).

### 2026-03-27 — Local training corpora removed
- **Deleted from disk (gitignored paths):** `gesture_drone/hagrid_detection/`, `gesture_drone/external/`, `gesture_drone/dataset_cropped/`, `gesture_drone/dataset/` (~190+ GB). **`venv/`** preserved.
- **Runtime:** Still **`gesture_drone/models/`** only for default stack; no behavior change for bridge / sim / Tello scripts.
- **Docs:** **STATUS** (this section + **Dataset / training corpora**); **ARCHITECTURE** (**Training YOLO hands**, **Models & data**); **`.ai-context/README.md`** (repo hygiene).

### 2026-03-26 — `tello_view.py`: experimental `-E` pipeline (superseded)
- **Note:** For a short period, **`--enhance-stream`** also ran **exposure compensation**, **CLAHE on LAB L**, and optional **`--enhance-cuda`**. That path was **removed 2026-03-27**; current behavior is **bilateral + unsharp** only (see entry above).

### 2026-03-26 — Tello preview: PowerShell arg forwarding + debug handoff
- **`tello`** function in **`Load-PowerShellAliases.ps1`** uses **`ValueFromRemainingArguments`** so `tello --enhance-stream` / **`tello -E`** reliably reaches Python (plain `@args` could drop flags in some profiles).
- **`tello_view.py`:** always prints **`[flags] enhance_stream=...`** after banner; **`-E`** alias for **`--enhance-stream`**.
- **`gesture_drone/docs/TELLO_VIEW_DEBUG_HANDOFF.md`** — share with others: UTF-8 UDP noise, command skew, enhance flag checks, minimal status bundle.

### 2026-03-26 — `tello_view.py`: Tello stream + TrustedHand tuning (preview only)
- **Video:** `set_video_resolution(720P)`, `set_video_fps(30)`, `set_video_bitrate(5Mbps)` after connect, before **`streamon()`** (errors logged, non-fatal). Implemented in **`tello_view.main()`** only (bridge/sim unchanged).
- **Perception:** **`trusted_hand_config_tello_camera()`** in **`perception_gating.py`** — softer HandLandmarker confidences, higher **`mp_infer_min_side`**, slightly lower **`mp_min_landmarks`**, relaxed temporal defaults; **`tello_view.init_perception`** uses **`dataclasses.replace`** with **`--k-create` / `--mp-miss-drop` / `--no-box-drop`** overrides. Scripts that call **`init_perception`** (**`tello_real_flight_test`**, **`tello_real_autonomy_v1`**) load the same preset; **autonomy v1** then clears **`tgate`** so TrustedHand does not run there.
- **Optional:** **`--enhance-stream`** — bilateral + unsharp on **`tello_view`** only (see **2026-03-27** entry).
- **Docs:** **STATUS** (this file); **ARCHITECTURE**; root **README** Tello preview blurb.

### 2026-03-26 — MediaPipe `hand_landmarker.task` tracked in git
- **Tracked:** `gesture_drone/models/hand_landmarker.task` (`.gitignore` exception for this filename only; other `*.task` remains ignored).
- **Docs:** Root **README.md** models table; **`.ai-context/README.md`**, **ARCHITECTURE**, **STATUS** (**Self-contained clone**, checklist, repo hygiene).

### 2026-03-26 — Core model weights committed to git
- **Tracked:** `gesture_drone/models/gesture_model.pt`, `gesture_drone/models/yolo_hands/weights/best.pt`, `gesture_drone/models/face_detection_yunet_2023mar.onnx` (surgical `.gitignore` exceptions; other `*.pt` / training trees still ignored).
- **Docs:** Root **README.md** models section; **`.ai-context/README.md`** handoff + repo hygiene; **ARCHITECTURE** (`.gitignore` row, **Models & data**, env checklist); **STATUS** (**Self-contained clone**).

### 2026-03-26 — `.ai-context` sync (physical + handoff)
- **README / ARCHITECTURE / STATUS:** Document **physical** scripts, corrected **`tello_view`** (shared **`hand_detection`** + TrustedHand), **ROS `angular.z`** when **SEARCH / FACE_LOCK**, **simulate_drone** autonomy/HUD, **`search_behavior.py`**, PowerShell aliases vs optional **`tello-hover`** wrapper.
- **README:** **Handoff quickstart** (read order, Windows vs WSL, sanity paths, physical order, **`PERCEPTION_GATING_DESIGN.md`** pointer).
- **ARCHITECTURE:** **Environment setup** expanded (Windows vs WSL, checklist); **ROS workspace path note**; **Key Classes** `detect_hand` consumers corrected; **Physical Tello — behavior and constraints**; **Troubleshooting (minimal)** table.
- **STATUS:** **Troubleshooting (quick reference)**.

### 2026-03-25 — `tello_hover_baseline.py`
- **Physical** djitellopy: connect → battery → Enter → **takeoff** → **10 s no rc** → **land**; Ctrl+C emergency land. Standalone stability test.

### 2026-03-25 — `tello_real_autonomy_v1.py` (physical Tello, no ROS)
- **djitellopy:** `takeoff` → settle (zero RC) → **SEARCH** (default **`cw`/`ccw`** steps; **`--search-mode rc`** for stick yaw) → **FACE_LOCK** (**`send_rc_control`** yaw) → **land** on confirmed **open_palm** or **Q** / Ctrl+C.
- Reuses **`tello_view.init_perception`**; repo **`Load-PowerShellAliases.ps1`** includes **`tello-autonomy`** (user **`$PROFILE`** may mirror with **`@args`**).
- **Pre-flight preview:** **[T]** in window → **Enter** in console for takeoff; **[Q]** abort preview.
- Defaults: **`--search-cw-degrees 15`**, **`--search-cw-interval 0.5`**, **`--search-yaw-rc 22`** (sign → cw vs ccw in cw-mode; magnitude in rc-mode), **`--lock-yaw-max-rc 28`**.

### 2026-03-25 — `simulate_drone.py`: H_HOLD + HUD (FACE_LOCK)
- **`H_HOLD`:** In `FACE_LOCK`, `hold_streak` increments only while `face_ok` and `abs(face_x_norm) <= EPS_X`; otherwise reset each frame. **`settled`** = `hold_streak >= H_HOLD` (debug semantics only; yaw still goes to zero immediately in deadband).
- **HUD `beh_line`:** Shows `hdg`, `yaw_cmd`, `db` (in deadband), `hold`/`H_HOLD`, `settled`, plus existing acq/loss/face_ok.

### 2026-03-25 — Trusted-hand gate (production path for bridge + sim)
- **`perception_gating.py`:** `TrustedHandGate` — K_CREATE / MP_MISS_DROP / NO_BOX_DROP; MP verify on YOLO crop (upscale for small crops); HUD `mp_why`; `load_hand_landmarker(MODEL_DIR, cfg)` confidence thresholds.
- **Default ON;** `--no-perception-gate` / `MLX_GESTURE_PERCEPTION_GATE=0` off. **`requirements.txt`:** `mediapipe`.
- **Validated in use:** YOLO may false-positive; MP rarely affirms non-hands; combined trust + classifier crop from YOLO.

### 2026-03-24 — `.ai-context` hand pipeline + diagnostics
- **README / ARCHITECTURE / STATUS:** Document YOLO weight paths and HaGRID training scripts; **`hand_detection.py`** vs **`tello_view`** inline detector; correct YOLO conf/imgsz for bridge+sim; YuNet on **bridge + sim**; SessionLogger columns and limits of FP categorization; pointer to `analyze_session_log.py`.

### 2026-03-24 — YuNet follow preview + gesture remap
- **`two_fingers` → `FOLLOW_ARM`** (not `MOVE_FORWARD`) in `gesture_bridge.py`, `gesture_ros2_node.py`, `simulate_drone.py`; labels in `tello_view.py`.
- **`yunet_face.py`:** YuNet ONNX (`face_detection_yunet_2023mar.onnx`) via Hugging Face; `FaceDetectorYN`, proximity + EMA.
- **`gesture_bridge.py`:** YuNet + face box + HUD line when confirmed `two_fingers`.
- **`.gitignore`:** optional track exception for the small YuNet ONNX filename.

### 2026-03-25 — Context documentation sync
- README / ARCHITECTURE / STATUS rewritten to reflect: YOLO (not MediaPipe) for **inference**; Tello + webcam sources; conservative ROS velocities; sim **HVEL/VVEL** scale; **`angular.z`** published for **SEARCH / FACE_LOCK** in **`gesture_ros2_node.py`**; PowerShell **`drone -Tello`**, **`simulate-tello`**, dual profile paths; `.cursorignore` / `.gitignore`; roadmap for face/follow without implying full retrain.

### 2026-03-24 — Engineering batch (see git / conversation)
- **Safety tuning:** `COMMAND_COOLDOWN` 1.2s; lock **8** / unlock **12**; ROS `VELOCITY_FORWARD` 0.12, `VELOCITY_UP` 0.10.
- **simulate_drone.py:** Fictional **m/s** for horizontal motion; `--world-width-m`, `--source tello`; `DroneState` speed fields; argparse.
- **gesture_bridge.py:** `--source tello`; Tello telemetry on HUD; cleanup on exit.
- **launch_all.ps1:** `-Source tello` → `gesture_bridge.py --source tello`.
- **PowerShell:** `drone` with `-Tello`; `simulate-tello` for 2D+Tello; profiles synced (OneDrive + Documents).
- **tello_view.py:** GestureFilter + streak HUD (already noted).
- **Indexing:** `.cursorignore`; `.gitignore` expanded (`hagrid_detection/`, `*.pt`, etc.).

### Earlier (2026-03-22 — 2026-03-23)
- HaGRID retrain 99.9% val; YOLO 320 + square crop + padding 0.08; `launch_all.ps1` + `drone`; Gazebo pretty world optimization; real Tello `tello_view.py`; GestureFilter rewrite + SessionLogger + sim progress bar — detail preserved in git history and sections above.

---

*For narrative ML arc, see `JOURNEY.md`. For Gazebo theory, see `SIMULATION_GUIDE.md`.*
