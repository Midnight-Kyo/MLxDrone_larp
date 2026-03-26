# Architecture

## System Overview

Two machines, one pipeline. Video input is either a **PC webcam** or the **Tello camera**. **Bridge, sim, and `tello_view`** share **`hand_detection.detect_hand`** (same YOLO weights table and TrustedHand path; see Inference Pipeline). **Physical** scripts reuse **`tello_view.init_perception`** where noted.

```
WINDOWS (GPU inference)                    WSL2 (Ubuntu 22.04, ROS2 + Gazebo)
┌──────────────────────────┐              ┌──────────────────────────────────┐
│  gesture_bridge.py       │   TCP/9090   │  gesture_ros2_node.py            │
│                          │──────────────│                                  │
│  Webcam OR Tello stream  │  JSON msgs   │  ROS2 Node: gesture_drone_bridge │
│    ↓                     │              │    ↓                             │
│  YOLOv8n hand detector   │              │  /drone1/cmd_vel (Twist)         │
│    ↓                     │              │  /drone1/tello_action (Service)  │
│  hand_detection + smooth  │              │    ↓                             │
│    ↓                     │              │  Gazebo + TelloPlugin            │
│  TrustedHandGate (MP)    │              │    ↓                             │
│    ↓                     │              │  3D Tello drone in pretty.world  │
│  EfficientNet-B0         │              │                                  │
│    ↓                     │              │                                  │
│  GestureFilter → Command │              │                                  │
│  (two_fingers → YuNet HUD)│             │                                  │
└──────────────────────────┘              └──────────────────────────────────┘
```

**2D simulator** (`simulate_drone.py`): same perception chain; commands drive a **virtual** top-down panel (no TCP). Optional `--source tello`. Optional **fist-toggle** **MANUAL / SEARCH / FACE_LOCK** on sim **heading** (see script; YuNet + `search_behavior`).

**Tello HUD only** (`tello_view.py`): **shared `hand_detection.detect_hand`** + **TrustedHandGate** + YuNet + GestureFilter (aligned with bridge/sim). **No** TCP, **no** motors — display and screenshots only.

**Physical Tello (djitellopy, no ROS in repo for flight control):**

- **`tello_real_autonomy_v1.py`** — pre-flight OpenCV preview → takeoff → settle (zero RC) → **SEARCH** (default **`rotate_clockwise` / `rotate_counter_clockwise`** steps; **`--search-mode rc`** for stick yaw) → **FACE_LOCK** (`send_rc_control` yaw only) → **land** (confirmed **open_palm** / **Q** / Ctrl+C). Reuses **`tello_view.init_perception`**.

- **`tello_hover_baseline.py`** — takeoff → **10 s hover, no `rc` commands** → land (stability baseline).

- **`tello_real_flight_test.py`** — scripted takeoff/hover/land; optional **`--onboard`** gesture HUD (still no gesture→motors by default in that path).

## TCP Protocol

- Transport: TCP socket, port **9090**
- Direction: Windows → WSL2 (one-way data, persistent connection)
- Format: newline-delimited JSON, one message per line
- WSL2 IP auto-detected via `wsl -d Ubuntu-22.04 -- hostname -I`

Message schema:
```json
{"command": "FOLLOW_ARM", "gesture": "two_fingers", "confidence": 0.923, "timestamp": 1711234567.89}
```

Valid commands: `FOLLOW_ARM`, `STOP`, `LAND`, `MOVE_UP`, `IDLE`

**`FOLLOW_ARM`:** Peace sign (`two_fingers`) arms the follow stack for preview only: **TCP sends this command** (not `MOVE_FORWARD`) so the ROS node publishes **zero forward velocity** (hover). **YuNet** (face box + proximity HUD) runs on **`gesture_bridge.py`** and **`simulate_drone.py`** while `two_fingers` is **confirmed** / follow latched — **HUD-only** on Windows (no follow `cmd_vel` on ROS yet).

## ROS2 Interface

All under namespace `/drone1/`:

| Interface | Type | Purpose |
|-----------|------|---------|
| `/drone1/cmd_vel` | Topic (geometry_msgs/Twist) | Velocity commands. **`linear.x`** = forward, **`linear.z`** = up. **`angular.z`** = yaw rate. Published at **10 Hz**. When TCP-driven **`beh_state`** is **`SEARCH`** or **`FACE_LOCK`**, the node publishes **zero `linear.*`** and sets **`angular.z`** from internal autonomy (`OMEGA_SEARCH` / face lock gains). Otherwise: **`MOVE_FORWARD`** / **`MOVE_UP`** / hover for **`FOLLOW_ARM`**, **`IDLE`**, **`STOP`** (see `gesture_ros2_node.py`). |
| `/drone1/tello_action` | Service (tello_msgs/TelloAction) | Discrete actions. `cmd="takeoff"` or `cmd="land"`. Returns `rc`: 1=OK, 3=BUSY. |

**Conservative cmd_vel defaults** (tunable in `gesture_ros2_node.py`): `VELOCITY_FORWARD = 0.12`, `VELOCITY_UP = 0.10` (m/s scale in plugin). Stale TCP (>3 s) forces command `STOP` (zero twist when flying).

## Tello Plugin State Machine

The Gazebo Tello plugin (`tello_plugin.cpp`) has 4 internal states:

```
landed ──takeoff──→ taking_off ──(alt>1.0m)──→ flying ──land──→ landing ──(alt<0.1m)──→ landed
```

Rules:
- `takeoff` only accepted when `landed`
- `land` only accepted when `flying`
- `cmd_vel` only processed when `flying`
- Any command during `taking_off` or `landing` returns `rc=3` (BUSY)
- `taking_off` lasts ~2s, `landing` lasts ~3s

`gesture_ros2_node.py` mirrors this state machine with timers to track transitions.

## Inference Pipeline (Windows Side)

### Hand detector weights (`load_hand_detector` in bridge / sim / `tello_view`)

| Priority | Path | Origin |
|----------|------|--------|
| 1 | `gesture_drone/models/yolo_hands/weights/best.pt` | Project: **fine-tune** Ultralytics `yolov8n.pt` on HaGRID-250k (see **Training YOLO hands** below). |
| 2 | `gesture_drone/models/hand_yolov8n.pt` | **Hugging Face** `Bingsu/adetailer` — downloaded on first use if missing. |

### `gesture_bridge.py` + `simulate_drone.py` — shared `hand_detection.detect_hand`

1. Frame from **OpenCV webcam** or **djitellopy** (`--source tello`).
2. BGR→RGB.
3. Optional **YuNet** largest face (`yunet_face.py`) on a strided schedule — passed as `face_xyxy` into `detect_hand` to **reject** hand boxes with high **IoU** vs face (`face_hand_iou_max` ≈ 0.22). If YuNet fails to load, this filter is **off** (logged once).
4. **YOLO** on a **downscaled** RGB copy if long edge > `MAX_INFER_SIDE_DEFAULT` (640); letterbox **`YOLO_IMGSZ=256`**; infer `conf=YOLO_INFER_CONF` (0.40); accept candidates ≥ **`YOLO_MIN_CONF` (0.58)** after **geometry** (aspect ratio, area vs frame) and face IoU.
5. Padding `PADDING_RATIO=0.08`, **BboxSmoother** (EMA `alpha=0.4`, `max_miss_frames=8`).
6. **Square crop** (centered on smoothed box) — keeps EfficientNet `Resize(256)`+`CenterCrop(224)` from clipping fingers.
7. **TrustedHandGate** (`perception_gating.py`): MediaPipe HandLandmarker **verifies** a real hand in the crop **before** classification; temporal trust keys **EfficientNet / GestureFilter** (default **on**; `--no-perception-gate` / `MLX_GESTURE_PERCEPTION_GATE=0` to disable).
8. Preprocess: `Resize(256)` → `CenterCrop(224)` → `ToTensor` → ImageNet normalize.
9. EfficientNet-B0 → softmax → class + confidence.
10. **GestureFilter**: weighted votes × recency, dead-band `min_vote_share=0.60`, lock/unlock hysteresis.
11. Raw confidence ≥ **`CONFIDENCE_THRESHOLD` (0.85)** required to vote.
12. **`COMMAND_COOLDOWN` (1.2 s)** between *accepted* command changes (`gesture_bridge.py`, `simulate_drone.py`).
13. TCP JSON (`gesture_bridge.py`) or sim physics (`simulate_drone.py`).

Per-frame diagnostics: `yolo_n`, `yolo_top_conf`, `yolo_pick_conf`, `face_iou`, `reject_stage` (see **Session diagnostics** in STATUS).

**Gesture lock (aligned across bridge + sim):** `GESTURE_LOCK_FRAMES=8`, `GESTURE_UNLOCK_FRAMES=12`.

### `tello_view.py` — **`hand_detection.detect_hand`** (same path as bridge/sim for detection)

- Uses project **`hand_detection`** pipeline + **TrustedHand** + YuNet + **`GestureFilter`** (see `tello_view.py` imports and loop).

### Follow preview + YuNet

- **`gesture_bridge.py`:** When **confirmed** gesture is `two_fingers`, YuNet draws largest face + **proximity** HUD (EMA). ONNX: `face_detection_yunet_2023mar.onnx` (HF if missing).
- **`simulate_drone.py`:** YuNet also used for **face overlap filtering** + same style follow preview when latched — not only bridge.

**Target flight flow (future, not fully automated yet):** takeoff → hover → search (face / yaw) → **follow vs hover** (discrete command) → **re-acquire** when the face is lost.

## Training YOLO hands (offline)

1. `prepare_yolo_hands.py` — HaGRID-250k → `gesture_drone/hagrid_detection/yolo_hands/` (`dataset.yaml`, images, labels).
2. `train_yolo_hands.py` — `YOLO("yolov8n.pt").train(...)` → writes `gesture_drone/models/yolo_hands/` (`weights/best.pt`).
3. `compare_detectors.py` — side-by-side Bingsu vs HaGRID weights (requires `best.pt` present).

Inference scripts **auto-pick** `best.pt` when the file exists.

## 2D Simulator Physics (`simulate_drone.py`)

- **Horizontal:** motion in **pixels/s** (`SIM_MOVE_SPEED_PX_S`). Display **HVEL** in **m/s** using a **fictional scale**: panel width `SIM_W` (500 px) = **`--world-width-m`** meters (default **10 m**), so `m/px = world_width_m / SIM_W`, and cruise `HVEL ≈ SIM_MOVE_SPEED_PX_S * m/px`.
- **Vertical (sim altitude):** **`SIM_ALT_SPEED_M_S`** (m/s); shown as **VVEL** on HUD.
- **Argparse:** positional **webcam index**, `--source {webcam,tello}`, `--world-width-m`.
- **SessionLogger** + streak bar on cam panel (same filter as bridge).

## Model Details

- Architecture: EfficientNet-B0 (torchvision), classifier head `Dropout(0.3) → Linear(1280, 4)`
- Weights: `gesture_drone/models/gesture_model.pt` (`model_state_dict` + `class_names`)
- Classes: `["fist", "open_palm", "thumbs_up", "two_fingers"]` (ImageFolder order)
- Training: 2-phase transfer learning, class weights, WeightedRandomSampler, augmentations (see STATUS / JOURNEY)

## Key Classes

| Location | Class | Purpose |
|----------|-------|---------|
| `simulate_drone.py` | `GestureFilter` | Temporal gating; also duplicated in `gesture_bridge.py` (same logic). |
| `simulate_drone.py` | `SessionLogger` | Per-frame CSV in `gesture_drone/logs/` (`yolo_*`, `reject_stage`, classifier margin). |
| `hand_detection.py` | `detect_hand` | Shared YOLO crop + filters (bridge, sim, `tello_view`, physical autonomy). |
| `simulate_drone.py` | `DroneState` | 2D position, altitude, `forward_speed_m_s` / `climb_speed_m_s` for HUD. |
| All inference scripts | `BboxSmoother` | EMA on YOLO boxes. |

## File Reference

### Project Root: `/home/kyo/Projects/MLxDrone_larp/`

| File | Purpose |
|------|---------|
| `requirements.txt` | Python deps (PyTorch, OpenCV, ultralytics, djitellopy, …) |
| `.gitignore` | venv, datasets, `hagrid_detection/`, logs, `**/runs/`, broad `*.pt` / `*.onnx` with **exceptions** for `gesture_drone/models/gesture_model.pt`, `yolo_hands/weights/best.pt`, `face_detection_yunet_2023mar.onnx` |
| `.cursorignore` | Same class of paths for Cursor indexing |
| `.env` | Secrets (e.g. HF_TOKEN). Never committed. |
| `.ai-context/` | AI context (this folder) |
| `JOURNEY.md`, `SIMULATION_*.md` | Human docs |

### Scripts: `gesture_drone/scripts/`

| File | Runs On | Purpose |
|------|---------|---------|
| `gesture_bridge.py` | Windows | `hand_detection.detect_hand` + EfficientNet + GestureFilter → TCP. `--source {webcam,tello}`, `--host`, `--port`. **YuNet** + proximity HUD when `two_fingers` confirmed (`yunet_face.py`). |
| `yunet_face.py` | Windows | YuNet ONNX path, `FaceDetectorYN`, largest face, proximity + EMA. |
| `gesture_ros2_node.py` | WSL2 | TCP → `cmd_vel` + `tello_action`. |
| `launch_gazebo_bridge.sh` | WSL2 | Gazebo + `gesture_ros2_node`. `--no-windows` when launched from `launch_all.ps1`. |
| `launch_all.ps1` | Windows | WSL window + wait on :9090 + `gesture_bridge.py`. **`-CameraIndex`** (default 2), **`-Source {webcam,tello}`**. |
| `simulate_drone.py` | Windows | 2D sim + `hand_detection` + SessionLogger + scale HUD + `--source` / `--world-width-m`; optional **SEARCH / FACE_LOCK** on heading. |
| `tello_view.py` | Windows | Tello stream + **`hand_detection.detect_hand`** + TrustedHand + gestures; **display only**. |
| `tello_real_autonomy_v1.py` | Windows | **Physical** djitellopy: preview → takeoff → SEARCH / FACE_LOCK → land; no ROS. |
| `tello_hover_baseline.py` | Windows | **Physical** djitellopy: takeoff → 10 s hover (**no rc**) → land. |
| `tello_real_flight_test.py` | Windows | **Physical** djitellopy: minimal takeoff/hover/land; optional `--onboard` HUD. |
| `search_behavior.py` | Windows / WSL | Shared **M_ACQUIRE**, **M_LOSS**, **face_ok_and_x_norm**, **`OMEGA_SEARCH`**, lock gains — sim, ROS node, bridge TCP extras, physical autonomy. |
| `hand_detection.py` | Windows | Shared detection + `reject_stage` diagnostics (bridge + sim + `tello_view` + physical autonomy). |
| `perception_gating.py` | Windows | **TrustedHandGate**: MP verifies YOLO crop; `behavior_allow` gates GestureFilter (**on by default**; `--no-perception-gate` or `MLX_GESTURE_PERCEPTION_GATE=0` to disable). Thresholds: `--k-create`, `--mp-miss-drop`, `--no-box-drop`. |
| `analyze_session_log.py` | — | Summarize `session_*.csv` (`reject_stage` counts, filter stats). |
| `gesture_drone/docs/PERCEPTION_GATING_DESIGN.md` | — | Full design + parameters for perception gate. |
| `tello_view.ps1` | Windows | Optional wrapper calling `tello_view.py` via UNC path. |
| `collect_data.py`, `crop_hands.py`, `train_model.py`, YOLO prep/train, etc. | — | Data and training (see table in STATUS). |

### Models & data

**In repo:** `gesture_drone/models/gesture_model.pt`, `gesture_drone/models/yolo_hands/weights/best.pt`, `gesture_drone/models/face_detection_yunet_2023mar.onnx` (tracked in git). **Local-only (typical):** `hand_landmarker.task`, optional `hand_yolov8n.pt` fallback, YOLO `last.pt`, backups — still gitignored. **Datasets** under `gesture_drone/` (large trees) remain gitignored; see STATUS for paths.

### ROS2 Workspace: `/root/ros2_ws/`

Unchanged: `pretty.world`, `pretty_launch.py`, `tello_plugin.cpp` (upstream), `tello.xml`, build via `colcon` (venv off).

**Path note:** **`/root/ros2_ws/`** here is a placeholder/conventional root; on a developer machine the workspace is often **`~/ros2_ws`** or **`/home/<user>/ros2_ws`**. Confirm with your **`colcon`** layout and **`launch_gazebo_bridge.sh`** — what matters is the built packages and world assets, not the literal path string.

## Environment Setup

(Windows venv, WSL ROS2 Humble, Gazebo Classic 11, GPU env vars — unchanged; see prior docs.)

### Windows vs WSL (who runs what)

- **Windows:** `gesture_bridge.py`, `simulate_drone.py`, `tello_view.py`, and all **djitellopy** scripts (physical Tello). Stack: **PyTorch**, Ultralytics YOLO, OpenCV, MediaPipe, etc., from **`requirements.txt`**. Use a **venv** and/or set **`MLX_PYTHON`** so PowerShell shims (`Load-PowerShellAliases.ps1`) call the correct **`python.exe`**.
- **WSL2 (Ubuntu 22.04):** **ROS 2 Humble**, **Gazebo Classic 11**, **`gesture_ros2_node.py`**, **`launch_gazebo_bridge.sh`**. Build with **`colcon`** in the ROS workspace; the ROS node's Python is the distro environment, **not** the Windows venv.
- **Cross-boundary:** Windows → WSL **TCP port 9090** (JSON lines). WSL IP from **`wsl -d Ubuntu-22.04 -- hostname -I`** (or your distro name).
- **GPU / WSLg:** Training and ONNX/YOLO often use **Windows** CUDA; Gazebo rendering may use **WSLg**. Do not assume one `python` or one venv controls both sides.

### Checklist (minimal)

- [ ] **Windows:** venv active (or `MLX_PYTHON` set), `pip install -r requirements.txt`; **`hand_landmarker.task`** present under `gesture_drone/models/` (download separately). Gesture / HaGRID-hand / YuNet ONNX weights ship in the repo.
- [ ] **WSL:** workspace sourced, `colcon build` succeeded, `ros2` + Gazebo launch finds **`pretty.world`** and the gesture bridge node.
- [ ] **Network:** Windows reaches WSL on **:9090**; for real Tello, laptop on the drone's Wi‑Fi as required by **djitellopy**.

## Running

### Gazebo + bridge (webcam)

```powershell
# Profile shorthand (webcam index 2 default):
drone
drone 1
```

Or: `powershell -File ...\launch_all.ps1 -CameraIndex 2 -Source webcam`

### Gazebo + bridge (Tello Wi‑Fi)

```powershell
drone -Tello
```

Or: `launch_all.ps1 -Source tello`

### 2D simulator only

```powershell
simulate                    # webcam index 2 in profile
python "...\simulate_drone.py" --source tello
```

### Tello preview only (no Gazebo)

```powershell
tello
```

### Physical Tello — hover baseline (no RC after takeoff)

```powershell
python "\\wsl$\Ubuntu-22.04\home\kyo\Projects\MLxDrone_larp\gesture_drone\scripts\tello_hover_baseline.py"
```

### Physical Tello — autonomy v1 (SEARCH / FACE_LOCK, no ROS)

```powershell
python "\\wsl$\Ubuntu-22.04\home\kyo\Projects\MLxDrone_larp\gesture_drone\scripts\tello_real_autonomy_v1.py"
```

### Manual bridge (no `launch_all`)

**WSL:** `bash launch_gazebo_bridge.sh`  
**Windows:** `python "\\wsl$\Ubuntu-22.04\home\kyo\Projects\MLxDrone_larp\gesture_drone\scripts\gesture_bridge.py" 2`  
or add `--source tello` when on drone Wi‑Fi.

## PowerShell profiles

Two copies often exist; **`$PROFILE` may point at OneDrive**:

- `C:\Users\<user>\OneDrive\Documents\WindowsPowerShell\Microsoft.PowerShell_profile.ps1`
- `C:\Users\<user>\Documents\WindowsPowerShell\Microsoft.PowerShell_profile.ps1`

**`Load-PowerShellAliases.ps1`** defines **`drone`**, **`bridge`**, **`simulate`**, **`tello`** (HUD), **`tello-realtest`**, **`tello-autonomy`**, and **`mlx-help`**. Users often duplicate **`$PROFILE`** entries or extend with wrappers (e.g. **`simulate-tello`** → `simulate_drone.py --source tello`, **`tello-hover`** → `tello_hover_baseline.py`) using the same **`$PSScriptRoot`** pattern as **`tello-autonomy`**.

## Retraining

```bash
cd /home/kyo/Projects/MLxDrone_larp && source venv/bin/activate
python gesture_drone/scripts/train_model.py
```

## Physical Tello — behavior and constraints

**Scope:** Physical flight uses **djitellopy** in-repo only; there is **no** ROS **`cmd_vel`** path for the real drone in this repo.

- **`tello_real_autonomy_v1.py`:** After connect/stream, **pre-takeoff OpenCV preview** (full HUD). **[T]** arms “ready to fly”; **Enter** in the **console** starts **takeoff**; **[Q]** in the preview window aborts **before** takeoff (**Ctrl+C** aborts preview the same way). Post-takeoff: short settle (**zero RC**), then **SEARCH** (default **discrete `rotate_clockwise` / `rotate_counter_clockwise`** steps to limit stick-coupling drift; **`--search-mode rc`** restores continuous yaw via **`send_rc_control`**), then **FACE_LOCK** (**yaw RC only**). **Land:** GestureFilter-confirmed **open_palm**, or **[Q]** in flight; **Ctrl+C** in flight triggers land, then **`finally`** (zero yaw RC, **`land()`**, **`streamoff`**, **`end()`**, destroy windows).
- **`tello_hover_baseline.py`:** **Takeoff → ~10 s hover with no RC → land.** Use as a **stability / battery / link** sanity check before autonomy.
- **`tello_real_flight_test.py`:** Minimal scripted flight; optional **`--onboard`** HUD — not the main gesture→motor product path.

**Constraints (non-exhaustive):** Fly only where law and safety allow; maintain **line of sight**, **clear volume**, and **healthy battery**; expect **Wi‑Fi latency and drops**. **Open-palm land** is software-gated — false positives are possible; **hover baseline** validates hardware before trusting autonomy. **FACE_LOCK** and **`rc`** SEARCH can still show **lateral drift**; tune flags in the script rather than assuming hover-perfect heading.

## Troubleshooting (minimal)

| Symptom | Likely cause | What to try |
|--------|----------------|------------|
| `tello_action` **rc=3 BUSY** | `takeoff` / `land` / `cmd_vel` during **`taking_off`** / **`landing`** | Wait ~2–3 s between discrete commands; avoid gesture spam (see STATUS **Known issues**). |
| Gazebo “hangs” on launch | Shader compile / cold start | Wait **15–30 s** first open; see STATUS **BUG-3**. |
| Tello stream: decode / PPS noise, brief **no frame** | H.264 startup | Normal; wait for steady frames. **Autonomy v1** pre-takeoff preview exists partly so you **see video before mounting**. |
| ROS hover / stop while flying | TCP gap **>3 s** | Restore Windows bridge or network; node forces **STOP** (watchdog). |
| Physical **SEARCH** drifts sideways | Stick yaw **`rc`** coupling on some firmware/hardware | Prefer default **`--search-mode cw`** (discrete rotations); use **`rc`** only when intentional. |
| Physical vs **Gazebo** behavior differ | Different stacks | Gazebo uses **ROS** + plugin; real drone uses **djitellopy** — tune and test **each** path separately. |
