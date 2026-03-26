# AI Context -- MLxDrone_larp

This folder exists for AI assistants. Read these files to understand the project and continue work without needing prior chat history.

## Project in One Paragraph

A hand-gesture-controlled drone system. **YOLOv8n** (HaGRID `best.pt` or HF fallback) proposes the hand box; **bridge + sim + `tello_view` / physical autonomy** run **`TrustedHandGate`** (**MediaPipe** verifies a real hand in the **YOLO crop**; temporal trust gates **behavior** while YOLO still supplies the classifier crop). Upstream: **`hand_detection.py`** (geometry, optional **YuNet**, `BboxSmoother`, square crop) — used by **bridge, sim, `tello_view`, and `tello_real_autonomy_v1`**. **EfficientNet-B0** classifies; **GestureFilter** gates commands. This stack strongly suppresses YOLO false positives **without** room-specific YOLO hard-negative retraining.

**Execution paths:**

- **2D OpenCV simulator** (`simulate_drone.py`, no ROS) — optional **MANUAL / SEARCH / FACE_LOCK** on **`drone.heading`** (yaw-only sim semantics).
- **Gazebo + ROS2** (Windows `gesture_bridge.py` → TCP → WSL2 `gesture_ros2_node.py` → `/drone1/cmd_vel` + `tello_action`) — **SEARCH / FACE_LOCK** via **`angular.z`** when fist-edge + face fields drive autonomy on the **simulated** drone.
- **Tello camera preview** (`tello_view.py`) — **no** flight; same perception stack as sim/bridge (Shared **`hand_detection`**, TrustedHand, YuNet).
- **Physical Tello (djitellopy, no ROS):** **`tello_real_autonomy_v1.py`** — preview → takeoff → **SEARCH** (default **cw/ccw** SDK steps; optional **`--search-mode rc`**) → **FACE_LOCK** (**`send_rc_control`** yaw only) → land on confirmed **open_palm** / **Q** / Ctrl+C. **`tello_hover_baseline.py`** — connect, takeoff, **10 s hover, no RC**, land (stability sanity check). **`tello_real_flight_test.py`** — minimal takeoff/hover/land and optional onboard HUD.

Perception can use a **PC webcam** or the **Tello camera** (`--source tello` on `gesture_bridge.py` and `simulate_drone.py`; Gazebo via `launch_all.ps1 -Source tello` or `drone -Tello`). Pipeline spans **Windows** (inference, physical Tello) and **WSL2 Ubuntu 22.04** (ROS2/Gazebo).

## Gesture-to-Command Mapping

| Gesture                  | Command      | Drone Action                         |
| ------------------------ | ------------ | ------------------------------------ |
| two_fingers (peace sign) | FOLLOW_ARM | Hover / follow-stack hook; not forward cruise (see ARCHITECTURE) |
| fist                     | STOP         | Hover in place                       |
| open_palm                | LAND         | Descend and land                     |
| thumbs_up                | MOVE_UP      | Take off if landed, then ascend      |

*(Physical autonomy `tello_real_autonomy_v1.py` only uses **open_palm** for land among gestures; other mapped gestures are HUD-only there.)*

## Read Order

1. **PERSONALITY.md** -- How to communicate with this user. Read this first, every time.
2. **ARCHITECTURE.md** -- System design, data flows, every file and what it does, environment setup, build instructions.
3. **STATUS.md** -- What works now, what's broken, session-log diagnostics, dataset stats, prioritized next steps. **This file is the living document -- update it after completing any work.**

## Human-Readable Docs (for the user, not for you)

- `JOURNEY.md` -- Narrative walkthrough of the ML/perception phase (data collection, training, evaluation).
- `SIMULATION_GUIDE.md` -- Deep-dive theory on ROS2, Gazebo, and how the simulation was built.
- `SIMULATION_STATUS.md` -- User-facing summary of current simulation capabilities and model issues.

## Repo hygiene (indexing / git)

- **`.cursorignore`** -- Excludes datasets, weights, logs, `runs/`, etc. from Cursor codebase indexing.
- **`.gitignore`** -- Datasets, `hagrid_detection/`, most `*.pt` / `*.onnx` / `*.task`, logs, `runs/`. **Tracked exceptions:** `gesture_drone/models/gesture_model.pt`, `gesture_drone/models/yolo_hands/weights/best.pt`, `gesture_drone/models/face_detection_yunet_2023mar.onnx` so a fresh clone has core inference weights without manual file copies.

## Update Protocol

After completing any task:

1. Update `STATUS.md` with what you changed, what's now working/broken, and adjust the next steps.
2. Update `ARCHITECTURE.md` only if you added/removed/moved files or changed the system design.
3. Do NOT edit `PERSONALITY.md` unless the user explicitly asks for it.
4. Do NOT edit human docs (JOURNEY.md, SIMULATION_GUIDE.md, SIMULATION_STATUS.md) unless the user asks.

## Handoff quickstart

1. Read **PERSONALITY.md**, then **ARCHITECTURE.md**, then **STATUS.md** (this folder).
2. **Two runtimes:** gesture inference and physical **djitellopy** scripts run on **Windows** (project venv + `requirements.txt`). **ROS 2 + Gazebo** run in **WSL2 Ubuntu 22.04** — not the same Python as Windows (details in ARCHITECTURE **Environment setup**). Core gesture / YOLO-hand / YuNet weights are **in git** under `gesture_drone/models/`; you still need **`hand_landmarker.task`** locally (see root **README.md**).
3. **No motors:** verify perception with **`simulate_drone.py`** (2D) or **`tello_view.py`** (Tello HUD only) before caring about Gazebo or physical flight.
4. **Simulated drone:** **`drone`** / **`launch_all.ps1`** — Windows bridge → TCP **:9090** → WSL **`gesture_ros2_node.py`** + Gazebo (ARCHITECTURE **Running**).
5. **Physical Tello:** run **`tello_hover_baseline.py`** (hover, no RC) before **`tello_real_autonomy_v1.py`**; read ARCHITECTURE **Physical Tello — behavior and constraints** and **Troubleshooting (minimal)** first.
6. **TrustedHand design:** `gesture_drone/docs/PERCEPTION_GATING_DESIGN.md` — parameters and rationale beyond the summary in ARCHITECTURE.
