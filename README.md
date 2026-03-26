# MLxDrone

Hand-gesture control for a simulated DJI Tello (ROS 2 + Gazebo) and optional **real** Tello flight via **djitellopy**. A single perception stack runs on **Windows** (YOLO hand box, MediaPipe verification, EfficientNet gesture classification, temporal **GestureFilter**), while **Ubuntu 22.04 + ROS 2 Humble** runs the 3D simulation.

---

## Features

- **Vision:** YOLOv8n hand proposals, **TrustedHandGate** (MediaPipe on the YOLO crop), optional **YuNet** face rejection / follow HUD, EfficientNet-B0 4-class gestures.
- **Simulation:** `gesture_bridge.py` (Windows) streams TCP JSON to `gesture_ros2_node.py` (WSL); Gazebo **cmd_vel** + discrete **takeoff/land** through the Tello plugin.
- **2D simulator:** `simulate_drone.py` — same filters as the bridge, virtual top-down drone, optional Tello camera source, session CSV logging.
- **Physical drone (optional):** `tello_view.py` (camera + HUD only), `tello_real_autonomy_v1.py` (yaw search + face lock + palm land), `tello_hover_baseline.py` (hover sanity check).

---

## Quick start (Windows)

### 1. Clone and Python environment

```powershell
cd <your-clone>
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

Optional **NVIDIA GPU:** install a matching **torch** / **torchvision** wheel from [pytorch.org](https://pytorch.org/get-started/locally/) *before* or *instead of* the CPU build pulled by `pip install -r requirements.txt`.

Set `MLX_PYTHON` to your `python.exe` if you use PowerShell helpers and the default `python` is wrong:

```powershell
$env:MLX_PYTHON = "C:\path\to\.venv\Scripts\python.exe"
```

### 2. Models (required for inference)

**Shipped in the repo** under **`gesture_drone/models/`** after clone:

| File | Role |
|------|------|
| **`gesture_model.pt`** | EfficientNet gesture head |
| **`yolo_hands/weights/best.pt`** | HaGRID-tuned hand detector |
| **`face_detection_yunet_2023mar.onnx`** | YuNet (face vs hand + follow HUD) |
| **`hand_landmarker.task`** | MediaPipe **Hand Landmarker** (TrustedHand gate) |

Optional: **`hand_yolov8n.pt`** — Hugging Face fallback if **`best.pt`** is absent and Ultralytics downloads (see `gesture_bridge.py` / `hand_detection`). **`yunet_face.py`** can still fetch YuNet if the ONNX is missing.

To refresh the landmarker from upstream, use [MediaPipe hand_landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) and replace `gesture_drone/models/hand_landmarker.task`.

Other bulky artifacts (datasets, extra `*.task` files, `last.pt`, backup checkpoints, training runs) stay **gitignored**.

### 3. PowerShell helpers (optional)

```powershell
cd gesture_drone\scripts
.\Install-MLxPowerShellProfile.ps1   # once
. .\Load-PowerShellAliases.ps1       # this session
mlx-help
```

### 4. Run the 2D simulator (no ROS)

From **`gesture_drone/scripts`** (so imports resolve), with venv active:

```powershell
python simulate_drone.py 0
python simulate_drone.py --source tello
```

---

## Simulation (Windows + WSL 2)

**Prerequisites on WSL (Ubuntu 22.04):** ROS 2 **Humble**, Gazebo Classic, your **ros2_ws** built with the Tello simulation packages (see **`SIMULATION_GUIDE.md`** and **`.ai-context/ARCHITECTURE.md`**). The bridge node uses **`rclpy`** and **`tello_msgs`** — install those via your ROS workspace, not `requirements.txt`.

**Flow:** WSL runs `launch_gazebo_bridge.sh`; Windows runs `gesture_bridge.py`. `launch_all.ps1` starts both and waits for TCP **9090**.

```powershell
cd gesture_drone\scripts
.\launch_all.ps1
.\launch_all.ps1 -CameraIndex 1
.\launch_all.ps1 -Source tello
```

The script resolves the repo path from **`$PSScriptRoot`** and converts paths for WSL. Override the distro name if needed by setting **`WSL_DISTRO_NAME`** in the environment (default **`Ubuntu-22.04`**).

**First launch:** Gazebo may take **15–30 s** (shader compile).

---

## Physical Tello (Windows only)

1. Connect the PC to the **Tello Wi‑Fi**.
2. **`tello_view.py`** — safe preview, no motors:

   ```powershell
   cd gesture_drone\scripts
   python tello_view.py
   ```

3. **`tello_hover_baseline.py`** — short hover test (no RC after takeoff), then land.
4. **`tello_real_autonomy_v1.py`** — preview **[T]** → console **Enter** to take off; **open palm** (confirmed), **Q**, or **Ctrl+C** to land. Default search uses **discrete cw/ccw** rotations; **`--search-mode rc`** restores stick yaw (can couple with lateral drift).

Fly only where regulations and safety allow, with clear space and enough battery.

---

## Limitations

- **Tello drift / coupling:** Real drones translate while yawing, especially under **`send_rc_control`**. Default **SEARCH** uses discrete rotations partly for that reason; **FACE_LOCK** still may need gain tuning.
- **Two stacks:** Gazebo uses **ROS**; the physical scripts use **djitellopy** only — behavior will not match 1:1.
- **Classifier:** Closed-set gestures; borderline poses (e.g. single finger vs peace sign) can confuse the network — see **`.ai-context/STATUS.md`**.
- **`rc=3` BUSY** on simulated takeoff/land if commands overlap plugin states.

---

## Project structure

```
MLxDrone_larp/
├── README.md                 # This file
├── requirements.txt          # Windows inference / training (pip; see GPU note inside)
├── JOURNEY.md                # ML / data narrative
├── SIMULATION_GUIDE.md       # ROS / Gazebo deep dive
├── SIMULATION_STATUS.md      # User-facing sim notes
├── .gitignore
├── .ai-context/              # Compact docs for AI assistants (architecture, status)
└── gesture_drone/
    ├── models/               # Tracked inference: gesture, best.pt, YuNet ONNX, hand_landmarker.task
    ├── logs/                 # Session CSVs (gitignored except .gitkeep)
    ├── docs/
    │   └── PERCEPTION_GATING_DESIGN.md
    └── scripts/              # bridge, sim, Tello, training, launch scripts
```

ROS workspace paths are machine-specific; see **`.ai-context/ARCHITECTURE.md`**.

---

## Story (short)

This repo grew out of a **compressed week** of ML + robotics work: custom **YOLO** hands on **HaGRID**, a small **EfficientNet** gesture head, aggressive **temporal gating** to fight noisy detections, then wiring the same pipeline to **Gazebo** over a TCP bridge so a simulated **Tello** responds to hands instead of a RC transmitter. **TrustedHand** added **MediaPipe verification** without throwing away YOLO crops; **YuNet** reduced face/hand confusion and enabled a follow **HUD** prototype. Physical **djitellopy** scripts reuse the same perception init as the sim where possible — with all the caveats of real quadrotor dynamics.

---

## Docs index

| Doc | Audience |
|-----|----------|
| **`README.md`** | Humans: install, run, limits |
| **`.ai-context/ARCHITECTURE.md`** | System design, TCP, ROS, file map |
| **`.ai-context/STATUS.md`** | What works, known bugs, tuning |
| **`SIMULATION_GUIDE.md`** | Gazebo / ROS theory |
| **`JOURNEY.md`** | Dataset / training narrative |

---

## License / safety

Software is provided as-is for research and education. You are responsible for compliance with local law and for safe operation of any real aircraft.
