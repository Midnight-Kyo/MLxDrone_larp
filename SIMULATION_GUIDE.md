# ROS2 + Gazebo Simulation -- Theory & Everything That Happened

> This document explains every concept, tool, decision, and workaround from the drone simulation phase. The goal is that after reading this, you could explain it all in an interview and reproduce it from scratch without an LLM.

---

## Table of Contents

1. [Why We Needed This](#1-why-we-needed-this)
2. [What Is ROS2](#2-what-is-ros2)
3. [What Is Gazebo](#3-what-is-gazebo)
4. [The Architecture We Built](#4-the-architecture-we-built)
5. [Step-by-Step: What Was Installed and Why](#5-step-by-step-what-was-installed-and-why)
6. [The Build System: colcon, CMake, ament](#6-the-build-system-colcon-cmake-ament)
7. [The Tello Simulation Packages](#7-the-tello-simulation-packages)
8. [The Two Scripts We Wrote](#8-the-two-scripts-we-wrote)
9. [Every Workaround and Why It Was Needed](#9-every-workaround-and-why-it-was-needed)
10. [How to Run Everything](#10-how-to-run-everything)
11. [What You Can Say in an Interview](#11-what-you-can-say-in-an-interview)

---

## 1. Why We Needed This

Up to this point, our drone simulator was a **2D panel drawn with OpenCV** (`simulate_drone.py`). A colored dot moved around a black rectangle. It proved the perception-to-command pipeline worked, but it wasn't a real robotics simulation.

The jump to ROS2 + Gazebo gives us:

- A **3D physics-accurate drone** that responds to velocity commands with realistic flight dynamics (gravity, drag, momentum)
- **ROS2** -- the actual framework used in industry for robotics. If a job description mentions "ROS" or "ROS2", this is what they mean
- **The develop-simulate-deploy workflow** -- the standard robotics cycle where you test in simulation before touching real hardware

This is the difference between "I built a toy demo" and "I built a robotics pipeline."

---

## 2. What Is ROS2

### The One-Sentence Version

ROS2 (Robot Operating System 2) is a **communication framework** that lets different programs (called "nodes") talk to each other using a publish/subscribe messaging system.

### The Longer Version

ROS2 is **not** an operating system. The name is misleading. It's a set of libraries and tools that solve a very specific problem in robotics: **how do you get multiple independent programs to communicate?**

A real robot has many processes running simultaneously:
- A camera driver producing image frames
- A perception algorithm processing those frames
- A path planner deciding where to go
- A motor controller executing movements
- A safety system monitoring battery and obstacles

Each of these is a separate program (a **node**). They need to share data. ROS2 provides the plumbing.

### Core Concepts

**Node:** A single program that does one thing. Our `gesture_ros2_node.py` is a node. The Gazebo Tello plugin is a node. The camera controller is a node. Each runs independently.

**Topic:** A named channel for one-way data streaming. Think of it like a radio frequency.
- `/drone1/cmd_vel` is a topic that carries velocity commands
- `/drone1/image_raw` is a topic that carries camera images
- Any node can **publish** (broadcast) to a topic, and any node can **subscribe** (listen) to it

**Message:** The data format sent on a topic. Each topic has a specific message type:
- `geometry_msgs/Twist` carries velocity data (linear x/y/z + angular x/y/z)
- `sensor_msgs/Image` carries camera images
- You can also define custom messages (the Tello packages define `tello_msgs/FlightData`)

**Service:** A two-way request/response pattern. Unlike topics (fire and forget), services wait for a reply:
- We call `/drone1/tello_action` with `{cmd: "takeoff"}` and get back `{rc: 1}` (1 = success)
- This is used for one-shot actions like takeoff and land, not continuous data

**Publisher:** Code that writes messages to a topic. Our node creates one:
```python
self.cmd_pub = self.create_publisher(Twist, "/drone1/cmd_vel", 10)
```
The `10` is the queue size -- it buffers up to 10 messages if the subscriber can't keep up.

**Subscriber:** Code that reads messages from a topic. The Tello Gazebo plugin subscribes to `/drone1/cmd_vel` and moves the drone whenever a new Twist message arrives.

**Client:** Code that calls a service. Our node creates one:
```python
self.tello_client = self.create_client(TelloAction, "/drone1/tello_action")
```

### The Key Insight

ROS2 nodes don't know about each other. The publisher doesn't know who's listening. The subscriber doesn't know who's publishing. They're connected only by topic names. This is called **loose coupling** and it's the whole point -- you can swap out any component without changing the others.

### ROS2 Humble

"Humble" is a version of ROS2 (named after a turtle, like all ROS releases). It's the Long Term Support (LTS) release that matches Ubuntu 22.04. Different Ubuntu versions pair with different ROS2 versions -- you can't mix them.

| Ubuntu Version | ROS2 Version |
|---------------|--------------|
| 20.04         | Foxy, Galactic |
| 22.04         | **Humble** (LTS) |
| 24.04         | Jazzy |

---

## 3. What Is Gazebo

### The One-Sentence Version

Gazebo is a **3D physics simulator** for robots -- you define a robot model, place it in a virtual world, and it obeys physics (gravity, collisions, aerodynamics).

### How It Works

Gazebo has two parts:

1. **gzserver** -- the physics engine. It runs the simulation: calculates forces, moves objects, detects collisions. It can run without any GUI (headless).

2. **gzclient** -- the 3D viewer. It renders the simulation so you can see it. This is the window that shows the 3D world with the drone in it.

When we run `gazebo --verbose`, it starts both. When we run `gzserver --verbose`, it starts only the physics engine with no GUI.

### Gazebo + ROS2

Out of the box, Gazebo and ROS2 are separate systems. The **gazebo_ros_pkgs** bridge connects them:

- `libgazebo_ros_init.so` -- publishes Gazebo's simulation clock to ROS2 (`/clock` topic)
- `libgazebo_ros_factory.so` -- provides a ROS2 service to spawn models into Gazebo

In the launch file, we start Gazebo with these plugins loaded:
```python
ExecuteProcess(cmd=[
    'gazebo', '--verbose',
    '-s', 'libgazebo_ros_init.so',    # clock bridge
    '-s', 'libgazebo_ros_factory.so',  # spawn service
    world_path
])
```

The `-s` flag loads a **system plugin** -- a shared library (`.so` file) that extends Gazebo's functionality.

### Gazebo Classic vs. New Gazebo

There are two Gazebos, and this causes confusion:

- **Gazebo Classic** (versions 1-11, package: `gazebo11`) -- the original, stable, widely-used version. This is what the tello-ros2-gazebo repo uses.
- **New Gazebo** (formerly Ignition, versions Fortress/Garden/Harmonic) -- the rewrite with a new architecture. It's the future but has less community support right now.

We use Gazebo Classic (11) because the Tello simulation was built for it.

### URDF and SDF

Robots in simulation are defined by model files:

- **URDF (Unified Robot Description Format)** -- an XML format used by ROS to describe a robot's links (rigid bodies) and joints (connections between links). The Tello is defined in `tello_1.urdf`.
- **SDF (Simulation Description Format)** -- Gazebo's own model format. More powerful than URDF (supports sensors, plugins, world properties).

The Tello URDF defines the drone's body, a camera sensor, and connects the TelloPlugin.

---

## 4. The Architecture We Built

```
┌─────────────────────────────┐       TCP        ┌──────────────────────────────────┐
│        WINDOWS              │      Socket       │           WSL2                   │
│                             │    port 9090      │                                  │
│  Webcam                     │                   │   gesture_ros2_node.py           │
│    │                        │                   │     │                            │
│    ▼                        │                   │     ├─ publishes /drone1/cmd_vel │
│  MediaPipe (hand detection) │  ──────────────►  │     ├─ calls /drone1/tello_action│
│    │                        │  JSON commands:   │     │                            │
│    ▼                        │  {"command":      │     ▼                            │
│  EfficientNet (classify)    │   "MOVE_FORWARD", │   Gazebo + TelloPlugin          │
│    │                        │   "gesture":      │     │                            │
│    ▼                        │   "two_fingers",  │     ▼                            │
│  gesture_bridge.py          │   "confidence":   │   3D Drone moves                │
│                             │   0.95}           │                                  │
└─────────────────────────────┘                   └──────────────────────────────────┘
```

### Why the Split?

The webcam runs on Windows (WSL2 doesn't have direct webcam access). ROS2 and Gazebo run on WSL2 (they're Linux-native). So we need a bridge between the two operating systems.

### Why TCP?

TCP (Transmission Control Protocol) is the most basic reliable network protocol. Windows and WSL2 share a network -- WSL2 has its own IP address that Windows can reach. We open a TCP socket on port 9090 in WSL2, and Windows connects to it.

We chose TCP over alternatives because:
- **It's simple** -- Python's `socket` library is built-in, no extra dependencies
- **It's reliable** -- TCP guarantees delivery and ordering (unlike UDP)
- **It works across the Windows/WSL2 boundary** -- they share a virtual network

The protocol is dead simple: newline-delimited JSON. Each message is one JSON object followed by `\n`:
```json
{"command": "MOVE_FORWARD", "gesture": "two_fingers", "confidence": 0.95, "timestamp": 1774184000.5}
```

---

## 5. Step-by-Step: What Was Installed and Why

### 5.1 ROS2 Humble Desktop

```bash
sudo apt install ros-humble-desktop ros-dev-tools
```

**What this installs (~2GB):**
- `rclpy` -- the Python library for creating ROS2 nodes
- `rclcpp` -- the C++ library (needed to build the Tello plugin)
- Message types: `geometry_msgs`, `sensor_msgs`, `std_msgs`, etc.
- Tools: `ros2 topic`, `ros2 service`, `ros2 launch`, `ros2 node`
- Visualization: `rviz2` (3D data viewer), `rqt` (GUI tools)

**The sourcing line:**
```bash
source /opt/ros/humble/setup.bash
```
This adds ROS2's binaries, libraries, and Python packages to your shell's PATH. Without this line, your terminal doesn't know ROS2 exists. We added it to `~/.bashrc` so it loads automatically.

### 5.2 Gazebo Classic + ROS Bridge

```bash
sudo apt install gazebo libgazebo-dev ros-humble-gazebo-ros-pkgs
```

- `gazebo` / `libgazebo-dev` -- the simulator itself and development headers (needed to compile the TelloPlugin)
- `ros-humble-gazebo-ros-pkgs` -- the bridge that connects Gazebo to ROS2

### 5.3 Additional Dependencies

```bash
sudo apt install ros-humble-cv-bridge                    # OpenCV ↔ ROS image conversion
sudo apt install ros-humble-camera-calibration-parsers   # camera intrinsics
sudo apt install ros-humble-teleop-twist-keyboard        # keyboard control of cmd_vel
sudo apt install ros-humble-joy                          # joystick driver
sudo apt install ros-humble-robot-state-publisher        # publishes robot model transforms
sudo apt install libasio-dev                             # C++ async networking (Tello driver)
sudo apt install libignition-rendering6-dev              # 3D rendering (Gazebo GUI)
pip3 install transformations                             # 3D math (quaternions, euler angles)
```

Each one was required by the Tello packages -- they're declared as dependencies in the `package.xml` files.

### 5.4 What Was NOT Installed

We did **not** install:
- Gazebo Fortress/Garden/Harmonic (the new Gazebo) -- not needed, the Tello repo uses Classic
- Docker -- we got native Gazebo working on WSL2, so no containerization needed
- Any ROS2 packages from source -- everything came from apt

---

## 6. The Build System: colcon, CMake, ament

### The Problem

ROS2 packages are not pip-installable Python packages. They're a mix of C++ and Python code with custom message definitions. You need a special build system.

### colcon

`colcon` (collective construction) is ROS2's build tool. It's like `make` but for an entire workspace of packages.

```bash
cd ~/ros2_ws
colcon build --symlink-install
```

What this does:
1. Scans `src/` for packages (identified by `package.xml` files)
2. Analyzes dependencies between packages to determine build order
3. Builds each package using its declared build system (CMake for C++, setuptools for Python)
4. Installs results to `install/`

The `--symlink-install` flag creates symlinks instead of copies, so you can edit Python files without rebuilding.

### The Workspace Structure

```
~/ros2_ws/                    # workspace root
├── src/                      # source packages go here
│   └── tello-ros2-gazebo/    # the cloned repo
│       ├── ros2_shared/      # utility package
│       └── tello_ros/
│           ├── tello_msgs/       # custom message definitions
│           ├── tello_description/ # URDF robot models
│           ├── tello_driver/     # real Tello communication
│           └── tello_gazebo/     # simulation plugin
├── build/                    # intermediate build artifacts (auto-generated)
├── install/                  # compiled results (auto-generated)
└── log/                      # build logs (auto-generated)
```

### CMakeLists.txt

Each C++ package has a `CMakeLists.txt` that tells CMake how to compile it. The Tello plugin's key parts:

```cmake
find_package(gazebo REQUIRED)        # find Gazebo headers/libraries
find_package(rclcpp REQUIRED)        # find ROS2 C++ library
find_package(tello_msgs REQUIRED)    # find our custom messages

add_library(TelloPlugin SHARED       # compile a shared library (.so)
  src/tello_plugin.cpp               # the actual drone physics code
)
```

### package.xml

Every ROS2 package has a `package.xml` declaring its name, dependencies, and build type. This is how `colcon` knows what order to build things.

### ament

`ament` is the underlying build system framework. Think of it as the glue between CMake and ROS2. When you see `ament_cmake`, `ament_package()`, or `ament_target_dependencies()`, that's ament providing ROS2-aware CMake functions.

---

## 7. The Tello Simulation Packages

The [TIERS/tello-ros2-gazebo](https://github.com/TIERS/tello-ros2-gazebo) repository comes from a Finnish university research group. It has 5 packages:

### tello_msgs

Custom ROS2 message definitions:
- `TelloAction.srv` -- a service: you send `{cmd: "takeoff"}` and get back `{rc: 1}` (OK) or `{rc: 2}` (error)
- `FlightData.msg` -- battery, temperature, flight status
- `TelloResponse.msg` -- string responses from the drone

### tello_description

URDF model files for the Tello drone. The `replace.py` script generates numbered variants (tello_1, tello_2, ... tello_8) for multi-drone simulations.

### tello_gazebo

The core simulation package:
- `tello_plugin.cpp` -- a Gazebo plugin written in C++ that simulates drone flight dynamics. It subscribes to `/drone1/cmd_vel` and applies forces to the simulated drone body. It handles takeoff, landing, hovering, and battery drain.
- `inject_entity.py` -- spawns the drone model into a running Gazebo world
- `simple.world` -- an SDF world file defining the simulation environment (ground plane, lighting)
- `simple_launch.py` -- a ROS2 launch file that starts everything

### tello_driver

The real-hardware communication driver. It talks to a physical Tello drone via UDP. We don't use this for simulation, but it had to be compiled because other packages depend on it.

### ros2_shared

A utility library with shared C++ code used by tello_driver.

### What the Launch File Does

`simple_launch.py` starts 5 processes:

1. **Gazebo** with the simple world and ROS2 bridge plugins
2. **inject_entity.py** to spawn the drone URDF into Gazebo
3. **robot_state_publisher** to broadcast the drone's URDF to the ROS2 transform tree
4. **joy_node** to read joystick input (we don't use a joystick, but it's part of the launch)
5. **tello_joy_main** to translate joystick input to cmd_vel (again, we don't use this)

The important outputs are the ROS2 interfaces the TelloPlugin creates:
- Topic `/drone1/cmd_vel` (input) -- send Twist messages to move the drone
- Service `/drone1/tello_action` (input) -- call with "takeoff" or "land"
- Topic `/drone1/odom` (output) -- the drone's position and orientation
- Topic `/drone1/image_raw` (output) -- simulated camera feed
- Topic `/drone1/flight_data` (output) -- battery, flight status

---

## 8. The Two Scripts We Wrote

### gesture_bridge.py (Windows side)

**What it does:** Runs the existing MediaPipe + EfficientNet gesture recognition pipeline, but instead of drawing a 2D simulation panel, it sends commands over TCP to WSL2.

**It is essentially `simulate_drone.py` but with the simulation panel replaced by a network socket.**

Key differences from simulate_drone.py:
- No DroneState class or 2D rendering
- Added `socket` for TCP communication
- Added `get_wsl_ip()` to auto-detect the WSL2 IP address (runs `wsl hostname -I` on Windows)
- Added `argparse` for command-line options (host, port)
- Added connection retry logic and reconnection on failure
- HUD shows "ROS2 CONNECTED" / "DISCONNECTED" status

### gesture_ros2_node.py (WSL2 side)

**What it does:** A ROS2 node that bridges TCP commands to ROS2 topics/services.

**Key design decisions:**

1. **Threading:** The TCP server runs in a separate thread because `rclpy.spin()` blocks the main thread. The ROS2 event loop (spin) handles timers and callbacks on the main thread, while the TCP thread handles network I/O.

2. **Timer-based publishing:** Instead of publishing cmd_vel only when a TCP message arrives, we publish at a fixed 10 Hz rate. This is important because the Tello plugin expects continuous velocity commands -- if you stop publishing, the drone doesn't know if you want it to hover or if the connection died. Publishing zeros explicitly means "hover."

3. **Staleness detection:** If no TCP message arrives for 3 seconds, we assume the connection died and automatically command STOP. This is a safety pattern common in real robotics.

4. **Auto-takeoff:** When the first MOVE_FORWARD or MOVE_UP command arrives, we automatically call the takeoff service. This mirrors how `simulate_drone.py` handled auto-takeoff.

**The ROS2 patterns used:**

```python
# Creating a node
class GestureDroneNode(Node):
    def __init__(self):
        super().__init__("gesture_drone_bridge")  # node name

# Publishing to a topic
self.cmd_pub = self.create_publisher(Twist, "/drone1/cmd_vel", 10)
twist = Twist()
twist.linear.x = 0.5  # forward at 0.5 m/s
self.cmd_pub.publish(twist)

# Calling a service
self.tello_client = self.create_client(TelloAction, "/drone1/tello_action")
request = TelloAction.Request()
request.cmd = "takeoff"
future = self.tello_client.call_async(request)  # non-blocking

# Timer (runs a function at a fixed rate)
self.timer = self.create_timer(0.1, self.publish_cmd_vel)  # 10 Hz

# Spinning (runs the event loop)
rclpy.spin(node)  # blocks, processes all callbacks/timers
```

---

## 9. Every Workaround and Why It Was Needed

### 9.1 Virtual Environment Conflict

**Problem:** `colcon build` was using our project's venv Python (`/home/kyo/Projects/MLxDrone_larp/venv/bin/python3`) instead of system Python (`/usr/bin/python3`). The venv doesn't have `catkin_pkg`, which ROS2's build system needs.

**Why it happened:** The venv was activated in the shell. When CMake runs Python scripts during the build, it uses whatever `python3` is on the PATH. Our venv's Python was first.

**Fix:** Deactivated the venv and removed it from PATH before building:
```bash
unset VIRTUAL_ENV
export PATH=$(echo $PATH | tr ':' '\n' | grep -v "MLxDrone_larp/venv" | tr '\n' ':')
```

**Lesson:** ROS2 builds must use system Python. Your ML venv is for ML scripts only. They live in separate worlds.

### 9.2 Script Permission Errors

**Problem:** `replace.py` in tello_description failed with "Permission denied."

**Why it happened:** Git doesn't always preserve executable permissions on clone. The CMakeLists.txt runs `replace.py` directly (like `./replace.py`), which requires the execute bit.

**Fix:** `chmod +x replace.py inject_entity.py`

**Lesson:** If a build fails with "Permission denied" on a script, it's almost always a missing execute bit.

### 9.3 Missing rclcpp_components in CMake

**Problem:** `tello_driver` failed to compile: `fatal error: rclcpp_components/register_node_macro.hpp: No such file or directory`

**Why it happened:** The original repo was written for ROS2 Galactic. The `CMakeLists.txt` listed `rclcpp_components` in `find_package()` but **didn't include it in the `ament_target_dependencies()`** for the actual build targets. In Galactic this worked (different include path resolution), in Humble it doesn't.

**Fix:** Added `rclcpp_components` to both `DRIVER_NODE_DEPS` and `JOY_NODE_DEPS` lists in the CMakeLists.txt.

**Lesson:** When using an older ROS2 repo on a newer ROS2 version, expect minor build fixes. The fix is usually adding missing dependencies to `ament_target_dependencies()`.

### 9.4 Gazebo Galactic → Humble

**Problem:** The tello-ros2-gazebo README says to install `ros-galactic-*` packages.

**Why it happened:** The repo was developed for ROS2 Galactic, which is end-of-life. We're on Humble.

**Fix:** Replaced every `ros-galactic-*` package with its `ros-humble-*` equivalent. The package names are identical except for the distro prefix.

### 9.5 inject_entity.py Can't Find `transformations`

**Problem:** The drone spawn script failed with `ModuleNotFoundError: No module named 'transformations'`.

**Why it happened:** `transformations` was installed in the venv with `pip install`, but ROS2's Python scripts run under system Python.

**Fix:** Installed it for system Python: `sudo pip3 install transformations`

### 9.6 ALSA Audio Errors

**Problem:** Gazebo printed errors about ALSA sound devices.

**Why it happened:** WSL2 doesn't have audio hardware. Gazebo tries to initialize audio for collision sounds and fails.

**Fix:** None needed -- it's harmless. Gazebo continues to work, just without sound effects.

### 9.7 Slow Gazebo First Launch

**Problem:** The first Gazebo launch took very long, and `inject_entity.py` sometimes timed out waiting for the `spawn_entity` service.

**Why it happened:** Gazebo's first startup downloads/caches 3D models from the model database and compiles shaders. This is a one-time cost.

**Fix:** Just waited longer. Subsequent launches are much faster.

---

## 10. How to Run Everything

### Step 1: Start the simulation (WSL2 terminal)

```bash
bash ~/Projects/MLxDrone_larp/gesture_drone/scripts/launch_gazebo_bridge.sh
```

This starts:
- Gazebo with the Tello drone in a 3D world
- The gesture_ros2_node.py listening on TCP port 9090

Wait about 15 seconds for Gazebo to fully initialize.

### Step 2: Start the gesture recognition (Windows PowerShell)

```powershell
python "\\wsl$\Ubuntu-22.04\home\kyo\Projects\MLxDrone_larp\gesture_drone\scripts\gesture_bridge.py" 2
```

The `2` is your camera index. This starts the webcam, runs MediaPipe + EfficientNet, and sends commands to WSL2.

### Step 3: Show gestures

- **Two fingers** → drone moves forward
- **Thumbs up** → drone moves up
- **Fist** → drone stops/hovers
- **Open palm** → drone lands

### Manual Testing Without Webcam

You can control the drone directly from WSL2:

```bash
# Takeoff
ros2 service call /drone1/tello_action tello_msgs/srv/TelloAction "{cmd: 'takeoff'}"

# Move forward
ros2 topic pub --once /drone1/cmd_vel geometry_msgs/Twist "{linear: {x: 0.5}}"

# Move up
ros2 topic pub --once /drone1/cmd_vel geometry_msgs/Twist "{linear: {z: 0.5}}"

# Stop
ros2 topic pub --once /drone1/cmd_vel geometry_msgs/Twist "{}"

# Land
ros2 service call /drone1/tello_action tello_msgs/srv/TelloAction "{cmd: 'land'}"

# Check position
ros2 topic echo /drone1/odom --once
```

---

## 11. What You Can Say in an Interview

### "Tell me about your ROS2 experience."

"I built a gesture-controlled drone simulation using ROS2 Humble and Gazebo on Ubuntu 22.04. The system has a perception pipeline running MediaPipe and a custom-trained EfficientNet classifier for hand gesture recognition, which feeds commands into a ROS2 node that publishes Twist messages to cmd_vel and calls a TelloAction service for takeoff and landing. The Tello simulation runs in Gazebo using a university research project's plugin for flight dynamics."

### "What ROS2 concepts did you use?"

"Nodes, topics, services, publishers, clients, timers, and launch files. My gesture bridge node publishes geometry_msgs/Twist to /drone1/cmd_vel at 10 Hz for continuous velocity control, and calls the tello_action service for discrete actions like takeoff and land. I also had to deal with threading -- the TCP receiver runs in a background thread while rclpy.spin handles the ROS2 event loop on the main thread."

### "What challenges did you face?"

"The main challenge was the split-system architecture. The webcam runs on Windows, but ROS2 and Gazebo are Linux-native on WSL2. I bridged them with a TCP socket sending newline-delimited JSON. I also had to adapt an older ROS2 Galactic package to build on Humble -- fixing CMake dependency issues and Python environment conflicts between my ML virtual environment and ROS2's system Python."

### "How does the simulation work?"

"Gazebo runs a physics simulation with a Tello drone model. The TelloPlugin is a C++ Gazebo plugin that subscribes to ROS2 velocity commands and applies corresponding forces to the simulated drone body. It handles flight states (grounded, hovering, flying) and simulates basic aerodynamics. The drone has a simulated camera that publishes images to a ROS2 topic, just like a real Tello would."

### "What's the difference between a topic and a service?"

"A topic is a continuous data stream -- publish and forget. The subscriber gets every message but the publisher doesn't wait for confirmation. I use topics for velocity commands because they need to flow continuously at 10 Hz. A service is a request-response pair -- the caller blocks until it gets a reply. I use services for takeoff and land because those are one-shot actions where I need to know if it succeeded."

---

## File Locations

| File | Location | Purpose |
|------|----------|---------|
| gesture_bridge.py | `gesture_drone/scripts/gesture_bridge.py` | Windows: webcam → gestures → TCP |
| gesture_ros2_node.py | `gesture_drone/scripts/gesture_ros2_node.py` | WSL2: TCP → ROS2 topics/services |
| launch_gazebo_bridge.sh | `gesture_drone/scripts/launch_gazebo_bridge.sh` | One-command launcher for WSL2 side |
| Tello ROS2 workspace | `~/ros2_ws/` | Gazebo simulation packages |
| ROS2 setup | `/opt/ros/humble/setup.bash` | ROS2 environment |
| Gazebo models | `~/ros2_ws/install/tello_gazebo/share/tello_gazebo/models` | 3D drone model |

---

## Libraries and Tools Summary

| Tool | What It Is | Why We Use It |
|------|-----------|---------------|
| **ROS2 Humble** | Robotics communication framework | Pub/sub messaging between nodes |
| **Gazebo Classic 11** | 3D physics simulator | Simulates the Tello drone with real physics |
| **gazebo_ros_pkgs** | ROS2 ↔ Gazebo bridge | Lets Gazebo and ROS2 share data |
| **colcon** | ROS2 workspace build tool | Compiles C++ plugins and custom messages |
| **CMake + ament** | Build system | Compiles the TelloPlugin C++ code |
| **rclpy** | ROS2 Python library | Our bridge node is written in Python |
| **geometry_msgs/Twist** | ROS2 message type | Carries velocity commands (linear + angular) |
| **tello_msgs/TelloAction** | Custom ROS2 service | Takeoff/land commands |
| **TCP sockets** | Network communication | Bridges Windows (webcam) to WSL2 (ROS2) |
| **JSON** | Data format | Encodes gesture commands over TCP |
| **WSLg** | WSL2 GUI subsystem | Renders the Gazebo 3D window on Windows |
