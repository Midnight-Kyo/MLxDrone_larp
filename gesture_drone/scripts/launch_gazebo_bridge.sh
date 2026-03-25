#!/bin/bash
# Launch the full MLxDrone stack in one command:
#   1. Gazebo with the Tello drone (WSL2)
#   2. gesture_ros2_node.py TCP bridge (WSL2)
#   3. gesture_bridge.py vision pipeline (Windows PowerShell window, auto-spawned)
#
# Usage:
#   bash ~/Projects/MLxDrone_larp/gesture_drone/scripts/launch_gazebo_bridge.sh [camera_index]
#
#   camera_index  Windows camera device index (default: 0). Use 1, 2 etc. for
#                 external webcams. The original default was 2 in older docs;
#                 adjust to whichever index your webcam is on Windows.

CAMERA_INDEX="${1:-0}"
SPAWN_WINDOWS=true
for arg in "$@"; do
    [[ "$arg" == "--no-windows" ]] && SPAWN_WINDOWS=false
done
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── Build Windows UNC path for gesture_bridge.py ──────────────────────────
# WSL_DISTRO_NAME is set automatically by WSL (e.g. "Ubuntu-22.04").
# The UNC path lets powershell.exe reach files inside WSL from Windows.
WIN_SCRIPT="\\\\wsl\$\\${WSL_DISTRO_NAME}${SCRIPT_DIR}/gesture_bridge.py"
WIN_SCRIPT="${WIN_SCRIPT//\//\\}"   # convert forward slashes to backslashes

echo "=============================================="
echo "  TELLO GAZEBO + GESTURE BRIDGE  (all-in-one)"
echo "=============================================="
echo "  Camera index : $CAMERA_INDEX"
echo "  Windows path : $WIN_SCRIPT"
echo "=============================================="
echo ""

# ── Clean slate ───────────────────────────────────────────────────────────
killall -9 gzserver gzclient gazebo 2>/dev/null || true
sleep 1

# ── ROS2 environment ──────────────────────────────────────────────────────
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
export GAZEBO_MODEL_PATH=~/ros2_ws/install/tello_gazebo/share/tello_gazebo/models
source /usr/share/gazebo/setup.sh

# Use RTX 3070 GPU via WSLg d3d12 passthrough
export GALLIUM_DRIVER=d3d12
export MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA

# ── Launch Gazebo ─────────────────────────────────────────────────────────
echo "[1/3] Starting Gazebo..."
ros2 launch tello_gazebo pretty_launch.py &
GAZEBO_PID=$!

echo "      Waiting for Gazebo to initialise (10 s)..."
sleep 10

# ── Launch ROS2 bridge node (WSL side) ────────────────────────────────────
echo "[2/3] Starting gesture_ros2_node (TCP listener on port 9090)..."
python3 "$SCRIPT_DIR/gesture_ros2_node.py" &
ROS_BRIDGE_PID=$!
sleep 1

# ── Launch gesture_bridge.py on Windows (skipped when called from launch_all.ps1) ──
WIN_PID=""
if [ "$SPAWN_WINDOWS" = true ]; then
    echo "[3/3] Spawning Windows PowerShell window for gesture_bridge.py..."
    WIN_PID=$(powershell.exe -Command "
\$p = Start-Process powershell.exe \`
    -ArgumentList '-NoExit', '-NoLogo', \`
                  '-Command', 'python \"$WIN_SCRIPT\" $CAMERA_INDEX' \`
    -PassThru
\$p.Id
" 2>/dev/null | tr -d '\r\n')

    if [ -n "$WIN_PID" ] && [ "$WIN_PID" -gt 0 ] 2>/dev/null; then
        echo "      Windows PowerShell window launched (PID $WIN_PID)."
    else
        echo "      WARNING: Could not confirm Windows PID — window may still have opened."
        echo "      If the gesture window did not appear, run manually on PowerShell:"
        echo "        python \"$WIN_SCRIPT\" $CAMERA_INDEX"
        WIN_PID=""
    fi
else
    echo "[3/3] Skipping Windows spawn (called from launch_all.ps1)."
fi

echo ""
echo "All processes running. Press Ctrl+C to stop everything."
echo ""

# ── Cleanup on exit ───────────────────────────────────────────────────────
cleanup() {
    echo ""
    echo "Shutting down..."

    # Kill Windows PowerShell window (and its children) if we have the PID
    if [ -n "$WIN_PID" ]; then
        powershell.exe -Command "
            Stop-Process -Id $WIN_PID -Force -ErrorAction SilentlyContinue
            # Also kill any python processes that were children of that window
            Get-WmiObject Win32_Process | Where-Object { \$_.ParentProcessId -eq $WIN_PID } |
                ForEach-Object { Stop-Process -Id \$_.ProcessId -Force -ErrorAction SilentlyContinue }
        " 2>/dev/null || true
    fi

    kill $ROS_BRIDGE_PID 2>/dev/null || true
    kill $GAZEBO_PID 2>/dev/null || true
    killall -9 gzserver gzclient gazebo 2>/dev/null || true
    echo "Done."
}

trap cleanup EXIT INT TERM

wait $GAZEBO_PID
