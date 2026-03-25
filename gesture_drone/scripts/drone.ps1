# Gazebo + ROS (WSL) + gesture bridge — same as launch_all.ps1
# Usage (PowerShell):
#   & "\\wsl$\Ubuntu-22.04\home\kyo\Projects\MLxDrone_larp\gesture_drone\scripts\drone.ps1"
#   & .\drone.ps1 -CameraIndex 1
#   & .\drone.ps1 -Source tello

$here = $PSScriptRoot
& (Join-Path $here "launch_all.ps1") @args
