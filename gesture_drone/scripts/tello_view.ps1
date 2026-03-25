# Physical Tello camera + gesture HUD (no flight commands).
# 1) Join the drone WiFi (TELLO-XXXX)   2) Run:  .\tello_view.ps1
# From repo root:  .\gesture_drone\scripts\tello_view.ps1

$ErrorActionPreference = "Stop"

$Distro   = "Ubuntu-22.04"
$WinBase  = "\\wsl$\$Distro\home\kyo\Projects\MLxDrone_larp"
$PyScript = "$WinBase\gesture_drone\scripts\tello_view.py"

python $PyScript
