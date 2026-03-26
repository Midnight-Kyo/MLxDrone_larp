# MLxDrone — PowerShell shims (drone / bridge / simulate / tello)
#
# Auto-install into $PROFILE (run once from Windows PowerShell):
#   cd <repo>\gesture_drone\scripts
#   .\Install-MLxPowerShellProfile.ps1
#
# Or load manually for this session only:
#   . .\Load-PowerShellAliases.ps1
#
# Use the same Python you use for this project (activate your Windows venv first, or set MLX_PYTHON).

$script:MLxScripts = $PSScriptRoot
$script:MLxRepo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path

function script:Get-MlxPython {
    if ($env:MLX_PYTHON) { return $env:MLX_PYTHON }
    return "python"
}

function global:drone {
    & (Join-Path $script:MLxScripts "launch_all.ps1") @args
}

function global:simulate {
    $py = Get-MlxPython
    & $py (Join-Path $script:MLxScripts "simulate_drone.py") @args
}

function global:bridge {
    $py = Get-MlxPython
    & $py (Join-Path $script:MLxScripts "gesture_bridge.py") @args
}

function global:tello {
    # ValueFromRemainingArguments: reliably forward `tello --enhance-stream` / `tello -E`
    # (plain `$args` can fail depending on profile / PS version).
    param(
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]] $Passthrough
    )
    $py = Get-MlxPython
    $path = Join-Path $script:MLxScripts "tello_view.py"
    if ($null -ne $Passthrough -and $Passthrough.Count -gt 0) {
        & $py $path @Passthrough
    } else {
        & $py $path
    }
}

function global:tello-realtest {
    $py = Get-MlxPython
    & $py (Join-Path $script:MLxScripts "tello_real_flight_test.py") @args
}

function global:tello-autonomy {
    $py = Get-MlxPython
    & $py (Join-Path $script:MLxScripts "tello_real_autonomy_v1.py") @args
}

function global:mlx-help {
    Write-Host @"

MLxDrone commands (after loading Load-PowerShellAliases.ps1):

  drone              Gazebo + ROS (WSL) + gesture bridge  [launch_all.ps1 defaults]
  drone -CameraIndex 1
  drone -Source tello

  bridge 2           Gesture bridge only, webcam index 2
  bridge --source tello

  simulate           2D simulator + gestures
  simulate 1

  tello              Tello camera HUD only (no ROS)
  tello --enhance-stream   or   tello -E   Bilateral + unsharp + ENHANCED HUD badge
  tello-realtest     Real Tello: takeoff → hover → you type land (djitellopy, no ROS)
  tello-autonomy     Real Tello v1: SEARCH → FACE_LOCK (yaw) → open palm land (no ROS)

Repo: $($script:MLxRepo)

Set MLX_PYTHON to your venv python.exe if `python` is wrong, e.g.:
  `$env:MLX_PYTHON = 'C:\path\to\venv\Scripts\python.exe'

"@
}

Write-Host "MLxDrone aliases loaded: drone, bridge, simulate, tello, tello-realtest, tello-autonomy, mlx-help" -ForegroundColor Green
