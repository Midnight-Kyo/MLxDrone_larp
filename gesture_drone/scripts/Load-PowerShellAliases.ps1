# MLxDrone — PowerShell shims (drone / bridge / simulate / tello)
#
# Auto-install into $PROFILE (run once from Windows PowerShell):
#   cd <repo>\gesture_drone\scripts
#   .\Install-MLxPowerShellProfile.ps1
#
# Or load manually for this session only:
#   . .\Load-PowerShellAliases.ps1
#
# Python selection (first match wins; no need to set env vars if you use the default winvenv path):
#   1. $env:MLX_PYTHON — explicit python.exe (highest priority)
#   2. $env:MLX_WIN_VENV_HOME — folder that contains .venv; uses .venv\Scripts\python.exe
#   3. $env:USERPROFILE\mlx-drone-winvenv\.venv\Scripts\python.exe — common Windows GPU/camera venv
#   Add this file to $PROFILE (see Install-MLxPowerShellProfile.ps1) so every new PowerShell picks it up.

$script:MLxScripts = $PSScriptRoot
$script:MLxRepo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path

function script:Get-MlxPython {
    if ($env:MLX_PYTHON) {
        return $env:MLX_PYTHON
    }
    if ($env:MLX_WIN_VENV_HOME) {
        $venvRoot = Join-Path $env:MLX_WIN_VENV_HOME ".venv"
        $exe = Join-Path $venvRoot "Scripts\python.exe"
        if (Test-Path -LiteralPath $exe) {
            return $exe
        }
    }
    $auto = Join-Path $env:USERPROFILE "mlx-drone-winvenv\.venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $auto) {
        return $auto
    }
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
    # ValueFromRemainingArguments: reliably forward `tello --autonomy-preview`, etc.
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
    # -u unbuffered; launcher prints before importing torch/YOLO (avoids "silent hang" on cold start).
    & $py -u (Join-Path $script:MLxScripts "tello_autonomy_launcher.py") @args
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
  tello-realtest     Real Tello: takeoff → hover → you type land (djitellopy, no ROS)
  tello-autonomy     Real Tello v1 (via launcher: message before slow torch/YOLO import)

Repo: $($script:MLxRepo)

Python: auto-uses ``%USERPROFILE%\mlx-drone-winvenv\.venv`` when that exists; otherwise set:
  `$env:MLX_PYTHON = 'C:\path\to\venv\Scripts\python.exe'
or ``MLX_WIN_VENV_HOME`` to the parent folder of ``.venv``.

"@
}

Write-Host "MLxDrone aliases loaded: drone, bridge, simulate, tello, tello-realtest, tello-autonomy, mlx-help" -ForegroundColor Green
