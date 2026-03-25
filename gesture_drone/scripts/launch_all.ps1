param(
    [int]$CameraIndex = 2,
    [ValidateSet('webcam', 'tello')]
    [string]$Source = 'webcam'
)

# Repo paths derived from this script (portable across clone locations).
$ScriptsDir     = $PSScriptRoot
$GestureRoot    = Split-Path -Parent $ScriptsDir
$RepoRoot       = Split-Path -Parent $GestureRoot
$PyScript       = Join-Path $RepoRoot "gesture_drone\scripts\gesture_bridge.py"
$BashScriptWin  = Join-Path $RepoRoot "gesture_drone\scripts\launch_gazebo_bridge.sh"

$Distro = $env:WSL_DISTRO_NAME
if (-not $Distro) { $Distro = "Ubuntu-22.04" }

function ConvertTo-WslPath {
    param(
        [Parameter(Mandatory)][string]$WindowsPath,
        [Parameter(Mandatory)][string]$WslDistro
    )
    if (-not (Test-Path -LiteralPath $WindowsPath)) {
        throw "Path not found: $WindowsPath"
    }
    $full = (Get-Item -LiteralPath $WindowsPath).FullName
    # \\wsl$\Distro\home\...
    if ($full -match '^\\\\wsl\$\\([^\\]+)\\(.+)$') {
        $suffix = $matches[2] -replace '\\', '/'
        return "/$suffix"
    }
    # \\wsl.localhost\Distro\home\...
    if ($full -match '^\\\\wsl\.localhost\\([^\\]+)\\(.+)$') {
        $suffix = $matches[2] -replace '\\', '/'
        return "/$suffix"
    }
    $u = & wsl.exe -d $WslDistro -- wslpath -u $full 2>$null
    if ($LASTEXITCODE -eq 0 -and $u) { return $u.Trim() }
    throw "Could not convert path to WSL (tried wslpath): $full"
}

$BashWsl = ConvertTo-WslPath -WindowsPath $BashScriptWin -WslDistro $Distro

$PythonExe = $env:MLX_PYTHON
if (-not $PythonExe) { $PythonExe = "python" }

Write-Host ""
Write-Host "  ===========================================" -ForegroundColor Cyan
Write-Host "   TELLO GAZEBO + GESTURE BRIDGE             " -ForegroundColor Cyan
if ($Source -eq 'tello') {
    Write-Host "   Video        : Tello camera -> ROS/Gazebo " -ForegroundColor Cyan
} else {
    Write-Host "   Video        : Webcam index $CameraIndex   " -ForegroundColor Cyan
}
Write-Host "  ===========================================" -ForegroundColor Cyan
Write-Host ""

# --- Step 1: open a second window running the WSL side --------------------
Write-Host "[1/2] Opening WSL window (Gazebo + ROS2 bridge)..." -ForegroundColor Green

$wslCmd = "wsl.exe -d $Distro bash `"$BashWsl`" --no-windows $CameraIndex"

$procParams = @{
    FilePath     = "powershell.exe"
    ArgumentList = "-NoLogo", "-NoExit", "-Command", $wslCmd
    PassThru     = $true
}
$wslWindow = Start-Process @procParams

# --- Step 2: wait for the ROS2 bridge on port 9090 ------------------------
Write-Host "[2/2] Waiting for ROS2 bridge on port 9090..." -ForegroundColor Green

$wslIp = $null
try {
    $raw   = wsl.exe -d $Distro -- hostname -I 2>$null
    $wslIp = ($raw -split '\s+')[0].Trim()
} catch {}
if (-not $wslIp) { $wslIp = "localhost" }
Write-Host "      WSL2 IP: $wslIp"

$ready = $false
for ($i = 0; $i -lt 90; $i++) {
    try {
        $tcp = New-Object System.Net.Sockets.TcpClient
        $tcp.Connect($wslIp, 9090)
        $tcp.Close()
        $ready = $true
        break
    } catch {
        $dots = "." * (($i % 3) + 1)
        Write-Host ("`r      Waiting" + $dots + "   ") -NoNewline
        Start-Sleep 1
    }
}

if ($ready) {
    Write-Host "`r      Bridge is ready!              " -ForegroundColor Green
} else {
    Write-Host "`r      Timed out - starting anyway." -ForegroundColor Yellow
}

Write-Host ""
if ($Source -eq 'tello') {
    Write-Host "  Starting gesture bridge with Tello camera (join TELLO Wi-Fi first)..." -ForegroundColor Green
} else {
    Write-Host "  Starting gesture recognition on webcam $CameraIndex..." -ForegroundColor Green
}
Write-Host "  Close this window or Ctrl+C to stop everything." -ForegroundColor DarkGray
Write-Host ""

# --- Run gesture_bridge.py in this window ---------------------------------
try {
    if ($Source -eq 'tello') {
        & $PythonExe $PyScript --source tello
    } else {
        & $PythonExe $PyScript $CameraIndex
    }
} finally {
    Write-Host ""
    Write-Host "  Shutting down..." -ForegroundColor Yellow
    if ($wslWindow -and -not $wslWindow.HasExited) {
        Stop-Process -Id $wslWindow.Id -Force -ErrorAction SilentlyContinue
    }
    wsl.exe -d $Distro -- bash -c "killall -9 gzserver gzclient gazebo 2>/dev/null; true" 2>$null
    Write-Host "  Done." -ForegroundColor Green
    Start-Sleep 2
}
