# Run ONCE from Windows PowerShell. Adds MLxDrone aliases to your $PROFILE.
#   cd <repo>\gesture_drone\scripts
#   .\Install-MLxPowerShellProfile.ps1

$ErrorActionPreference = "Stop"
$loader = Join-Path $PSScriptRoot "Load-PowerShellAliases.ps1"
if (-not (Test-Path -LiteralPath $loader)) {
    Write-Error "Missing: $loader"
}

$line = ". `"$loader`""
$begin = "# --- BEGIN MLxDrone (auto) ---"
$end   = "# --- END MLxDrone (auto) ---"
$block = @"

$begin
$line
$end

"@

$profileDir = Split-Path -Parent $PROFILE
if (-not (Test-Path -LiteralPath $profileDir)) {
    New-Item -ItemType Directory -Path $profileDir -Force | Out-Null
}

if (-not (Test-Path -LiteralPath $PROFILE)) {
    New-Item -ItemType File -Path $PROFILE -Force | Out-Null
}

$existing = Get-Content -LiteralPath $PROFILE -Raw -ErrorAction SilentlyContinue
if ($null -eq $existing) { $existing = "" }

if ($existing -match [regex]::Escape($begin)) {
    Write-Host "MLxDrone block already in profile: $PROFILE" -ForegroundColor Yellow
    exit 0
}

Add-Content -LiteralPath $PROFILE -Value $block -Encoding UTF8
Write-Host "Appended MLxDrone loader to: $PROFILE" -ForegroundColor Green
Write-Host "Open a NEW PowerShell window, then run: drone | bridge 2 | simulate | tello | mlx-help" -ForegroundColor Cyan
