param(
  [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
& $Python -m pip install --upgrade pip
& $Python -m pip install -r requirements.txt

if (Test-Path "models/sam2/segment-anything-2-real-time") {
  & $Python -m pip install -e models/sam2/segment-anything-2-real-time
} else {
  Write-Host "SAM2 repo not found. Run scripts/download_assets.ps1 first."
}
