param(
  [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
& $Python "scripts/download_assets.py" @Args
