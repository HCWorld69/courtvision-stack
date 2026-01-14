param(
  [string]$Config = "configs/default.yaml",
  [string]$Step = "all",
  [string]$Video = ""
)

$ErrorActionPreference = "Stop"
$cmd = @("-m", "src.pipeline", "--config", $Config, "--step", $Step)
if ($Video -ne "") {
  $cmd += @("--video", $Video)
}
python @cmd
