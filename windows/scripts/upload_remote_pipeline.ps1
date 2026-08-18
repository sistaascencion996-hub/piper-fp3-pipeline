$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$ConfigPath = Join-Path $Root "config\pipeline.local.json"
$Cfg = Get-Content $ConfigPath -Raw | ConvertFrom-Json
$Remote = "$($Cfg.remote.ssh_user)@$($Cfg.remote.host)"

$RemoteDest = "$HOME/piper-fp3-pipeline"

ssh $Remote "mkdir -p '$RemoteDest'"
if ($LASTEXITCODE -ne 0) { throw "SSH failed" }

scp -r (Join-Path $Root "remote\*") "${Remote}:$RemoteDest/"
if ($LASTEXITCODE -ne 0) { throw "SCP failed" }

Write-Host ""
Write-Host "Remote package uploaded to $RemoteDest"
Write-Host "Then on Ubuntu:"
Write-Host "  cd $RemoteDest/scripts"
Write-Host "  chmod +x pipeline_remote.sh"
Write-Host "  ./pipeline_remote.sh status"
