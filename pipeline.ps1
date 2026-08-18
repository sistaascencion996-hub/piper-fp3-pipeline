param(
    [Parameter(Position=0)]
    [ValidateSet("collect","upload","run","status")]
    [string]$Command,

    [int]$Episode = 0,
    [double]$Duration = 30.0
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$ConfigPath = Join-Path $Root "config\pipeline.local.json"
$Cfg = Get-Content $ConfigPath -Raw | ConvertFrom-Json

$Python = $Cfg.windows.python
$RawRoot = $Cfg.windows.raw_root
$Remote = "$($Cfg.remote.ssh_user)@$($Cfg.remote.host)"
$RemoteRawRoot = $Cfg.remote.raw_root

function EpisodeName([int]$N) {
    if ($N -le 0) {
        throw "请使用 -Episode 指定编号，例如：.\pipeline.ps1 collect -Episode 21"
    }
    return ("episode_{0:D6}" -f $N)
}

switch ($Command) {
    "collect" {
        $Ep = EpisodeName $Episode
        New-Item -ItemType Directory -Force -Path $RawRoot | Out-Null

        $Collector = Join-Path $Root "windows\01_collect\collect_piper_d405_d455.py"

        & $Python $Collector `
          --episode $Ep `
          --output $RawRoot `
          --duration $Duration `
          --record-fps 10 `
          --countdown 5 `
          --instruction $Cfg.project.task `
          --d405-serial $Cfg.cameras.D405_wrist_serial `
          --d455-serial $Cfg.cameras.D455_external_serial

        if ($LASTEXITCODE -ne 0) {
            throw "采集脚本退出码：$LASTEXITCODE"
        }
    }

    "upload" {
        $Ep = EpisodeName $Episode
        $LocalEpisode = Join-Path $RawRoot $Ep
        if (-not (Test-Path $LocalEpisode)) {
            throw "找不到：$LocalEpisode"
        }

        ssh $Remote "mkdir -p '$RemoteRawRoot'"
        if ($LASTEXITCODE -ne 0) {
            throw "SSH 创建远端目录失败"
        }

        scp -r $LocalEpisode "${Remote}:$RemoteRawRoot/"
        if ($LASTEXITCODE -ne 0) {
            throw "SCP 上传失败"
        }

        Write-Host "上传完成：$Ep"
        Write-Host "远端下一步：./pipeline_remote.sh prepare"
    }

    "run" {
        $Client = Join-Path $Root "windows\05_robot\piper_fp3_robot_client.py"

        & $Python $Client `
          --server-ip $Cfg.remote.host `
          --port $Cfg.inference.port `
          --hand-serial $Cfg.cameras.D405_wrist_serial `
          --varied-serial $Cfg.cameras.D455_external_serial `
          --speed-percent $Cfg.robot_run.speed_percent `
          --return-speed-percent $Cfg.robot_run.return_speed_percent `
          --return-timeout $Cfg.robot_run.return_timeout_sec `
          --gripper-width $Cfg.robot_run.gripper_width_m `
          --gripper-force $Cfg.robot_run.gripper_force `
          --min-depth $Cfg.robot_run.min_depth_m `
          --max-depth $Cfg.robot_run.max_depth_m
    }

    "status" {
        Write-Host "Windows Python: $Python"
        Write-Host "Raw root: $RawRoot"
        Write-Host "Remote: $Remote"
        Write-Host "D405: $($Cfg.cameras.D405_wrist_serial)"
        Write-Host "D455: $($Cfg.cameras.D455_external_serial)"
        Write-Host "FP3 server: $($Cfg.remote.host):$($Cfg.inference.port)"
    }

    default {
        Write-Host "用法："
        Write-Host "  .\pipeline.ps1 collect -Episode 21 -Duration 30"
        Write-Host "  .\pipeline.ps1 upload  -Episode 21"
        Write-Host "  .\pipeline.ps1 run"
        Write-Host "  .\pipeline.ps1 status"
    }
}
