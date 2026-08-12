param(
    [switch]$NoOpen
)

$ErrorActionPreference = 'Stop'

$reportRoot = Split-Path -Parent $PSScriptRoot
$serverScript = Join-Path $PSScriptRoot 'serve_video_reports.cjs'
$node = (Get-Command node -ErrorAction Stop).Source

function Test-Portal([int]$Port) {
    try {
        $response = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/local-video-inventory.json" -UseBasicParsing -TimeoutSec 1
        return $response.StatusCode -eq 200 -and $response.Content -match '"videos"\s*:\s*765'
    }
    catch {
        return $false
    }
}

function Test-PortInUse([int]$Port) {
    $client = [System.Net.Sockets.TcpClient]::new()
    try {
        $task = $client.ConnectAsync('127.0.0.1', $Port)
        return $task.Wait(350) -and $client.Connected
    }
    catch {
        return $false
    }
    finally {
        $client.Dispose()
    }
}

$port = $null
foreach ($candidate in 18768..18778) {
    if (Test-Portal $candidate) {
        $port = $candidate
        break
    }
    if (-not (Test-PortInUse $candidate)) {
        $port = $candidate
        $stdout = Join-Path $env:TEMP "vlm-video-server-$candidate.out.log"
        $stderr = Join-Path $env:TEMP "vlm-video-server-$candidate.err.log"
        $arguments = @("`"$serverScript`"", "`"$reportRoot`"", "$candidate")
        Start-Process -FilePath $node -ArgumentList $arguments -WindowStyle Hidden `
            -RedirectStandardOutput $stdout -RedirectStandardError $stderr
        foreach ($attempt in 1..20) {
            Start-Sleep -Milliseconds 250
            if (Test-Portal $candidate) { break }
        }
        if (-not (Test-Portal $candidate)) {
            throw "Local video server failed to start. Log: $stderr"
        }
        break
    }
}

if ($null -eq $port) {
    throw 'Ports 18768-18778 are occupied; the local video server cannot start.'
}

$uri = "http://127.0.0.1:$port/"
if (-not $NoOpen) {
    Start-Process $uri
}
Write-Output $uri
