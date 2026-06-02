# Starts the local llama.cpp server with the tiny Qwen3-0.6B model.
# Usage:  .\scripts\start-server.ps1            # uses Qwen3-0.6B on port 8080
#         .\scripts\start-server.ps1 -Model models\Llama-3.2-3B-Instruct-Q4_K_M.gguf
#
# Leave this window open while you use Cup. Press Ctrl+C here to stop the server.
param(
    [string]$Model = "models\Qwen3-0.6B-Q4_K_M.gguf",
    [int]$Port = 8080,
    [int]$Threads = 2   # physical cores; raise if you have more
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$server = Join-Path $root "vendor\llama-cpu\llama-server.exe"
$modelPath = Join-Path $root $Model

if (-not (Test-Path $server))    { throw "llama-server.exe not found at $server" }
if (-not (Test-Path $modelPath)) { throw "Model not found at $modelPath" }

Write-Host "Starting llama.cpp server: $Model on http://localhost:$Port (threads=$Threads)" -ForegroundColor Cyan
Write-Host "Keep this window open. Ctrl+C to stop." -ForegroundColor DarkGray
& $server -m $modelPath -c 4096 -t $Threads --port $Port
