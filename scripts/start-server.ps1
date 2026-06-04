# Starts the local llama.cpp server.
# Defaults: cuplm-v0 > Qwen2.5-Coder-1.5B > Qwen3-0.6B (picks first found).
#
# Usage:  .\scripts\start-server.ps1
#         .\scripts\start-server.ps1 -Model models\Qwen3-0.6B-Q4_K_M.gguf
#         .\scripts\start-server.ps1 -Model models\cuplm-v0-Q8_0.gguf -Threads 4
#
# Leave this window open while you use Cup. Press Ctrl+C to stop the server.
param(
    [string]$Model = "",
    [int]$Port = 8080,
    [int]$Threads = 2   # physical cores; raise if you have more
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$server = Join-Path $root "vendor\llama-cpu\llama-server.exe"

if (-not (Test-Path $server)) { throw "llama-server.exe not found at $server" }

# Auto-detect model if not specified: prefer fine-tuned > coder > general
if (-not $Model) {
    $candidates = @(
        "models\cuplm-v0-Q8_0.gguf",
        "models\qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
        "models\Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf",
        "models\Qwen3-0.6B-Q4_K_M.gguf"
    )
    foreach ($c in $candidates) {
        if (Test-Path (Join-Path $root $c)) { $Model = $c; break }
    }
    if (-not $Model) { throw "No model found in models/. Download one first." }
}

$modelPath = Join-Path $root $Model
if (-not (Test-Path $modelPath)) { throw "Model not found: $modelPath" }

Write-Host "Starting llama.cpp server" -ForegroundColor Cyan
Write-Host "  Model:   $Model" -ForegroundColor White
Write-Host "  URL:     http://localhost:$Port" -ForegroundColor White
Write-Host "  Threads: $Threads" -ForegroundColor White
Write-Host "Keep this window open. Ctrl+C to stop." -ForegroundColor DarkGray
& $server -m $modelPath -c 4096 -t $Threads --port $Port
