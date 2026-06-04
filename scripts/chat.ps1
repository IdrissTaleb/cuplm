# Start an interactive Cup chat against the local server.
# Usage:  .\scripts\chat.ps1                 # chat with current folder as workdir
#         .\scripts\chat.ps1 -Workdir C:\some\project
# Requires the server to be running (see start-server.ps1).
param(
    [int]$Port = 8080,
    [string]$Workdir = (Get-Location).Path
)
$env:CUP_SERVER = "http://localhost:$Port"
# --chat-format: proper <|im_start|> template for instruct models (Qwen2.5-Coder etc.)
# --no-think:    skip <think> phase on Qwen3 reasoning models
python -m cup.cli chat --workdir $Workdir --no-think --chat-format
