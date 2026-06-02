# Start an interactive Cup chat against the local server.
# Usage:  .\scripts\chat.ps1                 # chat about the current folder
#         .\scripts\chat.ps1 -Workdir C:\some\project
# Requires the server to be running (see start-server.ps1).
param(
    [int]$Port = 8080,
    [string]$Workdir = (Get-Location).Path
)
$env:CUP_SERVER = "http://localhost:$Port"
python -m cup.cli chat --workdir $Workdir --no-think
