$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) { throw 'Run setup.ps1 first.' }
Write-Host 'Scene & Source: http://127.0.0.1:8503/'
$browserHelper = Join-Path $PSScriptRoot 'open_browser.ps1'
Start-Process powershell.exe -WindowStyle Hidden -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', ('"{0}"' -f $browserHelper))
& '.\.venv\Scripts\python.exe' -m uvicorn scene_source.api:app --host 127.0.0.1 --port 8503
