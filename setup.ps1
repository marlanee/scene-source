$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    if (Get-Command py -ErrorAction SilentlyContinue) { & py -3.12 -m venv .venv }
    else { & python -m venv .venv }
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.10+ is required; Python 3.12 is recommended.' }
}
& '.\.venv\Scripts\python.exe' -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
if (-not (Test-Path -LiteralPath '.env')) { Copy-Item -LiteralPath '.env.example' -Destination '.env' }
& '.\.venv\Scripts\python.exe' manage.py prepare
if ($LASTEXITCODE -ne 0) { throw 'Data preparation failed.' }
& '.\.venv\Scripts\python.exe' manage.py index
if ($LASTEXITCODE -ne 0) { throw 'Index preparation failed.' }
Write-Host 'Ready. Run run.ps1 to open the service.'
