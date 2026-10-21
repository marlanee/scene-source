$ErrorActionPreference = 'SilentlyContinue'
for ($attempt = 0; $attempt -lt 40; $attempt++) {
    try {
        $ready = Invoke-RestMethod -Uri 'http://127.0.0.1:8503/api/status' -TimeoutSec 2
        if ($ready.version -eq '2.0.0') {
            Start-Process 'http://127.0.0.1:8503/'
            break
        }
    } catch {}
    Start-Sleep -Milliseconds 500
}
