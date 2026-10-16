# Lacuna installer — Windows 11.
# Usage: powershell -ExecutionPolicy Bypass -File install.ps1
$ErrorActionPreference = "Stop"

function Test-Python {
    try {
        $v = & python --version 2>$null
        return $v -match "Python 3\.(1[0-9]|[2-9][0-9])"
    } catch { return $false }
}

if (-not (Test-Python)) {
    Write-Host "[lacuna] Python 3.10+ not found; installing via winget..."
    winget install --id Python.Python.3.12 -e --accept-package-agreements --accept-source-agreements
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
                [System.Environment]::GetEnvironmentVariable("Path", "User")
}

python --version
pip install .
Write-Host "[lacuna] installed. Try: lacuna simulate"
