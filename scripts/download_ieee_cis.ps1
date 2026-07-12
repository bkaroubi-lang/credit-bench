# Downloads IEEE-CIS Fraud Detection dataset via Kaggle CLI.
# Requires: kaggle package in active env, and %USERPROFILE%\.kaggle\kaggle.json.
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$dataDir = Join-Path $repoRoot "data\raw"
New-Item -ItemType Directory -Force -Path $dataDir | Out-Null

$kaggleConfig = Join-Path $env:USERPROFILE ".kaggle\kaggle.json"
if (-not (Test-Path $kaggleConfig)) {
    Write-Host "Missing Kaggle API token at $kaggleConfig"
    Write-Host ""
    Write-Host "  1. Go to https://www.kaggle.com/settings"
    Write-Host "  2. Click 'Create New API Token'  ->  downloads kaggle.json"
    Write-Host "  3. Move it to $kaggleConfig"
    Write-Host "  4. Accept the IEEE-CIS competition rules at"
    Write-Host "     https://www.kaggle.com/competitions/ieee-fraud-detection/rules"
    exit 1
}

$kaggleExe = Get-Command kaggle -ErrorAction SilentlyContinue
if (-not $kaggleExe) {
    Write-Host "kaggle CLI not found on PATH. Activate the credit-bench env first:"
    Write-Host "  conda activate credit-bench"
    exit 1
}

Push-Location $dataDir
try {
    Write-Host "Downloading IEEE-CIS Fraud Detection archive (~1 GB)..."
    kaggle competitions download -c ieee-fraud-detection --force
    if ($LASTEXITCODE -ne 0) { throw "kaggle download failed (exit $LASTEXITCODE)" }

    Write-Host "Extracting..."
    Expand-Archive -Force -Path "ieee-fraud-detection.zip" -DestinationPath .
    Remove-Item "ieee-fraud-detection.zip"
} finally {
    Pop-Location
}

Write-Host ""
Write-Host "IEEE-CIS files in $dataDir"
Get-ChildItem $dataDir | Format-Table Name, Length -AutoSize
