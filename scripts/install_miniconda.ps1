# Silent Miniconda install to %USERPROFILE%\miniconda3
# Does NOT modify PATH. Activate via:
#   & "$env:USERPROFILE\miniconda3\shell\condabin\conda-hook.ps1"
$ErrorActionPreference = "Stop"

$target = Join-Path $env:USERPROFILE "miniconda3"
if (Test-Path $target) {
    Write-Host "Miniconda already exists at $target - skipping install."
    Write-Host "To activate: & `"$target\shell\condabin\conda-hook.ps1`""
    exit 0
}

$installer = Join-Path $env:TEMP "Miniconda3-latest-Windows-x86_64.exe"
$url = "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe"

Write-Host "Downloading Miniconda installer from $url ..."
Invoke-WebRequest -Uri $url -OutFile $installer

Write-Host "Installing (silent, JustMe, no PATH mods) to $target ..."
$args = "/InstallationType=JustMe /RegisterPython=0 /AddToPath=0 /S /D=$target"
Start-Process -Wait -FilePath $installer -ArgumentList $args

Remove-Item $installer -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "Miniconda installed at $target"
Write-Host ""
Write-Host "Next steps in a fresh PowerShell window:"
Write-Host "  & `"$target\shell\condabin\conda-hook.ps1`""
Write-Host "  conda env create -f environment.yml"
Write-Host "  conda activate credit-bench"
