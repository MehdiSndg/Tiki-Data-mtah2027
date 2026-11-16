<#
.SYNOPSIS
  MTAH 2027 proje ortamini tek komutla kurar ve dogrular (Windows).

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File kurulum\setup_env.ps1
  powershell -ExecutionPolicy Bypass -File kurulum\setup_env.ps1 -Video     # video kare sayilari da
  powershell -ExecutionPolicy Bypass -File kurulum\setup_env.ps1 -Decode    # + tam decode (dakikalar)

  Gereksinim: uv (https://docs.astral.sh/uv/). Python 3.11 kurulu degilse uv kendisi indirir.
#>
param([switch]$Video, [switch]$Decode)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "uv bulunamadi. Asagidakilerden biriyle kurup bu scripti tekrar calistirin:"
    Write-Host '  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"'
    Write-Host "  pip install uv"
    exit 1
}

$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    Write-Host "== .venv olusturuluyor (Python 3.11)"
    uv venv --python 3.11 .venv
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host "== Paketler kilit dosyasina gore kuruluyor (kurulum\requirements.txt)"
uv pip sync --python $Python kurulum\requirements.txt
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "== ffmpeg .venv icine yerlestiriliyor"
& $Python kurulum\check_env.py --install-ffmpeg
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "== Dogrulama"
$CheckArgs = @()
if ($Video) { $CheckArgs += "--video" }
if ($Decode) { $CheckArgs += "--decode" }
& $Python kurulum\check_env.py @CheckArgs
$Status = $LASTEXITCODE

Write-Host ""
Write-Host "Ortami etkinlestirmek icin:  .venv\Scripts\Activate.ps1"
exit $Status
