#!/usr/bin/env bash
# MTAH 2027 proje ortamini tek komutla kurar ve dogrular (macOS / Linux).
#
#   bash kurulum/setup_env.sh              # kurulum + dogrulama
#   bash kurulum/setup_env.sh --video      # video kare sayilari da
#   bash kurulum/setup_env.sh --decode     # + tactical videonun tam decode'u (dakikalar)
#
# Gereksinim: uv (https://docs.astral.sh/uv/). Python 3.11 kurulu degilse uv kendisi indirir.
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
    echo "uv bulunamadi. Asagidakilerden biriyle kurup bu scripti tekrar calistirin:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "  pip install uv"
    exit 1
fi

PY=.venv/bin/python
if [ ! -x "$PY" ]; then
    echo "== .venv olusturuluyor (Python 3.11)"
    uv venv --python 3.11 .venv
fi

echo "== Paketler kilit dosyasina gore kuruluyor (kurulum/requirements.txt)"
uv pip sync --python "$PY" kurulum/requirements.txt

echo "== ffmpeg .venv icine yerlestiriliyor"
"$PY" kurulum/check_env.py --install-ffmpeg

echo "== Dogrulama"
status=0
"$PY" kurulum/check_env.py "$@" || status=$?

echo
echo "Ortami etkinlestirmek icin:  source .venv/bin/activate"
exit $status
