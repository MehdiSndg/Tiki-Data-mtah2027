#!/bin/bash
# Set up our environment on a Colab runtime, without touching Colab's own Python.
#
#     bash scripts/colab_setup.sh pinned    # the project's locked env (uv.lock): torch 2.7.1+cu128, any GPU
#     bash scripts/colab_setup.sh cu128     # the 23 Oct trial env (torch 2.7.1+cu128 on top of the old cu124
#                                           # lock); kept to reproduce notes/BULGULAR.md
#
# Run from the repository root (cloned from the Drive bundle). Colab sets UV_SYSTEM_PYTHON=true,
# which makes `uv pip` install into the system Python; it is unset here and every install names its
# interpreter. The system package list is recorded before and after, and the run fails if it changed.
set -euxo pipefail
unset UV_SYSTEM_PYTHON UV_INSTALL_DIR UV_BREAK_SYSTEM_PACKAGES UV_CONSTRAINT UV_BUILD_CONSTRAINT
MODE=$1
D=/content/drive/MyDrive/mtah2027
python -m pip freeze > /content/system_before.txt
git log -1 --format="commit %h %s"
mkdir -p models && cp "$D/pkg/ball_v1.pt" "$D/pkg/yolo11m.pt" models/
uv --version
if [ "$MODE" = pinned ]; then
  uv sync --frozen
  PY=.venv/bin/python
else
  uv venv /content/venv128 -p 3.11
  PY=/content/venv128/bin/python
  uv export --frozen --no-hashes --no-emit-project \
    | grep -v -E "^(torch|torchvision|triton|nvidia-[a-z0-9-]+)==" > /content/req_rest.txt
  uv pip install --python "$PY" -r /content/req_rest.txt
  # torch last, so nothing installed after it can bring the cu124 build back
  uv pip install --python "$PY" --reinstall-package torch --reinstall-package torchvision \
    torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
  "$PY" -c "import torch; assert torch.__version__.startswith('2.7.1') and torch.version.cuda == '12.8', torch.__version__"
fi
"$PY" -c "import torch, ultralytics, numpy; print('torch', torch.__version__, '| cuda', torch.version.cuda, '|', torch.cuda.get_device_name(0), '| ultralytics', ultralytics.__version__, '| numpy', numpy.__version__)"
python -m pip freeze > /content/system_after.txt
cmp -s /content/system_before.txt /content/system_after.txt && echo SYSTEM_UNCHANGED || { echo SYSTEM_CHANGED; exit 1; }
echo "SETUP_DONE $MODE $PY"
