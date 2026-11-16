#!/bin/bash
# The strict-fp32 ball_v1 pass over the whole match on a Colab GPU, in the background.
#
#     bash scripts/colab_ball_pass.sh /content/drive/MyDrive/<folder with tactical/>
#
# After scripts/colab_setup.sh pinned, from the repository root. The tactical folder is copied to
# the local disk first (reading 5 GB of video through the Drive mount is slow); its checksum is
# printed to compare with the local copy. Chunks and the log go to Drive, so the run survives a
# lost connection, and analysis/p3_yolo_pass.py resumes from the last finished chunk.
set -euxo pipefail
SRC=$1
D=/content/drive/MyDrive/mtah2027
OUT=$D/runs/ball_v1_pass_fp32_$(nvidia-smi --query-gpu=name --format=csv,noheader | tr ' ' '_' | tr -cd 'A-Za-z0-9_')
mkdir -p /content/data/tactical "$D/runs"
cp -n "$SRC"/tactical/* /content/data/tactical/
sha256sum /content/data/tactical/BAR-ATH_tactical.mp4
export MTAH_DATA_DIR=/content/data
nohup .venv/bin/python analysis/p3_yolo_pass.py --weights models/ball_v1.pt --out "$OUT" \
  --classes 0 --conf 0.1 --fp32 > "$OUT.log" 2>&1 &
echo "STARTED pid $! out $OUT"
