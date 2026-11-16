#!/bin/bash
# A fresh Colab runtime, ready to work, in one go (after drive.mount('/content/drive') in a Python cell):
#
#     !bash /content/drive/MyDrive/mtah2027/pkg/colab_bootstrap.sh      (a copy of scripts/colab_bootstrap.sh)
#
# 1. the repository from the Drive bundle (branch p5-env-cu128) at /content/Hackathon;
# 2. the locked environment (scripts/colab_setup.sh pinned: torch 2.7.1+cu128);
# 3. the match and external data copied to the local disk, the videos checked against their
#    SHA-256, and linked into the repository's data/;
# 4. the Drive caches (not the tile archive) and the strict-fp32 Blackwell pass;
# 5. the shared caches built once (DFL, Metrica, our match), so parallel runs never write them.
# Everything is logged to /content/bootstrap.log; the last line is BOOTSTRAP_DONE or the failure.
set -euxo pipefail
exec > >(tee /content/bootstrap.log) 2>&1
D=/content/drive/MyDrive/mtah2027
TACTICAL_SHA=ff249f072aa7cd89a4a7f603a08f399aa0d34bd429982c847cd4a9373f645f14
BROADCAST_SHA=3dc2a7d952302d33df69a26e3d8935648015f8d7e1b86bf614436b7009abf9f8
PASS=$D/runs/ball_v1_pass_fp32_NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd /content
[ -d Hackathon ] || git clone -q "$D/pkg/mtah.bundle" Hackathon
cd Hackathon
# one command per line: set -e does not stop inside an && chain
git fetch -q "$D/pkg/mtah.bundle" p5-env-cu128:refs/remotes/b/p5
git checkout -q p5-env-cu128
# reports regenerated on an earlier runtime: kept in a stash, never thrown away
git diff --quiet || git stash push -q -m "colab_bootstrap $(date +%F_%T)"
git merge -q --ff-only b/p5
[ "$(git rev-parse HEAD)" = "$(git rev-parse b/p5)" ]
git log -1 --oneline
which uv || pip install -q uv
bash scripts/colab_setup.sh pinned

mkdir -p /content/data
for d in tactical broadcast external; do [ -d /content/data/$d ] || cp -r "$D/data/$d" /content/data/; done
echo "$TACTICAL_SHA  /content/data/tactical/BAR-ATH_tactical.mp4" | sha256sum -c
echo "$BROADCAST_SHA  /content/data/broadcast/BAR-ATH_broadcast.mp4" | sha256sum -c
mkdir -p data cache
for d in tactical broadcast external; do [ -e data/$d ] || ln -s /content/data/$d data/$d; done

for f in "$D"/cache/*; do
  case "$(basename "$f")" in ball_v1_tiles.tar) continue ;; esac
  [ -e "cache/$(basename "$f")" ] || cp -r "$f" cache/
done
[ -e cache/ball_v1_pass_blackwell ] || cp -r "$PASS" cache/ball_v1_pass_blackwell

export MTAH_DATA_DIR=/content/data
.venv/bin/python -c "
import glob, os, sys; sys.path.insert(0, '.')
from mtah.external import DFL, from_dfl, from_metrica_sample
from mtah.frames import load_frames
from mtah.match import from_tactical
for p in sorted(glob.glob(os.path.join(DFL, '*positions*.xml'))): from_dfl(p.rsplit('_', 1)[-1][:-4])
for g in (1, 2, 3): from_metrica_sample(g)
from_tactical(); load_frames(); print('CACHES_OK')
"
echo BOOTSTRAP_DONE
