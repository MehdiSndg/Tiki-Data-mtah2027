#!/usr/bin/env bash
# From the provided data bundle to the submission file (v28), step by step. Status on 14 Nov: steps 5-6 are verified
# byte for byte on this machine (v28 rebuilt after the 14 Nov refactor: cmp identical); steps 1-4 list what produces each
# input and are NOT yet re-run from an empty checkout. Lines marked [WEIGHTS] need a released weight file, because training is not
# bit-reproducible across machines; the training command that produced it is given beside it.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python

# 0. Inputs
#   data/tactical/, data/broadcast/         the organisers' bundle (not redistributed)
#   data/external/dfl/                      scripts/fetch_dfl.py      (DFL / IDSSE open data, CC BY 4.0)
#   data/external/metrica-sample-data/      scripts/fetch_metrica.sh  (Metrica sample data)
#   third_party/sn-teamspotting             git clone https://github.com/SoccerNet/sn-teamspotting (T-DEED code)
#   models/tdeed/checkpoint_best.pt         [WEIGHTS] SoccerNet's released T-DEED checkpoint
#   models/p9ci/{A,B,F,A3,F2,F3,F4}.pt      [WEIGHTS] ours: T-DEED + dense head fine-tuned on SoccerTrack v2 only
#                                           (analysis/p9bx_st_vcam_colab.py frames, analysis/p9ci_tdeed_dense_colab.py train;
#                                           A / A3: matches 117092 117093 118575 118576; B: 118577 118578 128057 128058 132877;
#                                           F / F2 / F3 / F4: all ten, seeds differ; 6 epochs, bs 2, lr 4e-4)
#   models/*_ea44a36fa4aceb67.pt, models/dead_gru_d012_m123x_d728296f74d7bcff.pt
#                                           [WEIGHTS] ours: team / boundary / dead-ball nets trained on degraded DFL + Metrica
#                                           (mtah.grumodel.external_gru, analysis/p4_common.dead_gru; built by scripts/kaggle/)
#   cache/team_in/mehmet_ali_topkara/barath/components.npz  [WEIGHTS] Mehmet Ali Topkara's delivered probabilities for our match
#   All [WEIGHTS] files: bash scripts/fetch_weights.sh <owner/repo>   (this repository's release v28, sha256 checked)

# 1. Caches of the provided match (no model, no label): tracking table, frame table, broadcast -> tactical frame map
$PY analysis/p3_yolo_pass.py                                      # -> cache/yolo/ (pretrained YOLO11m on COCO, inference only: persons for the keeper-slot check)
$PY -c "import sys; sys.path[:0]=['.','evaluation']; from mtah.data import load_tactical; from mtah.frames import load_frames; load_tactical(); load_frames()"
$PY analysis/h_broadcast_sync.py                                   # -> cache/broadcast_on_tactical.npz

# 2. Video channel: virtual-camera frames of the tactical video, then the five dense models (inference only)
$PY analysis/p9cc_barath_vcam_plan.py                              # -> cache/barath_vcam_plan.npz (windows around the ATD ball)
$PY analysis/p9cl_barath_vcam_extract.py                           # -> cache/barath_vcam/ (5.1 GB of JPEG shards)
for k in A B F A3 F2 F3 F4; do
  mkdir -p cache/barath_$k
  SNTS=third_party/sn-teamspotting TDEED_CKPT=models/tdeed/checkpoint_best.pt \
    $PY analysis/p9cm_barath_dense_infer.py models/p9ci/$k.pt cache/barath_$k/barath_dense.npz
done
$PY analysis/p9dn_barath_dense_avg.py A B F A3 F2                  # -> cache/barath_dense_avg5.npz (v26, v27a)
$PY analysis/p9dn_barath_dense_avg.py A B F A3 F2 F3 F4            # -> cache/barath_dense_avg7.npz (v28, v29)

# 3. Mali's member: his pipeline in its own environment (third_party/mali, kurulum/requirements-gpu.txt)
#   python run.py m0 --config main_ext_met && python run.py m0 --config ev_ext_mb
#   python tools/bar_from_nets.py --run main_ext_met               # -> probabilities for our match
#   python tools/team_export.py --out <dir>                        # -> barath/components.npz (p_home_given_alive, p_dead, p_boundary)
#   v26-v28 read his delivered cache/team_in/mehmet_ali_topkara/barath/components.npz. v29 reads the same recipe rebuilt by us
#   (his code, nets trained on Colab with MTAH_NO_PUBLIC=1, booster rebuilt here):
#   MTAH_DATA_DIR=data MTAH_NO_PUBLIC=1 <his venv>/python scripts/mali_components.py third_party/mali cache/team_in/mehmet_ali_topkara/barath_rebuilt

# 4. Our chain's models are loaded from models/ (cache key ea44a36fa4aceb67 = the sources in mtah.grumodel.HASH_FILES);
#    the dead-ball classifier (HistGradientBoosting, fixed seed) is refitted inside step 5 from data/external; v27a-v29
#    fit it on the production + realistic-dead copies (mtah/deadsim.py). v29 / v30 need the six recipe-C members
#    models/team_gruC*_340546b95641c360.pt (v30: seeds 0-5 and 6-11) [WEIGHTS] (mtah.members_c.external_gru_c, trained on Colab; the build refuses to
#    train them locally). The build peaks at ~7.4 GB RAM (v28: ~9.2 GB): run it alone.

# 5. The submission
$PY scripts/make_submission.py v28 evaluation/results/Tiki-Data.csv

# 6. Check
$PY evaluation/validate_submission.py evaluation/results/Tiki-Data.csv
sha256sum evaluation/results/Tiki-Data.csv                         # v28: ca3077cdc2aef58092451c6967926ba1e5aea43e43482b3998a15a800d94ca93 (rebuilt from a clean clone 17 Nov); v30 on 15 Nov: 6ade417eb27568ac97fdba1ee463018e22d6c300d7bde5552c7dddd1aea73be2; v29 on 14 Nov: 91edea7f72bc740dc9b309074b5d658201a1dc6d3d3111a1faa972f5206232b1 (v28: ca3077cdc2aef58092451c6967926ba1e5aea43e43482b3998a15a800d94ca93)
