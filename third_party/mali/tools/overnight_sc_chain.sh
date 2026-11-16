#!/usr/bin/env bash
# One-off overnight chain (user-approved 4-5 Nov night): clean replay -> 8 SkillCorner games -> CPU check -> screen.
cd "$(dirname "$0")/.." || exit 1
PY=.venv/Scripts/python.exe
log() { echo "$(date +%H:%M:%S) $*"; }
while ps -ef | grep -v grep | grep -q run_ext_replay; do sleep 60; done
log "replay finished: $(tail -n 1 outputs/clean_replay_ext.log)"
GAMES="2017461 2016236 2015213 2013725 2011166 2010085 2007721 2007448"
until $PY -c "import sys;sys.path.insert(0,'src');from mtah.io import skillcorner as s;g=set(s.games());sys.exit(0 if set('$GAMES'.split())<=g else 1)"; do sleep 120; done
log "8 SkillCorner games downloaded"
PYTHONIOENCODING=utf-8 $PY - <<'PYEOF' || { log "SC_CHECK_FAILED"; exit 1; }
import sys; sys.path.insert(0, "src")
import numpy as np
from mtah.io import skillcorner as s
from mtah.constants import H, A, D
for g in "2017461 2016236 2015213 2013725 2011166 2010085 2007721 2007448".split():
    d = s.cached_game(g)
    p = d.period > 0
    lab = d.labels[p]
    runs = np.flatnonzero(np.diff(lab) != 0).size
    print(g, "frames", d.n_frames, "played min", round(p.sum() / 25 / 60, 1), "D", round(np.mean(lab == D), 3),
          "H", round(np.mean(lab == H), 3), "ball vis", round(np.mean(np.isfinite(d.ball[p, 0])), 3), "label changes", runs, flush=True)
    assert 0.3 < np.mean(lab == D) < 0.6 and 80 < p.sum() / 25 / 60 < 110
print("SC_CHECK_OK")
PYEOF
log "starting screen"
$PY tools/run_day_screen.py --job-dir outputs/day_screen/20261105_ext_met_sc --plan ext-met-sc --max-trial-minutes 180 --overnight-until 09:30
log "SCREEN_EXIT $?"
