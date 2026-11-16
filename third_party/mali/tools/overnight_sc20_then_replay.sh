#!/usr/bin/env bash
# One-off (5 Nov night, user-approved): after screen 1 -> all 20 SkillCorner games -> screen 2 -> clean replay.
cd "$(dirname "$0")/.." || exit 1
PY=.venv/Scripts/python.exe
log() { echo "$(date +%H:%M:%S) $*"; }
until grep -qE "SCREEN_EXIT|SC_CHECK_FAILED" outputs/overnight_sc_chain.log; do sleep 120; done
while ps -ef | grep -v grep | grep -q run_day_screen; do sleep 60; done
log "screen 1 chain done: $(tail -n 1 outputs/overnight_sc_chain.log)"
until $PY -c "import sys;sys.path.insert(0,'src');from mtah.io import skillcorner as s;sys.exit(0 if len(s.games())==20 else 1)"; do sleep 120; done
log "all 20 SkillCorner games downloaded; starting screen 2"
$PY tools/run_day_screen.py --job-dir outputs/day_screen/20261105_ext_met_sc20 --plan ext-met-sc20 --max-trial-minutes 180 --overnight-until 11:30
log "SCREEN2_EXIT $?"
while ps -ef | grep -v grep | grep -q run_day_screen; do sleep 60; done
mv outputs/clean_replay_ext outputs/clean_replay_ext_failed_0002 2>/dev/null
log "starting clean replay"
$PY tools/run_ext_replay.py --job-dir outputs/clean_replay_ext
log "REPLAY_EXIT $?"
