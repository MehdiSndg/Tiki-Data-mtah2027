#!/usr/bin/env bash
# One-off (5 Nov night): rerun the clean-clone replay after the SkillCorner screen chain has finished.
cd "$(dirname "$0")/.." || exit 1
until grep -qE "SCREEN_EXIT|SC_CHECK_FAILED" outputs/overnight_sc_chain.log; do sleep 120; done
while ps -ef | grep -v grep | grep -q run_day_screen; do sleep 60; done
echo "$(date +%H:%M:%S) chain done: $(tail -n 1 outputs/overnight_sc_chain.log); starting replay"
mv outputs/clean_replay_ext "outputs/clean_replay_ext_failed_0002" 2>/dev/null
.venv/Scripts/python.exe tools/run_ext_replay.py --job-dir outputs/clean_replay_ext
echo "$(date +%H:%M:%S) REPLAY_EXIT $?"
