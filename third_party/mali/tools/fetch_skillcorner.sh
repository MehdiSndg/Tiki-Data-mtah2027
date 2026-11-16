#!/usr/bin/env bash
# SkillCorner open data (MIT, A-League 2024/25): resumable download into cache/skillcorner/<id>/.
# Tracking files are Git LFS (media.githubusercontent.com); the line is slow (~70 KB/s total), so 4 at a time.
cd "$(dirname "$0")/../cache/skillcorner" || exit 1
RAW=https://raw.githubusercontent.com/SkillCorner/opendata/master/data
MEDIA=https://media.githubusercontent.com/media/SkillCorner/opendata/master/data
[ -s matches.json ] || curl -sL --retry 5 -o matches.json $RAW/matches.json
IDS=$(../../.venv/Scripts/python.exe -c "import json;print(' '.join(str(m['id']) for m in json.load(open('matches.json'))))")
one() {
  m=$1; mkdir -p $m; cd $m
  for f in match.json phases_of_play.csv dynamic_events.csv; do
    [ -s ${m}_$f ] || curl -sL --retry 5 -o ${m}_$f $RAW/matches/$m/${m}_$f
  done
  for i in 1 2 3 4 5 6 7 8; do
    curl -sL --retry 5 -C - -o ${m}_tracking_extrapolated.jsonl $MEDIA/matches/$m/${m}_tracking_extrapolated.jsonl && break
    sleep 10
  done
  echo "$(date +%H:%M) $m $(du -b ${m}_tracking_extrapolated.jsonl | cut -f1)"
}
export -f one; export RAW MEDIA
printf "%s\n" $IDS | xargs -P 4 -I{} bash -c 'one {}'
echo "ALL_DONE $(date +%H:%M)"
