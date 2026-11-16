#!/usr/bin/env bash
# Stops our newest local python job when Windows' free memory falls under LIMIT_MB (WSL is shut down by Windows when it
# runs out, 5-7 Nov). Polls every POLL_S seconds; logs to the file given as $1. After stopping a job it waits COOLDOWN_S
# before it may stop another: WSL hands freed memory back to Windows slowly, so the next polls would otherwise see the
# old number and stop every job in turn (review 13 Nov).
#   MATCH        regex on a job's executable path (default: our two venvs' python)
#   FREE_MB_CMD  command printing Windows' free memory in MB (tests replace it)
#   EVICT_CMD    run first when memory is low (default: drop our data files from the page cache, which WSL holds as Windows
#                memory); a job is stopped only if Windows is still low EVICT_WAIT_S later (14 Nov)
LIMIT_MB=${LIMIT_MB:-2500}
POLL_S=${POLL_S:-20}
COOLDOWN_S=${COOLDOWN_S:-180}
MATCH=${MATCH:-'\.venv/bin/python|venvs/mali/bin/python'}
FREE_MB_CMD=${FREE_MB_CMD:-'timeout 15 powershell.exe -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1KB)"'}
EVICT_CMD=${EVICT_CMD:-"python3 $(dirname "$0")/evict_cache.py /home/mehdi/Hackathon/data /home/mehdi/Hackathon/cache /home/mehdi/Hackathon/models /home/mehdi/mali_work /home/mehdi/Hackathon/.venv /home/mehdi/venvs /home/mehdi/.cache /tmp/claude-1000"}
EVICT_WAIT_S=${EVICT_WAIT_S:-30}
LOG=${1:-/tmp/win_guard.log}
last_kill=0
while true; do
  free_mb=$(bash -c "$FREE_MB_CMD" 2>/dev/null | tr -d '\r')
  now=$(date +%s)
  if [[ "$free_mb" =~ ^[0-9]+$ ]] && [ "$free_mb" -lt "$LIMIT_MB" ] && [ $((now - last_kill)) -ge "$COOLDOWN_S" ]; then
    bash -c "$EVICT_CMD" > /dev/null 2>&1
    sleep "$EVICT_WAIT_S"
    free_mb=$(bash -c "$FREE_MB_CMD" 2>/dev/null | tr -d '\r')
    echo "$(date -u +%FT%T) page cache evicted; Windows free now ${free_mb} MB" >> "$LOG"
  fi
  if [[ "$free_mb" =~ ^[0-9]+$ ]] && [ "$free_mb" -lt "$LIMIT_MB" ] && [ $((now - last_kill)) -ge "$COOLDOWN_S" ]; then
    pid=$(ps -eo pid,comm,args --sort=start_time | MATCH="$MATCH" awk '$2 ~ /^python/ && $3 ~ ENVIRON["MATCH"] {p=$1} END {print p}')
    if [ -n "$pid" ]; then
      echo "$(date -u +%FT%T) Windows free ${free_mb} MB < ${LIMIT_MB}: stopping newest job $pid: $(ps -o args= -p $pid | cut -c1-120)" >> "$LOG"
      kill "$pid"
      last_kill=$now
    fi
  fi
  sleep "$POLL_S"
done
