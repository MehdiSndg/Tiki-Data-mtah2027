#!/usr/bin/env bash
# Kill this project's running GPU job when Windows' free memory stays under LOW MB (default 1200) for two checks in a row,
# before Windows stops the whole WSL VM (which also kills every other project in it). Marks <queue>.guard so that
# local_queue.sh puts the job back at the head of the queue.
#
#   setsid nohup scripts/mem_guard.sh eda_out/queue.txt >> eda_out/logs/mem_guard.log 2>&1 < /dev/null &
q=$1
cd "$(dirname "$0")/.."
low=0
while true; do
    f=$(powershell.exe -NoProfile -Command "[math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1KB)" 2>/dev/null | tr -d '\r')
    if [ -n "$f" ] && [ "$f" -lt "${LOW:-1200}" ]; then low=$((low + 1)); else low=0; fi
    if [ $low -ge 2 ]; then
        pids=$(ps -eo pid=,args= | awk -v me=$$ '$1 != me && $2 ~ /Hackathon\/\.venv\/bin\/python/ && ($3 ~ /^analysis\// || $3 ~ /make_submission/) {print $1}')
        if [ -n "$pids" ]; then
            touch "$q.guard"
            echo "$(date +%F_%T) Windows free ${f} MB: killing $pids"
            kill $pids
            sleep 120
        fi
        low=0
    fi
    sleep 10
done
