#!/usr/bin/env bash
# Keep the local GPU busy with one job at a time: run the first line of the queue file, drop it, repeat; wait while the
# queue is empty. Each line is "<log> <command...>" and runs in a memory cap (MEM, default 8G). Stop with: touch <queue>.stop
#
#   setsid nohup scripts/local_queue.sh eda_out/queue.txt >> eda_out/logs/local_queue.log 2>&1 < /dev/null &
#
# Windows stops the whole WSL VM when its own free memory nears ~1-2 GB (5-7 Nov, several crashes). So a job starts only
# when Windows has MINFREE MB free (default 4500), and scripts/mem_guard.sh kills a running job when Windows runs short;
# a job killed by the guard (<queue>.guard present) goes back to the head of the queue.
set -uo pipefail
q=$1
cd "$(dirname "$0")/.."
touch "$q"
win_free() { powershell.exe -NoProfile -Command "[math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1KB)" 2>/dev/null | tr -d '\r'; }
while [ ! -e "$q.stop" ]; do
    line=$(head -n 1 "$q")
    if [ -z "$line" ]; then sleep 60; continue; fi
    f=$(win_free)
    if [ -n "$f" ] && [ "$f" -lt "${MINFREE:-4500}" ]; then echo "$(date +%F_%T) waiting: Windows free ${f} MB"; sleep 120; continue; fi
    sed -i 1d "$q"
    log=${line%% *}
    cmd=${line#* }
    mkdir -p "$(dirname "$log")"
    echo "$(date +%F_%T) start $log: $cmd"
    systemd-run --user --scope -q -p MemoryHigh="${MEMHIGH:-infinity}" -p MemoryMax="${MEM:-8G}" -p MemorySwapMax="${SWAP:-0}" nice -n 10 bash -c "$cmd" > "$log" 2>&1 < /dev/null
    rc=$?
    if [ $rc -ne 0 ] && grep -q "CUDA out of memory" "$log"; then    # transient right after a WSL restart (5 Nov): one retry
        echo "$(date +%F_%T) CUDA OOM, retrying once in 120 s: $log"
        sleep 120
        systemd-run --user --scope -q -p MemoryHigh="${MEMHIGH:-infinity}" -p MemoryMax="${MEM:-8G}" -p MemorySwapMax="${SWAP:-0}" nice -n 10 bash -c "$cmd" > "$log" 2>&1 < /dev/null
        rc=$?
    fi
    sync                                   # outputs on disk before the next job: a WSL stop drops unwritten pages (5 Nov)
    echo "$(date +%F_%T) end rc=$rc $log"
    if [ -e "$q.guard" ]; then
        rm -f "$q.guard"
        { printf '%s\n' "$line"; cat "$q"; } > "$q.tmp" && mv "$q.tmp" "$q"; sync
        echo "$(date +%F_%T) killed by mem_guard, requeued: $log"
        sleep 300
    fi
    sleep 30
done
