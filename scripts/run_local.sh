#!/usr/bin/env bash
# Run one heavy local job (training on the RTX 4060) inside a memory cap, detached, with a log.
# WSL restarted twice on 29-30 Oct during local jobs: WSL keeps the files a job reads in its page cache and Windows
# ran out of memory. The cap (cgroup v2 via systemd, page cache included) keeps the job and its cache under MEM, for
# this project's jobs only; the global WSL settings stay as they are.
#
#   scripts/run_local.sh <log> <command...>        # MEM=12G by default
#   MEM=10G scripts/run_local.sh eda_out/logs/x.log uv run python analysis/x.py
set -euo pipefail
log=$1
shift
mkdir -p "$(dirname "$log")"
setsid nohup systemd-run --user --scope -q -p MemoryMax="${MEM:-12G}" -p MemorySwapMax=0 nice -n 10 "$@" > "$log" 2>&1 < /dev/null &
echo "started pid $! -> $log"
