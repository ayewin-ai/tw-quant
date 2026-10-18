#!/usr/bin/env bash
# Resumable backfill supervisor: restarts the fetcher if it stalls (each run capped at 15 min).
cd "$(dirname "$0")"
START="${1:-2024-07-01}"; END="${2:-$(TZ=Asia/Taipei date +%F)}"
for i in $(seq 1 30); do
  timeout 900 .venv/bin/python -u -m twq backfill --start "$START" --end "$END"
  rc=$?
  [[ $rc -eq 0 ]] && { echo "backfill complete"; break; }
  echo "backfill run $i exited rc=$rc, restarting..."
done
