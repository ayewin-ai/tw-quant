#!/usr/bin/env bash
# Resumable backfill of extra data (TAIFEX, world markets, quakes, GDELT, zh-TW news counts) since 2024-06.
cd "$(dirname "$0")"
PY=.venv/bin/python
START="${1:-2024-06-01}"; END="${2:-$(TZ=Asia/Taipei date +%F)}"
$PY -u -c "
import datetime as dt
from twq import fetch_ext as E
s,e=dt.date.fromisoformat('$START'),dt.date.fromisoformat('$END')
E.init()
print('tx',E.taifex_tx(s,e),flush=True); print('inst',E.taifex_inst(s,e),flush=True); print('pcr',E.taifex_pcr(s,e),flush=True)
print('world',E.world('3y'),flush=True); print('quakes',E.quakes(s,e),flush=True)
"
$PY -u -c "
import datetime as dt
from twq import fetch_ext as E
E.gdelt(dt.date.fromisoformat('$START'),dt.date.fromisoformat('$END'))
" &
$PY -u -c "
import datetime as dt
from twq import fetch_ext as E
E.news_counts_hist(dt.date.fromisoformat('$START'),dt.date.fromisoformat('$END'))
"
wait
echo "ext backfill done"
