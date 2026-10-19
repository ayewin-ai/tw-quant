#!/usr/bin/env bash
# 盤前更新：台指期夜盤(台灣 05:00 結束) + 美股收盤 + 隔夜事件 → 重算今天(台灣)的預測，更新網站並推送
# 建議排程：台灣 07:00～08:00 ＝ 波哥大 (UTC-5) 前一天 18:00～19:00
#   cron (box 時區 America/Bogota):  15 18 * * 0-4  /workspace/tw-quant/run_preopen.sh >> /workspace/tw-quant/logs/preopen.log 2>&1
set -uo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
echo "=================== $(date '+%F %T %Z') | 台灣 $(TZ=Asia/Taipei date '+%F %H:%M') 盤前更新 ==================="
$PY -m twq preopen ${1:+--target "$1"} || { echo "盤前更新失敗"; exit 1; }
$PY -m twq site || echo "網站資料產生失敗"
if [[ -n "$(git status --porcelain docs reports)" ]]; then
  git add docs reports && git commit -q -m "preopen update $(TZ=Asia/Taipei date +%F)" && git push -q origin main && echo "已推送" || echo "推送失敗"
fi
echo "完成 $(date '+%F %T %Z')"
