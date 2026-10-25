#!/usr/bin/env bash
# 盤前更新：台指期夜盤(台灣 05:00 結束) + 美股收盤 + 隔夜事件 → 重算今天(台灣)的預測，更新網站並推送
# 建議排程：台灣 07:00～08:00 ＝ 波哥大 (UTC-5) 前一天 18:00～19:00
#   cron (box 時區 America/Bogota):  15 18 * * 0-4  /workspace/tw-quant/run_preopen.sh >> /workspace/tw-quant/logs/preopen.log 2>&1
set -uo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
echo "=================== $(date '+%F %T %Z') | 台灣 $(TZ=Asia/Taipei date '+%F %H:%M') 盤前更新 ==================="
TPDATE="${1:-$(TZ=Asia/Taipei date +%F)}"
$PY -c "from twq import calendar as C; C.ensure_fresh()" || echo "日曆更新失敗（沿用快取）"
$PY -m twq.calendar retarget || true
if ! $PY -m twq.calendar is-open "$TPDATE" >/dev/null 2>&1; then
  echo "台灣 ${TPDATE} 休市：$($PY -m twq.calendar status "$TPDATE")；不產生盤前預測。下一交易日：$($PY -m twq.calendar next "$TPDATE")"
  $PY -m twq site || echo "網站資料產生失敗"
  if [[ -z "${TWQ_NOPUSH:-}" && -n "$(git status --porcelain docs reports config)" ]]; then
    git add docs reports config && git commit -q -m "calendar/site update $(TZ=Asia/Taipei date +%F) (休市日)" && git push -q origin main && echo "已推送" || echo "推送失敗"
  fi
  exit 0
fi
$PY -m twq preopen ${1:+--target "$1"} || { echo "盤前更新失敗"; exit 1; }
$PY -m twq site || echo "網站資料產生失敗"
if [[ -n "${TWQ_NOPUSH:-}" ]]; then
  echo "TWQ_NOPUSH 已設定：略過 commit/push"
elif [[ -n "$(git status --porcelain docs reports config)" ]]; then
  git add docs reports config && git commit -q -m "preopen update $(TZ=Asia/Taipei date +%F)" && git push -q origin main && echo "已推送" || echo "推送失敗"
fi
echo "完成 $(date '+%F %T %Z')"
