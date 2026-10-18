#!/usr/bin/env bash
# 每日流程：更新資料 → 檢討前一交易日的預測 → 預測下一交易日
# 建議排程：台灣時間 21:30 以後（融資融券約 21:00 才公布）＝ 美東/波哥大時間 08:30 (UTC-5)
#   cron (box 時區 America/Bogota):  30 8 * * 1-5  /workspace/tw-quant/run_daily.sh >> /workspace/tw-quant/logs/daily.log 2>&1
# 用法：./run_daily.sh [YYYY-MM-DD(台灣日期, 預設今天)]
set -uo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
TWDATE="${1:-$(TZ=Asia/Taipei date +%F)}"
TWHOUR=$(TZ=Asia/Taipei date +%H)
echo "=================== $(date '+%F %T %Z') | 台灣日期 ${TWDATE} (台灣現在 ${TWHOUR} 點) ==================="
if [[ -z "${1:-}" && "$TWHOUR" -lt 15 ]]; then
  echo "提醒：台灣時間尚未 15:00，當日收盤資料可能還沒公布。"
fi

echo "--- [1/4] 更新資料"
$PY -m twq update --end "$TWDATE" --days 10 || echo "更新發生錯誤（繼續執行，可能使用舊資料）"

echo "--- [2/4] 檢討尚未檢討、且目標日已有資料的預測"
$PY -m twq review --no-update || echo "檢討失敗"

echo "--- [3/4] 預測下一交易日"
if $PY -m twq has-data --date "$TWDATE"; then
  $PY -m twq predict --date "$TWDATE" --no-update
else
  echo "${TWDATE} 沒有交易資料（休市日、颱風假或資料尚未公布），略過預測。"
fi
echo "--- [4/4] 更新網站資料 docs/"
$PY -m twq site || echo "網站資料產生失敗"
echo "完成 $(date '+%F %T %Z')"
