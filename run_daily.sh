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

echo "--- [0/5] 交易日曆（人事行政總處＋證交所假日表＋臨時休市記錄）"
$PY -c "from twq import calendar as C; C.ensure_fresh()" || echo "日曆更新失敗（沿用快取）"
$PY -m twq.calendar status "$TWDATE" || true
$PY -m twq.calendar retarget || true   # 目標日其實休市的舊預測改指向真正的下一交易日
echo "--- [1/5] 更新資料"
$PY -m twq update --end "$TWDATE" --days 10 || echo "更新發生錯誤（繼續執行，可能使用舊資料）"

echo "--- [1b/5] 季財報（MOPS 彙總報表）＋券商目標價新聞（即時累積）"
timeout 900 $PY -c "from twq import fetch_fin; fetch_fin.refresh()" || echo "季財報更新失敗（沿用舊資料）"
timeout 300 $PY -m twq.broker_news || echo "券商新聞收集失敗"

echo "--- [2/5] 檢討尚未檢討、且目標日已有資料的預測"
$PY -m twq review --no-update || echo "檢討失敗"

echo "--- [2b] 依檢討結果自動調權（twq tune，有防過擬合門檻）"
timeout 900 $PY -m twq tune || echo "自動調權失敗或逾時（沿用現行 config/weights.json）"
echo "--- [3/5] 預測下一交易日"
if $PY -m twq has-data --date "$TWDATE"; then
  $PY -m twq predict --date "$TWDATE" --no-update
else
  if $PY -m twq.calendar is-open "$TWDATE" >/dev/null 2>&1; then
    echo "${TWDATE} 行事曆為開市日但沒有交易資料（臨時休市如颱風假，或資料尚未公布），略過預測。若確定是臨時休市請執行：$PY -m twq.calendar add-closure ${TWDATE} 原因 來源"
  else
    echo "${TWDATE} 休市日（$($PY -m twq.calendar status "$TWDATE" | cut -c1-120)），略過預測。"
  fi
fi
echo "--- [3b] 灰色預測 GM(1,1) 影子模式（不影響名單；失敗不中斷）"
if $PY -m twq has-data --date "$TWDATE"; then
  timeout 300 $PY -m twq.grey all --date "$TWDATE" || echo "灰色預測失敗（略過）"
else
  timeout 300 $PY -m twq.grey evaluate || echo "灰色預測檢討失敗（略過）"
fi
echo "--- [4/5] 更新網站資料 docs/"
$PY -m twq site || echo "網站資料產生失敗"
echo "--- [5/5] 推送網站更新到 GitHub"
if [[ -n "${TWQ_NOPUSH:-}" ]]; then
  echo "TWQ_NOPUSH 已設定：略過 commit/push（dry-run）"
elif [[ -n "$(git status --porcelain docs reports models/validation.json models/grey_backtest.json config)" ]]; then
  git add docs reports models/validation.json models/grey_backtest.json config && git commit -q -m "daily update ${TWDATE}" && git push -q origin main && echo "已推送" || echo "推送失敗"
else
  echo "沒有變更，不推送"
fi
echo "完成 $(date '+%F %T %Z')"
