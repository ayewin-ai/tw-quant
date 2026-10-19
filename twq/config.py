from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
import os
DB_PATH = Path(os.environ.get("TWQ_DB", str(DATA / "tw.db")))
PRED_DIR = DATA / "predictions"
REPORT_DIR = ROOT / "reports"
PREDICT_REPORT_DIR = REPORT_DIR / "predict"
REVIEW_REPORT_DIR = REPORT_DIR / "review"
MODEL_DIR = ROOT / "models"
LOG_DIR = ROOT / "logs"

for _d in (DATA, PRED_DIR, PREDICT_REPORT_DIR, REVIEW_REPORT_DIR, MODEL_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Polite delays between requests (seconds). TWSE blocks IPs that exceed ~3 req / 5 s.
TWSE_SLEEP = 5.0
TPEX_SLEEP = 2.0
MOPS_SLEEP = 3.0

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"

# Universe / liquidity filter
MIN_AVG_VALUE = 50_000_000      # 20 日平均成交金額 >= 5,000 萬元
MIN_PRICE = 10.0
TOP_N = 10

US_SYMBOLS = {"^SOX": "sox", "TSM": "tsm_adr", "^GSPC": "spx", "^IXIC": "nasdaq", "^VIX": "vix"}

DISCLAIMER = ("⚠️ 免責聲明：本報告由程式依公開資料自動產生，僅供研究與學習用途，"
              "不構成任何投資建議。模型預測準確度有限，過去表現不代表未來結果，投資請自行判斷並承擔風險。")

# Feature set / label used by the daily (post-close) model. Chosen from the ablation (see reports/validation.md).
PREDICT_SET = os.environ.get("TWQ_SET", "base")
PREDICT_LABEL = os.environ.get("TWQ_LABEL", "y")
# Prior-day limit-up stocks: "flag" (mark only), "penalize" (-0.05 score), "exclude" (drop from top list)
LIMITUP_MODE = os.environ.get("TWQ_LIMITUP", "flag")
