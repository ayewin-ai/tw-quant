# 台股量化判斷系統（tw-quant）

根據**免費公開資料**（證交所、櫃買中心、公開資訊觀測站、Yahoo 美股行情、Google News）每天收盤後預測
「下一個交易日哪些股票會跑贏」，隔天收盤後自動**檢討**：哪裡判斷錯、漏掉哪些因素，並把錯誤原因累積統計，
提示該加入哪些新特徵。

> ⚠️ **免責聲明**：本系統與其報告僅供研究與學習，**不構成任何投資建議**。短線預測的準確度本來就有限，
> 過去回測表現不代表未來，投資請自行判斷並承擔風險。

---

## 1. 資料來源（全部免費、公開）

| 資料 | 來源 | 頻率 |
|---|---|---|
| 上市個股日行情（開高低收、量、值、本益比）、加權指數 | 證交所 `exchangeReport/MI_INDEX` | 每日 |
| 上市三大法人買賣超 | 證交所 `fund/T86` | 每日 |
| 上市融資融券餘額 | 證交所 `exchangeReport/MI_MARGN` | 每日（約 21:00 公布） |
| 上櫃個股日行情 / 三大法人 / 融資融券 / 櫃買指數 | 櫃買中心 `www/zh-tw/afterTrading/dailyQuotes`、`insti/dailyTrade`、`margin/balance`、`indexInfo/inx` | 每日 |
| 除權除息（歷史與預告） | 證交所 `TWT49U`、`TWT48U`；櫃買 `bulletin/exDailyQ` | 每日 |
| 月營收 | 公開資訊觀測站彙總表 `mopsov.twse.com.tw/nas/t21/...`；最新月另用 openapi `t187ap05_L` / `mopsfin_t187ap05_O` | 每月 |
| 產業別 | 證交所 / 櫃買 openapi 公司基本資料 | 不定期 |
| 美股（費半、台積電 ADR、那斯達克、S&P500、VIX） | Yahoo Finance chart API | 每日 |
| 個股新聞標題（檢討用） | Google News RSS（zh-TW） | 檢討時查詢 |
| 休市日 | 證交所 `holidaySchedule` | 每年 |

所有請求都有間隔（證交所約 3.2 秒/次，櫃買 2 秒/次，觀測站 3 秒/次），避免被封鎖。
資料存於 SQLite：`data/tw.db`。

### 1b. 期貨、國際股市、事件風險（新增）
| 類別 | 來源 | 備註 |
|---|---|---|
| 台指期日/夜盤、基差 | 期交所 `futDataDown`（一般/盤後） | 2024-06 起；夜盤只給盤前模型 |
| 期貨三大法人未平倉 | 期交所 `futContractsDateDown` | 外資/投信淨 OI 與變化 |
| P/C 比 | 期交所 `pcRatioDown` | |
| 國際股市/總經 | Yahoo chart API：日經、KOSPI、恆生、上證(000001.SS，CSI300 抓不到)、DAX、STOXX50、ES/NQ 期貨、USD/TWD、US10Y、DXY、原油、黃金、VIX | 亞股用當日收盤；歐美用前一日（盤前模型用當日） |
| 地緣政治/突發事件 | GDELT 1.0 每日事件檔（doc API 對本機 IP 一直限流，改用事件檔） | 中台、戰爭、制裁關稅、緊急事件的報導占比與語調；只用 < t 的檔案 |
| 地震 | USGS fdsnws（台灣附近 M≥5） | 完整歷史 |
| 中文新聞量 | Google News RSS zh-TW 關鍵字 | 歷史每次上限 100 則，粗略；只用前一日 |
| 颱風 | 氣象署 API 拒絕存取 | **未使用**；只能靠新聞量與休市日間接反映 |

### 1c. ETF / 指數調整 / 法說會 / 科技大事件 / 政府基金（2026-10 新增）
| 類別 | 來源 | 歷史 | 備註 |
|---|---|---|---|
| ETF 持股 | MoneyDJ 全部持股頁（0050、006208、0056、00878、00919、00929、00940、00713、00881、00692） | **只有當日快照**，2026-10-19 起每日累積 | 沒有免費歷史持股，因此 `etf_w` 不放進回測 |
| 成分股調整 | Google News 標題解析（MSCI 季調、台灣50、0056、00878、00919、00929、00940、00713）＋規則生效日（MSCI 2/5/8/11 月最後交易日；台灣50 3/6/9/12 月第三個週五；其餘 ETF 近似） | 2024-06 起 | 標題解析有雜訊（同一檔同時判為納入/剔除者丟棄） |
| 法說會 | MOPS 法人說明會（t100sb02_1，上市＋上櫃，逐月） | 2024-05 起 | 只用 10 天內的未來法說日（通常已公告）；個股過去法說前後 5 日超額（事件研究，只用已發生事件） |
| 科技大事件 | 程式內建行事曆：台積電法說、輝達/蘋果財報、Apple 發表會、WWDC、GTC、Computex、CES、Tesla | 2024-07 起 | 2026 下半年部分日期為估計；各產業過去事件前後 5 日平均超額 |
| 國安基金/八大行庫/勞動基金 | Google News 每日新聞量（歷史與即時用同一方法） | 逐步回補 | 勞保/勞退/退撫基金只公布季/年度或委外資訊，八大行庫買賣超沒有官方日資料，只能用新聞量代理 |
| 預測封存 | `reports/archive/`：每日每檔預測（csv.gz）、隔日實際結果（跳空/盤中/收→收，按月分檔、只追加不覆寫）、`daily_comparison.csv` | 2026-10-15 起 | `python -m twq archive` |

## 2. 安裝

```bash
cd /workspace/tw-quant
python3 -m venv .venv
.venv/bin/pip install pandas numpy requests lightgbm scikit-learn feedparser lxml
```

## 3. 指令

```bash
PY=.venv/bin/python

$PY -m twq backfill --start 2024-07-01          # 回補歷史（約 1.5～2 小時，可背景執行、可中斷續跑）
$PY -m twq update                               # 補最近 10 天資料（台灣日期）
$PY -m twq status                               # 查看資料庫狀態
$PY -m twq validate                             # walk-forward 樣本外驗證 + Top-10 回測 → reports/validation.md
$PY -m twq predict --date 2026-10-16            # 用 10/16 收盤資料預測下一交易日
$PY -m twq review                               # 檢討所有「目標日已有資料、尚未檢討」的預測
$PY -m twq review --date 2026-10-15             # 指定檢討 10/15 產生的預測
./run_daily.sh                                  # 每日一鍵：更新 → 檢討 → 預測
```

### 盤前更新（`twq preopen`）
```bash
python -m twq preopen                 # 下一交易日，抓夜盤/美股/匯率/地震後重新評分
python -m twq preopen --limitup penalize   # flag(預設)/penalize/exclude 前日漲停股
python -m twq ablation --start 2025-03-01  # 特徵組比較，寫入 reports/validation.md
```
建議排程：台灣 07:15（夜盤 05:00 收、美股已收）＝ **波哥大前一日 18:15（週日～週四）**：
`15 18 * * 0-4 /workspace/tw-quant/run_preopen.sh`。
預測目標為「開盤→收盤」相對強弱，才是可實際執行的版本。

### 每日排程

台灣時間 21:30 之後執行最完整（行情約 14:30～16:30、法人約 16:30、融資融券約 21:00 公布）。
你在 America/Bogota（UTC-5），台灣 21:30＝你那邊 **08:30**。box 的時區是 America/Bogota，所以 cron：

```cron
30 8 * * 1-5  /workspace/tw-quant/run_daily.sh >> /workspace/tw-quant/logs/daily.log 2>&1
```

休市日（國定假日、颱風假）抓不到資料時會自動略過預測；檢討會自動對到「預測日之後第一個有資料的交易日」。

## 4. 模型

- **股票池**：上市櫃普通股（4 碼、非 0 開頭），20 日平均成交值 ≥ 5,000 萬、股價 ≥ 10 元、近 60 日至少 40 日有交易。
- **預測目標**：隔日報酬是否 **高於當日全體股票中位數**（純選股能力，不受大盤漲跌影響）。
  報告另外用「勝過加權指數」與「絕對上漲」來衡量。
- **報酬計算**：用交易所公布的「漲跌價差」（相對參考價），因此除權息日不會被誤判為大跌。
- **特徵（約 70 個）**
  - 技術：1/5/10/20/60 日報酬、均線乖離、RSI、波動率、K 棒型態、距高低點、漲跌停
  - 量能：量比、成交值 Z 分數、流動性
  - 籌碼：外資/投信/自營商 買賣超比率、5/20 日累積、連買連賣天數；融資融券變化、券資比
  - 大盤：加權/櫃買指數報酬與均線位置、上漲家數比例、外資全市場買賣超
  - 美股：前一晚費半、台積電 ADR、那斯達克、VIX
  - 產業：同產業當日/5 日平均報酬、相對產業強弱
  - 營收：最新月營收 YoY、MoM、YoY 變化、近 3 月平均（**假設每月 11 日後才可用**，避免偷看未來）
  - 評價：本益比（上市）
- **演算法**：LightGBM 二元分類。
- **驗證**：walk-forward（擴張視窗，每 21 個交易日重新訓練，只用當時已知資料），輸出 AUC、Rank IC、
  Top-10 命中率、十分位報酬，以及 Top-10 每日等權回測（含交易成本；同時提供「隔日開盤買、收盤賣」的可執行版本）。
- **每日預測**：用預測日之前所有已知標籤重新訓練（模型存在 `models/`），每檔股票用 SHAP（LightGBM `pred_contrib`）
  拆解出看多/看空的主要因素。

## 5. 每日檢討做什麼

對「Top 名單中下跌或輸給大盤的股票」與「漏掉的大漲股（不在 Top 名單、超額 > 3%）」逐檔分析：

1. **報酬拆解**：實際報酬 = 大盤 + 產業超額 + 個股特有。
2. **模型依據**：哪些特徵把機率推高（誤判）或壓低（漏掉）。
3. **漏掉的因素檢查**
   - 大盤系統性漲跌
   - 預測後當晚的美股（費半、台積電 ADR、那指）——模型在預測時看不到
   - 產業輪動（同產業一起漲跌）
   - 除權息、月營收公布期間
   - 新聞事件（Google News 標題關鍵字：法說會、營收、財報、訂單、併購、處置/注意股、目標價、利空/利多…）
   - 追高反轉（模型因動能看多但前一天已大漲）、籌碼反轉（看多主因是法人，但隔天法人轉賣）
   - 漲停題材、隔日法人同步大買
4. 累積寫入：
   - `reports/review_log.csv`：每一筆錯誤/遺漏的原因、主要特徵、新聞標籤
   - `reports/performance_log.csv`：每天的命中率、AUC、Rank IC、市場狀態
   - `reports/error_patterns.md`：原因類別累積次數 + **建議新增的特徵**（例如美股隔夜最常出現 → 加開盤前更新）

## 6. 檔案結構

```
twq/fetch.py      資料抓取（證交所、櫃買、觀測站、Yahoo）
twq/features.py   特徵工程、交易日判斷
twq/model.py      LightGBM、walk-forward 驗證、回測
twq/predict.py    每日預測 + 因素拆解 + Markdown 報告
twq/review.py     隔日檢討 + 根因分析 + 累積統計
twq/news.py       Google News RSS 新聞標題與關鍵字標籤
data/tw.db                    SQLite 資料庫
data/predictions/<日期>.json  每日完整預測（全部股票的機率、排名、因素）
reports/predict/predict_<日期>.md
reports/review/review_<預測日>.md
reports/validation.md         樣本外驗證報告
run_daily.sh                  每日一鍵流程
```

## 7. 限制（請務必了解）

- 預測是在**收盤後**做的，「收盤→隔日收盤」回測假設能用預測日收盤價買到，實務上做不到；請以「隔日開盤→收盤」版本為準。
- 預測產生後當晚美股的變化模型看不到（檢討會標示）；如需要可在台灣開盤前再跑一次預測（之後可加入盤前更新）。
- 未納入：期貨/選擇權、券商分點、借券、重大訊息公告全文、法說會行事曆、財報細項。這些都是檢討報告會建議加入的方向。
- 產業別使用目前的分類，歷史上若有變更會有些微誤差；下市股票的歷史仍在資料中，但產業可能顯示「未知」。
- 新聞判讀只用標題關鍵字，可能誤判；檢討報告會附上標題讓你自行判斷。
- 交易成本以手續費 6 折、證交稅 0.3% 估算，未計滑價與漲停買不到。

- 新增的期貨/國際/事件特徵是否有用，請看 `reports/validation.md` 的 Ablation 表；結果誠實列出，沒有勝過大盤就是沒有。
- 事件資料是粗略代理：GDELT 報導量與 Google News 計數不等於事件嚴重度，新聞量歷史不完整。

## 8. 網站（GitHub Pages）

`docs/` 是純靜態網站（Chart.js / marked 由 CDN 載入，不需要伺服器），說明演算法、資料取得方式，並畫出
樣本外回測淨值曲線、十分位報酬、命中率、累積錯誤原因，以及最新一天的預測與檢討。

```bash
.venv/bin/python -m twq site            # 由 DB / models / reports 重建 docs/data/*.json 並複製報告到 docs/reports/
cd docs && python3 -m http.server 8000  # 本機預覽 http://localhost:8000
```

`run_daily.sh` 最後一步會自動執行 `twq site`，commit + push 後網站就會更新。
GitHub 設定：Settings → Pages → Source 選 **Deploy from a branch**，Branch 選 `main`、資料夾 `/docs`。
