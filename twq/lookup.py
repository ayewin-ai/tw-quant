"""個股隔日收盤價「預測查詢」資料產生器 → docs/data/lookup.json

對最新交易日的所有上市櫃股票，分別用系統內各演算法估計「下一交易日收盤價」。
機率/排序型模型(LightGBM 收盤模型、盤前模型)本身不輸出價格，換算方式：
  依當日機率在全市場的百分位，查歷史樣本外(walk-forward OOS)同一百分位十分組的
  平均實現報酬，預測收盤 = 最新收盤 × (1 + 該組平均報酬)。
所有數字皆為模型預測，非實際價格，不構成投資建議。
CLI: python -m twq.lookup
"""
import datetime as dt
import gzip
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DB_PATH, MODEL_DIR, REPORT_DIR, ROOT, MIN_AVG_VALUE, MIN_PRICE
from .grey import gm11_next, WIN as GWIN

OUT = ROOT / "docs" / "data" / "lookup.json"
PRED = REPORT_DIR / "archive" / "predictions"
NB = 10
LIN_WIN = 10
BT_DAYS = 250


def _lin_next(X):
    n = X.shape[1]; t = np.arange(n) - (n - 1) / 2
    slope = (X * t).sum(1) / (t * t).sum()
    return X.mean(1) + slope * (n - (n - 1) / 2)


def _algos_from_matrix(P):
    """P: (m, n) 收盤價矩陣（最後一欄=最新）。回傳 dict name -> 預測收盤 (m,)"""
    last = P[:, -1]
    out = {"unchanged": last.copy()}
    prev = P[:, -2]
    out["momentum1"] = last * (last / prev)
    out["ma5"] = np.nanmean(P[:, -5:], axis=1)
    out["linear10"] = _lin_next(P[:, -LIN_WIN:])
    g = np.full(len(last), np.nan)
    ok = np.isfinite(P[:, -GWIN:]).all(1) & (P[:, -GWIN:] > 0).all(1)
    if ok.any():
        g[ok] = gm11_next(P[ok][:, -GWIN:])[0]
    out["grey"] = g
    return out


def _stat(pred_ret, act_ret):
    m = np.isfinite(pred_ret) & np.isfinite(act_ret)
    p, a = pred_ret[m], act_ret[m]
    nz = (np.sign(p) != 0) & (np.sign(a) != 0)
    return {"mae": float(np.mean(np.abs(p - a))) if m.any() else None,
            "dir_hit": float(np.mean(np.sign(p[nz]) == np.sign(a[nz]))) if nz.any() else None,
            "n": int(m.sum())}


def _calib(prob_pct, ret):
    b = np.minimum((prob_pct * NB).astype(int), NB - 1)
    return pd.Series(ret).groupby(b).mean().reindex(range(NB)).fillna(0).values


STALE_DAYS = 20        # 最後成交日距資料日超過 N 個交易日的代號視為已下市/暫停交易，不列入選單
NEED = {"unchanged": 1, "momentum1": 2, "ma5": 5, "linear10": LIN_WIN, "grey": GWIN}
ESB_CACHE = ROOT / "config" / "emerging_latest.json"


def _emerging():
    """興櫃(TPEx emerging) 最新行情：只有當日價，沒有歷史序列。抓不到就用上次快取。"""
    import requests
    rows = None
    try:
        r = requests.get("https://www.tpex.org.tw/openapi/v1/tpex_esb_latest_statistics", timeout=25,
                         headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        rows = r.json()
        ESB_CACHE.write_text(json.dumps(rows, ensure_ascii=False))
    except Exception as e:
        print("emerging fetch failed:", e)
        if ESB_CACHE.exists():
            rows = json.loads(ESB_CACHE.read_text())
    out = {}
    for x in rows or []:
        try:
            code = x["SecuritiesCompanyCode"].strip()
            px = float(x.get("LatestPrice") or 0) or float(x.get("PreviousAveragePrice") or 0)
            if px > 0:
                d = str(x.get("Date") or "")
            if len(d) == 7 and d.isdigit():
                d = f"{int(d[:3]) + 1911}-{d[3:5]}-{d[5:]}"
            out[code] = (x["CompanyName"].strip(), px, d)
        except Exception:
            continue
    return out


def _lgbm_reason(code, px_rows, r_last, dates):
    """為什麼 LightGBM 主模型沒有評分這檔：逐項檢查與 features.build 相同的篩選條件。"""
    import re
    if not re.fullmatch(r"[1-9]\d{3}", code):
        return "非一般普通股（ETF／債券ETF／ETN／特別股／TDR 等）不在 LightGBM 訓練範圍，只提供價格型演算法"
    why = []
    if r_last["date"] != dates[-1] or not (r_last["volume"] or 0) > 0:
        why.append("資料日無成交")
    if (r_last["close"] or 0) < MIN_PRICE:
        why.append(f"股價 {r_last['close']:g} < {MIN_PRICE:g} 元")
    v = px_rows["value"].tail(20)
    av = float(v.fillna(0).sum() / 20)
    if av < MIN_AVG_VALUE:
        why.append(f"20 日均成交值 {av/1e4:,.0f} 萬 < {MIN_AVG_VALUE/1e4:,.0f} 萬")
    h = int(((px_rows["volume"] > 0) & px_rows["close"].notna()).tail(60).sum())
    if h < 40:
        why.append(f"近 60 日只有 {h} 日有成交（<40，新上市或常無成交）")
    return "未通過流動性/歷史篩選：" + "；".join(why) if why else "符合條件，但最新預測檔未包含此檔"


def build():
    con = sqlite3.connect(DB_PATH)
    dates = [r[0] for r in con.execute("select distinct date from prices order by date desc limit ?", (BT_DAYS + 15,))][::-1]
    pxa = pd.read_sql_query("select date, code, name, market, close, volume, value from prices where date >= ?",
                            con, params=(dates[0],))
    con.close()
    px = pxa[pxa.close > 0]
    last_date = dates[-1]
    C = px.pivot(index="date", columns="code", values="close").reindex(dates)
    # 選單 = 近 STALE_DAYS 個交易日內至少有一筆收盤價的所有代號（含 ETF/ETN/特別股/上櫃）
    recent = set(px[px.date >= dates[-STALE_DAYS]].code)
    latest_meta = px.sort_values("date").groupby("code").tail(1).set_index("code")
    info = latest_meta.loc[sorted(recent)][["name", "market"]]
    # 太久沒成交（已下市等）：只記名稱與最後成交日，用於搜尋時提示
    gone = {c: [r.name, r.market, r.date] for c, r in latest_meta.iterrows() if c not in recent}
    codes = [c for c in info.index if C[c].iloc[-LIN_WIN:].notna().all()]   # 完整 10 日：用於回測統計與主要預測

    # ---- 回測：價格型演算法（近 BT_DAYS 日，全部股票）
    acc = {k: {"mae": [], "hit": [], "n": 0} for k in ["unchanged", "momentum1", "ma5", "linear10", "grey"]}
    A = C.values
    pr_all, ac_all = {k: [] for k in acc}, []
    for i in range(LIN_WIN, len(dates) - 1):
        W = A[i - LIN_WIN + 1:i + 1].T
        nxt = A[i + 1]
        ok = np.isfinite(W).all(1) & np.isfinite(nxt)
        if not ok.any():
            continue
        W, nxt = W[ok], nxt[ok]
        act = nxt / W[:, -1] - 1
        keep = np.abs(act) < 0.105  # 排除除權息等異常
        ac_all.append(act[keep])
        for k, v in _algos_from_matrix(W).items():
            pr_all[k].append((v / W[:, -1] - 1)[keep])
    ac = np.concatenate(ac_all)
    stats = {k: _stat(np.concatenate(v), ac) for k, v in pr_all.items()}
    bt_range = [dates[LIN_WIN], dates[-1]]

    # ---- LightGBM 收盤模型：OOS 機率百分位 → 平均實現報酬
    oos = pd.read_csv(MODEL_DIR / "oos_predictions.csv.gz", dtype={"code": str}).dropna(subset=["prob", "next_ret"])
    oos["pct"] = oos.groupby("date").prob.rank(pct=True, method="first") - 1e-9
    cal_cc = _calib(oos.pct.values, oos.next_ret.values)
    oos_b = np.minimum((oos.pct.values * NB).astype(int), NB - 1)
    stats["lgbm_close"] = _stat(cal_cc[oos_b], oos.next_ret.values)
    stats["lgbm_close"]["range"] = [oos.date.min(), oos.date.max()]
    stats["lgbm_close"]["auc_note"] = "機率>0.5 方向命中率 %.1f%%" % (100 * np.mean((oos.prob > 0.5) == (oos.next_ret > 0)))

    rows = {}
    pxg = {c: g for c, g in pxa.sort_values("date").groupby("code")}
    for c in info.index:
        ser = C[c].dropna()
        v = ser.values[-GWIN - 5:]
        n = len(v)
        ld = ser.index[-1]
        p, x = {}, {}
        for k, need in NEED.items():
            if n < need or (k == "grey" and not (v[-GWIN:] > 0).all()):
                p[k] = None
                x[k] = f"資料不足：只有 {n} 筆收盤價，需 {need} 筆" if n < need else "收盤價含 0，無法建模"
                continue
            P = v[None, :]
            val = _algos_from_matrix(np.concatenate([np.full((1, max(0, LIN_WIN - n)), np.nan), P], axis=1)
                                     if n < LIN_WIN else P)[k][0]
            p[k] = round(float(val), 2) if np.isfinite(val) else None
            if p[k] is None:
                x[k] = "本日未算（數值無法計算）"
        row = {"n": info.at[c, "name"], "m": info.at[c, "market"], "c": float(v[-1]), "p": p}
        if ld != last_date:
            row["ld"] = ld          # 最後成交日（非資料日 → 資料日無成交）
        if x:
            row["x"] = x
        rows[c] = row
    # 興櫃：只有最新價（無歷史序列），其餘演算法顯示資料不足
    for c, (nm, pxv, ed) in _emerging().items():
        if c in rows:
            continue
        rows[c] = {"n": nm, "m": "興櫃", "c": pxv, "ld": ed, "esb": 1,
                   "p": {k: (round(pxv, 2) if k == "unchanged" else None) for k in NEED},
                   "x": {k: "興櫃只有當日行情（無歷史收盤序列），資料不足" for k in NEED if k != "unchanged"},
                   "lr": "興櫃股票不在 LightGBM 訓練範圍（無日收盤序列、無三大法人／融資券資料）"}
    n_scored_universe = len(codes)

    target = None
    def load(kind, fname):
        f = PRED / fname
        if not f.exists():
            return None
        d = pd.read_csv(f, dtype={"code": str})
        d = d[d.pred_date == last_date]
        return d if len(d) else None
    close = load("close", f"close_{last_date}.csv.gz")
    if close is not None:
        target = close.target_date.iloc[0]
        close["pct"] = close.prob.rank(pct=True, method="first") - 1e-9
        for r in close.itertuples():
            if r.code in rows:
                b = min(int(r.pct * NB), NB - 1)
                rows[r.code]["p"]["lgbm_close"] = round(rows[r.code]["c"] * (1 + cal_cc[b]), 2)
                rows[r.code]["pr"] = round(float(r.prob), 4); rows[r.code]["rk"] = int(r.rank)
    pre_files = sorted(PRED.glob("preopen_*.csv.gz"))
    pre = None
    from . import calendar as _cal
    for f in pre_files[::-1]:
        pre = load("preopen", f.name)
        if pre is not None and not _cal.is_trading_day(pre.target_date.iloc[0]):
            pre = None   # 目標日休市（如颱風假）的盤前預測已作廢
        if pre is not None:
            break
    preopen_meta = None
    if pre is not None:
        target = target or pre.target_date.iloc[0]
        # 盤前模型目標是 開→收 超額；無個股 OOS 紀錄，借用收盤模型 OOS 的「開→收」十分組 + 「隔夜跳空」十分組
        oos_gap = (1 + oos.next_ret.values) / (1 + oos.next_oc.values) - 1
        cal_oc = _calib(oos.pct.values, oos.next_oc.values)
        gap_mean = float(np.nanmean(oos_gap))
        pre["pct"] = pre.prob.rank(pct=True, method="first") - 1e-9
        for r in pre.itertuples():
            if r.code in rows:
                b = min(int(r.pct * NB), NB - 1)
                rows[r.code]["p"]["lgbm_preopen"] = round(rows[r.code]["c"] * (1 + gap_mean) * (1 + cal_oc[b]), 2)
                rows[r.code]["pp"] = round(float(r.prob), 4); rows[r.code]["pk"] = int(r.rank)
        preopen_meta = {"pred_file": f.name}
    # 沒有被主模型/盤前模型評分的代號：寫出原因
    for c, r in rows.items():
        if r.get("esb"):
            r["x"]["lgbm_close"] = r["x"]["lgbm_preopen"] = r.pop("lr")
            continue
        g = pxg.get(c)
        if g is None:
            continue
        if "lgbm_close" not in r["p"]:
            last_row = g[g.close > 0].iloc[-1]
            rs = _lgbm_reason(c, g, last_row, dates)
            r.setdefault("x", {})["lgbm_close"] = rs
            r["x"]["lgbm_preopen"] = rs if "未通過" in rs or "不在" in rs else "盤前模型本日未算（盤前預測檔未包含此檔）"
        elif "lgbm_preopen" not in r["p"]:
            r.setdefault("x", {})["lgbm_preopen"] = f"盤前模型本日未算（盤前預測於開盤前產生，目前尚無 {target or '下一交易日'} 的盤前預測檔）"
    if target is None or not _cal.is_trading_day(target):
        target = _cal.next_trading_day(dt.date.fromisoformat(last_date)).isoformat()

    # 盤前模型準確度：用即時對照紀錄（若有）
    try:
        pva = pd.concat([pd.read_csv(f, dtype={"code": str}) for f in (REPORT_DIR / "archive" / "prediction_vs_actual").glob("*.csv")])
        pp = pva[pva.kind == "preopen"].dropna(subset=["ret_cc", "prob"])
        if len(pp):
            pp = pp.copy(); pp["pct"] = pp.groupby("pred_date").prob.rank(pct=True, method="first") - 1e-9
            gm = float(np.nanmean((1 + oos.next_ret.values) / (1 + oos.next_oc.values) - 1))
            cal_oc = _calib(oos.pct.values, oos.next_oc.values)
            b = np.minimum((pp.pct.values * NB).astype(int), NB - 1)
            stats["lgbm_preopen"] = _stat((1 + gm) * (1 + cal_oc[b]) - 1, pp.ret_cc.values)
            stats["lgbm_preopen"]["range"] = [pp.target_date.min(), pp.target_date.max()]
            stats["lgbm_preopen"]["live"] = True
    except Exception as e:
        print("preopen accuracy failed:", e)

    algos = [
        {"k": "lgbm_close", "name": "LightGBM 收盤模型（主模型）", "note": "收盤後資料預測隔日上漲機率；以歷史樣本外同機率十分組的平均實際報酬換算成價格", "src": "樣本外 walk-forward"},
        {"k": "lgbm_preopen", "name": "LightGBM 盤前模型", "note": "加入美股、台指期夜盤後重新評分（預測開→收）；價格換算借用收盤模型 OOS 的平均跳空＋開→收十分組報酬，較粗略", "src": "即時對照紀錄（天數少）"},
        {"k": "grey", "name": "灰色預測 GM(1,1)（影子模式）", "note": f"用最近 {GWIN} 個收盤價建 GM(1,1) 外推一步；尚在 30 天觀察期，不影響名單", "src": "歷史回測"},
        {"k": "linear10", "name": f"線性趨勢（{LIN_WIN} 日迴歸）", "note": f"最近 {LIN_WIN} 個收盤價做直線迴歸，外推下一天", "src": "歷史回測"},
        {"k": "ma5", "name": "5 日均價", "note": "預測隔日收盤回到最近 5 日平均", "src": "歷史回測"},
        {"k": "momentum1", "name": "昨日漲跌延續", "note": "假設今天的漲跌幅明天重演一次", "src": "歷史回測"},
        {"k": "unchanged", "name": "不變（基準）", "note": "隔日收盤 = 最新收盤；用來比較其他方法有沒有比較準", "src": "歷史回測"},
    ]
    for a in algos:
        s = stats.get(a["k"])
        if s and "range" not in s:
            s["range"] = bt_range
        a["acc"] = s
    payload = {"data_date": last_date, "target_date": target, "generated": dt.datetime.now().isoformat(timespec="minutes"),
               "algos": algos, "preopen": preopen_meta, "n_stocks": len(rows), "n_full": n_scored_universe, "n_gone": len(gone), "gone": gone,
               "note": "預測報酬誤差 MAE 為「預測漲跌幅 − 實際漲跌幅」的平均絕對值；方向命中率不計平盤；回測排除單日 |報酬|>10.5% 的除權息/異常日。",
               "s": rows}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=lambda o: None).replace("NaN", "null"))
    print(f"lookup.json: {len(rows)} 檔, {OUT.stat().st_size/1e6:.2f} MB, 資料日 {last_date} → 目標 {target}")
    return payload


if __name__ == "__main__":
    build()
