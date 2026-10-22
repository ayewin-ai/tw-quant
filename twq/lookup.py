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

from .config import DB_PATH, MODEL_DIR, REPORT_DIR, ROOT
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


def build():
    con = sqlite3.connect(DB_PATH)
    dates = [r[0] for r in con.execute("select distinct date from prices order by date desc limit ?", (BT_DAYS + 15,))][::-1]
    px = pd.read_sql_query("select date, code, name, market, close from prices where date >= ? and close > 0",
                           con, params=(dates[0],))
    con.close()
    last_date = dates[-1]
    C = px.pivot(index="date", columns="code", values="close").reindex(dates)
    info = px[px.date == last_date].set_index("code")[["name", "market"]]
    codes = [c for c in info.index if C[c].iloc[-LIN_WIN:].notna().all()]

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

    W = C[codes].values[-LIN_WIN:].T
    preds = _algos_from_matrix(W)
    last = W[:, -1]
    rows = {}
    for j, c in enumerate(codes):
        rows[c] = {"n": info.at[c, "name"], "m": info.at[c, "market"], "c": float(last[j]),
                   "p": {k: (round(float(v[j]), 2) if np.isfinite(v[j]) else None) for k, v in preds.items()}}

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
    for f in pre_files[::-1]:
        pre = load("preopen", f.name)
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
    if target is None:
        d = dt.date.fromisoformat(last_date) + dt.timedelta(days=1)
        while d.weekday() >= 5:
            d += dt.timedelta(days=1)
        target = d.isoformat()

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
               "algos": algos, "preopen": preopen_meta, "n_stocks": len(rows),
               "note": "預測報酬誤差 MAE 為「預測漲跌幅 − 實際漲跌幅」的平均絕對值；方向命中率不計平盤；回測排除單日 |報酬|>10.5% 的除權息/異常日。",
               "s": rows}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=lambda o: None).replace("NaN", "null"))
    print(f"lookup.json: {len(rows)} 檔, {OUT.stat().st_size/1e6:.2f} MB, 資料日 {last_date} → 目標 {target}")
    return payload


if __name__ == "__main__":
    build()
