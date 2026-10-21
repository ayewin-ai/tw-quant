"""灰色預測 GM(1,1) — 影子模式 (shadow mode)。

每天用最近 WIN 個收盤價對每檔股票(與加權指數)建 GM(1,1)，預測下一交易日收盤價/報酬/方向。
**不參與**主模型排序或權重；只存檔並於隔日與實際結果比較，累積 30 個交易日後再決定是否啟用。

GM(1,1)：x1 = cumsum(x0)；z_k = (x1_k + x1_{k-1})/2；以最小平方解 x0_k = -a z_k + b；
x1hat(k) = (x0_1 - b/a) e^{-a(k-1)} + b/a；x0hat(n+1) = x1hat(n+1) - x1hat(n)。
為數值穩定，先以序列第一個值正規化 (x0 / x0[0])，GM 對尺度不變。

CLI:  python -m twq.grey predict [--date D] | evaluate | backtest | all [--date D]
"""
import argparse
import datetime as dt
import gzip
import json
import sqlite3
import warnings

import numpy as np
import pandas as pd

from .config import DB_PATH, MODEL_DIR, PRED_DIR, REPORT_DIR, MIN_AVG_VALUE, MIN_PRICE

WIN = 6                 # 用最近 6 個收盤價 (5 個差分) 建模
TOP_N = 10
COST = 0.001425 * 2 * 0.6 + 0.003   # 手續費 6 折 ×2 + 證交稅 0.3%（開→收的往返成本，估計值）
NEED_DAYS = 30
ARCH = REPORT_DIR / "archive" / "grey"
DAILY = ARCH / "daily"
EVAL_LOG = ARCH / "eval_log.csv"
BT_PATH = MODEL_DIR / "grey_backtest.json"


def gm11_next(X):
    """X: (m, n) 正值序列矩陣（每列一檔）。回傳 (下一期預測值, 發展係數 a, 擬合 MAPE)。向量化。"""
    X = np.asarray(X, float)
    s = X[:, :1]
    x0 = X / s
    n = x0.shape[1]
    x1 = np.cumsum(x0, axis=1)
    z = 0.5 * (x1[:, 1:] + x1[:, :-1])          # (m, n-1)
    y = x0[:, 1:]
    # 對每列解 [-z 1][a b]^T = y 的最小平方（2x2 正規方程）
    Szz = (z * z).sum(1); Sz = z.sum(1); k = n - 1
    Szy = (z * y).sum(1); Sy = y.sum(1)
    det = Szz * k - Sz * Sz
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        a = -(k * Szy - Sz * Sy) / det
        b = (Szz * Sy - Sz * Szy) / det
        ba = b / a
        c = x0[:, 0] - ba
        t = np.arange(n + 1)
        x1h = c[:, None] * np.exp(-a[:, None] * t[None, :]) + ba[:, None]
        x0h = np.diff(x1h, axis=1)              # 預測 x0[1..n]
        fit = np.abs(x0h[:, :-1] - x0[:, 1:]) / x0[:, 1:]
        mape = np.nanmean(fit, axis=1)
        nxt = x0h[:, -1] * s[:, 0]
    bad = ~np.isfinite(nxt) | (np.abs(a) < 1e-12) | (nxt <= 0)
    # a≈0（完全平坦）時退化為最後一個值
    nxt = np.where(bad, X[:, -1], nxt)
    return nxt, a, mape


def _load_prices(con, start):
    return pd.read_sql_query(
        "select date, code, name, open, close, value from prices where date >= ? and close > 0",
        con, params=(start,))


def _panel(px):
    close = px.pivot(index="date", columns="code", values="close").sort_index()
    opn = px.pivot(index="date", columns="code", values="open").reindex(close.index)
    val = px.pivot(index="date", columns="code", values="value").reindex(close.index)
    return close, opn, val


def _universe(close, val, i):
    """與主模型相同的流動性門檻（20 日均成交額、股價），且需 4 碼普通股。"""
    v20 = val.iloc[max(0, i - 19): i + 1].mean()
    c = close.iloc[i]
    codes = close.columns[(v20 >= MIN_AVG_VALUE) & (c >= MIN_PRICE) & close.columns.str.fullmatch(r"\d{4}")]
    return codes


def _predict_at(close, val, i, codes=None):
    if codes is None:
        codes = _universe(close, val, i)
    W = close.iloc[i - WIN + 1: i + 1][codes]
    W = W.loc[:, W.notna().all()]
    if W.shape[1] == 0:
        return pd.DataFrame()
    nxt, a, mape = gm11_next(W.T.values)
    last = W.iloc[-1].values
    prev = W.iloc[-2].values
    return pd.DataFrame({"code": W.columns, "close": last, "grey_close": nxt,
                         "grey_ret": nxt / last - 1, "grey_dir": np.sign(nxt / last - 1).astype(int),
                         "a": a, "fit_mape": mape, "naive_ret": last / prev - 1})


# ---------------------------------------------------------------- daily shadow prediction
def predict(date=None):
    con = sqlite3.connect(DB_PATH)
    start = (dt.date.today() - dt.timedelta(days=90)).isoformat() if date is None else \
        (dt.date.fromisoformat(date) - dt.timedelta(days=90)).isoformat()
    px = _load_prices(con, start)
    close, _, val = _panel(px)
    if date is None:
        date = close.index[-1]
    if date not in close.index:
        print(f"[grey] {date} 無價格資料，略過"); return None
    i = close.index.get_loc(date)
    df = _predict_at(close, val, i)
    names = px[px.date == date].set_index("code")["name"]
    df.insert(1, "name", df.code.map(names))
    df.insert(0, "pred_date", date)
    df["grey_rank"] = df.grey_ret.rank(ascending=False, method="first").astype(int)
    # 加權指數
    ix = pd.read_sql_query("select date, close from index_daily where name='TAIEX' and date <= ? order by date",
                           con, params=(date,)).tail(WIN)
    if len(ix) == WIN and ix.date.iloc[-1] == date:
        n, a, m = gm11_next(ix.close.values[None, :])
        df = pd.concat([df, pd.DataFrame([{"pred_date": date, "code": "TAIEX", "name": "加權指數",
                                           "close": ix.close.iloc[-1], "grey_close": n[0],
                                           "grey_ret": n[0] / ix.close.iloc[-1] - 1,
                                           "grey_dir": int(np.sign(n[0] / ix.close.iloc[-1] - 1)),
                                           "a": a[0], "fit_mape": m[0],
                                           "naive_ret": ix.close.iloc[-1] / ix.close.iloc[-2] - 1,
                                           "grey_rank": 0}])], ignore_index=True)
    DAILY.mkdir(parents=True, exist_ok=True)
    gz = DAILY / f"grey_{date}.csv.gz"
    if gz.exists():
        print(f"[grey] {date} 已存檔（append-only，不覆寫）：{gz.name}")
        return df
    with gzip.open(gz, "wt", encoding="utf-8") as f:
        df.to_csv(f, index=False)
    mpath = ARCH / f"{date[:7]}.csv"
    df.to_csv(mpath, mode="a", header=not mpath.exists(), index=False, float_format="%.6g")
    print(f"[grey] {date}：{len(df)} 檔灰色預測已存檔（影子模式，不影響名單）")
    return df


# ---------------------------------------------------------------- metrics
def _metrics(g, act_c, act_o, prev_c):
    """g: DataFrame(code, grey_ret, naive_ret)。回傳 dict。"""
    g = g[g.code.isin(act_c.index)].copy()
    g["act_ret"] = act_c[g.code].values / prev_c[g.code].values - 1
    g["act_oc"] = act_c[g.code].values / act_o[g.code].values - 1
    g = g.dropna(subset=["act_ret"])
    nz = g[g.act_ret != 0]
    top = g.nsmallest(TOP_N, "grey_rank") if "grey_rank" in g else g.nlargest(TOP_N, "grey_ret")
    return {
        "n": len(g),
        "dir_hit": float((np.sign(nz.grey_ret) == np.sign(nz.act_ret)).mean()),
        "naive_dir_hit": float((np.sign(nz.naive_ret) == np.sign(nz.act_ret)).mean()),
        "up_share": float((nz.act_ret > 0).mean()),
        "mae": float((g.grey_ret - g.act_ret).abs().mean()),
        "mae_nochange": float(g.act_ret.abs().mean()),
        "mape_close": float(((g.grey_ret - g.act_ret).abs() / (1 + g.act_ret)).mean()),
        "rank_ic": float(g.grey_ret.rank().corr(g.act_ret.rank())),
        "naive_rank_ic": float(g.naive_ret.rank().corr(g.act_ret.rank())),
        "top_cc": float(top.act_ret.mean()), "top_oc": float(top.act_oc.mean()),
        "top_oc_net": float(top.act_oc.mean() - COST),
        "mkt_cc": float(g.act_ret.mean()), "mkt_oc": float(g.act_oc.mean()),
    }


def evaluate():
    con = sqlite3.connect(DB_PATH)
    done = set(pd.read_csv(EVAL_LOG).pred_date) if EVAL_LOG.exists() else set()
    files = sorted(DAILY.glob("grey_*.csv.gz"))
    rows = []
    for f in files:
        d = f.name[5:15]
        if d in done:
            continue
        nd = con.execute("select min(date) from prices where date > ?", (d,)).fetchone()[0]
        if not nd:
            continue
        g = pd.read_csv(f, dtype={"code": str})
        px = pd.read_sql_query("select date, code, open, close from prices where date in (?, ?)", con, params=(d, nd))
        c0 = px[px.date == d].set_index("code").close
        c1 = px[px.date == nd].set_index("code").close
        o1 = px[px.date == nd].set_index("code").open
        st = g[g.code != "TAIEX"]
        m = _metrics(st, c1, o1, c0)
        r = {"pred_date": d, "target_date": nd, **m}
        # 加權指數方向
        tx = g[g.code == "TAIEX"]
        ix = con.execute("select close from index_daily where name='TAIEX' and date=?", (nd,)).fetchone()
        if len(tx) and ix:
            act = ix[0] / tx.close.iloc[0] - 1
            r.update(taiex_grey_ret=float(tx.grey_ret.iloc[0]), taiex_act_ret=act,
                     taiex_dir_hit=int(np.sign(tx.grey_ret.iloc[0]) == np.sign(act)))
        # 主模型同一天的前 10 名（收盤後版）
        pj = PRED_DIR / f"{d}.json"
        if pj.exists():
            o = json.loads(pj.read_text())
            codes = [p["code"] for p in sorted(o["predictions"], key=lambda p: p["rank"])][:TOP_N]
            codes = [c for c in codes if c in c1.index and c in c0.index]
            if codes:
                r["main_top_cc"] = float((c1[codes] / c0[codes] - 1).mean())
                r["main_top_oc"] = float((c1[codes] / o1[codes] - 1).mean())
        rows.append(r)
        print(f"[grey] 檢討 {d}→{nd}: 方向命中 {m['dir_hit']:.1%}，Rank IC {m['rank_ic']:+.3f}，前10 收→收 {m['top_cc']:+.2%}")
    if rows:
        new = pd.DataFrame(rows)
        new.to_csv(EVAL_LOG, mode="a", header=not EVAL_LOG.exists(), index=False, float_format="%.6g")
    return rows


# ---------------------------------------------------------------- historical backtest
def backtest(start="2024-09-01"):
    con = sqlite3.connect(DB_PATH)
    px = _load_prices(con, (dt.date.fromisoformat(start) - dt.timedelta(days=60)).isoformat())
    close, opn, val = _panel(px)
    ix = pd.read_sql_query("select date, close from index_daily where name='TAIEX' order by date", con).set_index("date").close
    dates = close.index
    rows = []
    for i in range(max(WIN, 20), len(dates) - 1):
        if dates[i] < start:
            continue
        g = _predict_at(close, val, i)
        if g.empty:
            continue
        g["grey_rank"] = g.grey_ret.rank(ascending=False, method="first")
        m = _metrics(g, close.iloc[i + 1], opn.iloc[i + 1], close.iloc[i])
        r = {"pred_date": dates[i], "target_date": dates[i + 1], **m}
        w = ix.loc[:dates[i]].tail(WIN)
        if len(w) == WIN and dates[i + 1] in ix.index:
            n_, _, _ = gm11_next(w.values[None, :])
            act = ix[dates[i + 1]] / w.iloc[-1] - 1
            r["taiex_dir_hit"] = int(np.sign(n_[0] / w.iloc[-1] - 1) == np.sign(act))
            r["taiex_naive_hit"] = int(np.sign(w.iloc[-1] / w.iloc[-2] - 1) == np.sign(act))
            r["taiex_up"] = int(act > 0)
        rows.append(r)
    b = pd.DataFrame(rows)
    summ = {k: float(b[k].mean()) for k in ["dir_hit", "naive_dir_hit", "up_share", "mae", "mae_nochange",
                                             "mape_close", "rank_ic", "naive_rank_ic", "top_cc", "top_oc",
                                             "top_oc_net", "mkt_cc", "mkt_oc", "taiex_dir_hit",
                                             "taiex_naive_hit", "taiex_up"]}
    summ["rank_ic_t"] = float(b.rank_ic.mean() / b.rank_ic.std() * np.sqrt(len(b)))
    summ["days"] = len(b); summ["period"] = f"{b.pred_date.iloc[0]} ~ {b.target_date.iloc[-1]}"
    summ["cum_top_oc_net"] = float((1 + b.top_oc_net).prod() - 1)
    summ["cum_top_cc"] = float((1 + b.top_cc).prod() - 1)
    summ["cum_mkt_cc"] = float((1 + b.mkt_cc).prod() - 1)
    out = {"window": WIN, "cost": COST, "summary": summ,
           "curve": {"dates": b.target_date.tolist(),
                     "top_cc": ((1 + b.top_cc).cumprod() - 1).round(5).tolist(),
                     "top_oc_net": ((1 + b.top_oc_net).cumprod() - 1).round(5).tolist(),
                     "mkt_cc": ((1 + b.mkt_cc).cumprod() - 1).round(5).tolist(),
                     "rank_ic_20": b.rank_ic.rolling(20).mean().round(4).tolist()}}
    BT_PATH.write_text(json.dumps(out, ensure_ascii=False, default=float))
    print(json.dumps(summ, ensure_ascii=False, indent=1))
    return out


def site_payload():
    out = {"need_days": NEED_DAYS, "window": WIN, "cost": COST}
    if BT_PATH.exists():
        bt = json.loads(BT_PATH.read_text()); out["backtest"] = bt
    if EVAL_LOG.exists():
        e = pd.read_csv(EVAL_LOG).drop_duplicates("pred_date", keep="first").sort_values("pred_date")
        out["live"] = json.loads(e.to_json(orient="records"))
        out["progress"] = len(e)
    else:
        out["live"] = []; out["progress"] = 0
    files = sorted(DAILY.glob("grey_*.csv.gz"))
    if files:
        g = pd.read_csv(files[-1], dtype={"code": str})
        out["first_pred_date"] = files[0].name[5:15]
        out["latest_pred_date"] = files[-1].name[5:15]
        out["latest_top"] = json.loads(g[g.code != "TAIEX"].nsmallest(TOP_N, "grey_rank")[
            ["code", "name", "close", "grey_close", "grey_ret", "a", "fit_mape"]].to_json(orient="records"))
        tx = g[g.code == "TAIEX"]
        if len(tx):
            out["latest_taiex"] = json.loads(tx[["close", "grey_close", "grey_ret"]].to_json(orient="records"))[0]
    return out


def main(argv=None):
    p = argparse.ArgumentParser(prog="twq.grey")
    p.add_argument("cmd", choices=["predict", "evaluate", "backtest", "all"])
    p.add_argument("--date", default=None)
    a = p.parse_args(argv)
    if a.cmd in ("evaluate", "all"):
        evaluate()
    if a.cmd in ("predict", "all"):
        predict(a.date)
    if a.cmd == "backtest":
        backtest()


if __name__ == "__main__":
    main()
