"""投組層級風險分析（離線、只用公開價量；不依賴 gs-quant 套件/帳號）。

概念出處（只借概念與公式，程式為自行以 numpy/pandas 重寫，不 vendor）：
  • Goldman Sachs gs-quant (Apache-2.0) — gs_quant/timeseries（volatility、exponential_std、max_drawdown、beta、correlation、
    bollinger/macd/RSI、zscores/winsorize）、gs_quant/markets/optimizer（風險/報酬最適化的概念）、
    gs_quant/models（因子風險模型：總風險 = 因子風險 + 特定風險）。
  • LAIN-21/dalio_uncorrelated_returns — Ray Dalio「聖杯」：15 條不相關報酬流可把風險降 80%；風險平價(risk parity)、
    反波動(inverse volatility) 加權、分散化比率(diversification ratio)、有效獨立賭注數(effective number of independent bets)。
本模組提供：報酬流相關矩陣、分散化比率、有效獨立賭注、Dalio 聖杯曲線、加權方式（等權/反波動/風險平價/低相關）、
單日 Top-N 的投組風險報告（VaR/CVaR/最大回撤/風險貢獻/產業集中度/beta），以及加權方式的護欄回測。
全部是「描述與研究」，預設不影響每日名單；只有通過與 `twq tune` 相同護欄的項目才會啟用。
"""
import datetime as dt
import json

import numpy as np
import pandas as pd

from . import db
from .config import MODEL_DIR, REPORT_DIR, ROOT

ANN = 252
RISK_DIR = REPORT_DIR / "risk"


# ------------------------------------------------------------------ 單一序列統計
def max_drawdown(ret):
    eq = (1 + pd.Series(ret).fillna(0)).cumprod()
    return float((eq / eq.cummax() - 1).min())


def var_cvar(ret, a=0.05):
    r = np.sort(np.asarray(pd.Series(ret).dropna()))
    if len(r) < 20:
        return None, None
    k = max(1, int(len(r) * a))
    return float(-np.quantile(r, a)), float(-r[:k].mean())


def ewma_vol(ret, lam=0.94):
    r = pd.Series(ret).dropna()
    return float(np.sqrt((r ** 2).ewm(alpha=1 - lam, adjust=False).mean().iloc[-1])) if len(r) > 10 else None


def stream_stats(ret):
    r = pd.Series(ret).dropna()
    if len(r) < 20:
        return {}
    v, c = var_cvar(r)
    sd = r.std()
    return {"n": int(len(r)), "mean_bp": float(r.mean() * 1e4), "vol_ann": float(sd * np.sqrt(ANN)),
            "sharpe": float(r.mean() / sd * np.sqrt(ANN)) if sd > 0 else None, "mdd": max_drawdown(r),
            "var95": v, "cvar95": c, "ewvol_ann": (ewma_vol(r) or 0) * np.sqrt(ANN), "skew": float(r.skew())}


# ------------------------------------------------------------------ 分散化度量
def shrink_cov(R, lam=0.3):
    """樣本共變異數向「等相關」目標收縮（避免 10 檔×60 日的 cov 太雜）。R: DataFrame(days x assets)"""
    S = R.cov().values
    sd = np.sqrt(np.diag(S))
    C = S / np.outer(sd, sd)
    n = len(C)
    rbar = (C.sum() - n) / (n * (n - 1)) if n > 1 else 0
    T = np.full_like(C, rbar)
    np.fill_diagonal(T, 1)
    Cs = (1 - lam) * C + lam * T
    return Cs * np.outer(sd, sd)


def diversification_ratio(w, cov):
    w = np.asarray(w, float)
    sd = np.sqrt(np.diag(cov))
    pv = float(np.sqrt(w @ cov @ w))
    return float((w * sd).sum() / pv) if pv > 0 else None


def eff_bets(corr):
    """有效獨立賭注數。回傳 (參與比 PR=(Σλ)²/Σλ²,  熵版 exp(-Σp ln p), p=λ/Σλ)。完全不相關 N 檔 → N；完全相關 → 1。"""
    C = np.asarray(corr, float)
    C = np.nan_to_num(C, nan=0.0)
    np.fill_diagonal(C, 1)
    lam = np.clip(np.linalg.eigvalsh(C), 1e-12, None)
    p = lam / lam.sum()
    return float(lam.sum() ** 2 / (lam ** 2).sum()), float(np.exp(-(p * np.log(p)).sum()))


def holy_grail(sr, rho, nmax=15):
    """Dalio：N 條平均相關 ρ、個別夏普 SR 的報酬流 → 組合夏普 = SR·sqrt(N/(1+(N-1)ρ))。"""
    return [{"n": n, "sharpe": float(sr * np.sqrt(n / (1 + (n - 1) * rho))),
             "risk_reduction": float(1 - np.sqrt((1 + (n - 1) * rho) / n))} for n in range(1, nmax + 1)]


# ------------------------------------------------------------------ 加權方式
def w_equal(n):
    return np.full(n, 1.0 / n)


def w_inv_vol(vol):
    v = 1.0 / np.clip(np.asarray(vol, float), 1e-6, None)
    return v / v.sum()


def w_risk_parity(cov, iters=300):
    """等風險貢獻 (ERC)。cov 為 n×n。以反波動為起點的乘法迭代。"""
    n = len(cov)
    sd = np.sqrt(np.diag(cov))
    w = w_inv_vol(sd)
    for _ in range(iters):
        rc = w * (cov @ w)
        tgt = rc.mean()
        w = w * (tgt / np.clip(rc, 1e-18, None)) ** 0.5
        w = w / w.sum()
    return w


def w_min_corr(corr, vol):
    """低相關加權（簡化版 Minimum Correlation，Varadi 2012 概念）：權重 ∝ (1 − 平均相關)/波動，相關越低、波動越小權重越高。"""
    C = np.nan_to_num(np.asarray(corr, float), nan=0.0)
    n = len(C)
    avg = (C.sum(1) - 1) / max(n - 1, 1)
    s = np.clip(1.0 - avg, 0.05, None) / np.clip(np.asarray(vol, float), 1e-6, None)
    return s / s.sum()


SCHEMES = {"equal": "等權（現行）", "inv_vol": "反波動", "risk_parity": "風險平價(ERC)", "min_corr": "低相關加權"}


def weights_for(R, schemes=SCHEMES):
    """R: DataFrame(days x picks) 過去報酬 → dict scheme -> weights ndarray"""
    n = R.shape[1]
    out = {"equal": w_equal(n)}
    R = R.dropna(axis=1, thresh=int(len(R) * 0.6)).reindex(columns=R.columns).fillna(0)
    sd = R.std().values
    sd = np.where(sd <= 0, np.nanmedian(sd[sd > 0]) if (sd > 0).any() else 0.02, sd)
    out["inv_vol"] = w_inv_vol(sd)
    try:
        cov = shrink_cov(R)
        out["risk_parity"] = w_risk_parity(cov)
        corr = R.corr().values
        out["min_corr"] = w_min_corr(corr, sd)
    except Exception:
        out["risk_parity"] = out["inv_vol"]
        out["min_corr"] = out["inv_vol"]
    return out


def select_min_corr(cands, R, k=10, lam=1.0):
    """從分數排序的候選(top-N)中貪婪挑 k 檔：位置越前越好，同時與已選者的平均相關越低越好。cands: code list (已依分數排序)"""
    C = R.reindex(columns=cands).corr().fillna(0)
    chosen = [cands[0]]
    while len(chosen) < k and len(chosen) < len(cands):
        best, bs = None, 1e9
        for i, c in enumerate(cands):
            if c in chosen:
                continue
            s = i / len(cands) + lam * C.loc[c, chosen].mean()
            if s < bs:
                best, bs = c, s
        chosen.append(best)
    return chosen


# ------------------------------------------------------------------ 報酬流
SECTOR_GROUPS = {
    "半導體": ["半導體"],
    "其他電子": ["電子零組件", "電腦及週邊", "光電", "通信網路", "其他電子", "電子通路", "資訊服務", "數位雲端"],
    "金融保險": ["金融保險"],
    "生技醫療": ["生技醫療"],
    "傳產": ["水泥", "食品", "塑膠", "紡織纖維", "電機機械", "電器電纜", "化學", "玻璃陶瓷", "造紙", "鋼鐵", "橡膠", "汽車",
           "建材營造", "航運", "觀光餐旅", "貿易百貨", "油電燃氣", "綠能環保", "運動休閒", "居家生活", "文化創意", "農業科技"],
}
STREAM_NAMES = {
    "close_top10": "收盤模型 Top-10（收→收）", "close_top10_oc": "收盤模型 Top-10（開→收）",
    "preopen_top10_oc": "盤前模型 Top-10（開→收）", "grey_top10": "灰色預測 Top-10（收→收，影子）",
    "taiex": "加權指數(TAIEX)", "etf0050": "元大台灣50(0050)", "ew_liquid": "流動股等權(全市場基準)",
    "naive_winners": "樸素：前日漲幅前10", "naive_losers": "樸素：前日跌幅前10", "naive_lowvol": "樸素：低波動前10",
    **{f"sec_{k}": f"產業：{k}" for k in SECTOR_GROUPS},
}


def _wide(con, start="2024-06-01"):
    px = pd.read_sql_query("SELECT date,code,close,ret,value,volume FROM prices WHERE date>=?", con, params=(start,))
    px = px[px.code.str.fullmatch(r"[1-9]\d{3}|0050")]
    info = pd.read_sql_query("SELECT code,industry FROM stock_info", con).set_index("code").industry
    idx = pd.read_sql_query("SELECT date,pct FROM index_daily WHERE name='TAIEX'", con).set_index("date").pct
    return px, info, idx


def build_streams():
    """日報酬流（日期=報酬實現日）。模型流取自 models/ablation_daily_*（walk-forward OOS，開→收已含成本者另列）。"""
    con = db.connect()
    px, info, idx = _wide(con)
    con.close()
    S = {}
    cal = sorted(set(idx.index) | set(px.date))
    nxt_day = dict(zip(cal[:-1], cal[1:]))

    def csv(name, col, key):
        f = MODEL_DIR / f"ablation_daily_{name}.csv"
        if f.exists():
            d = pd.read_csv(f).set_index("date")
            S[key] = d[col]
            S[key].index = [nxt_day.get(x) for x in d.index]      # ablation 以「訊號日」記帳 → 改成報酬實現日
            S[key] = S[key][S[key].index.notna()]
    csv("base_y", "cc", "close_top10")
    csv("base_y", "oc", "close_top10_oc")
    csv("preopen_y_oc", "oc", "preopen_top10_oc")
    gb = MODEL_DIR / "grey_backtest.json"
    if gb.exists():
        c = json.loads(gb.read_text())["curve"]
        cum = pd.Series(c["top_cc"], index=c["dates"])          # 累積報酬曲線（以實現日記帳）→ 轉日報酬
        S["grey_top10"] = (1 + cum).pct_change()
    S["taiex"] = idx
    R = px.pivot(index="date", columns="code", values="ret").clip(-0.2, 0.2)
    if "0050" in R:
        S["etf0050"] = R["0050"]
    stk = [c for c in R.columns if c != "0050"]
    Rs = R[stk]
    V = px.pivot(index="date", columns="code", values="value").reindex(columns=stk)
    Cl = px.pivot(index="date", columns="code", values="close").reindex(columns=stk)
    liquid = (V.rolling(20, min_periods=10).mean() >= 5e7) & (Cl >= 10)
    nxt = Rs.shift(-1)                       # 下一日報酬 → 以「實現日」標記
    L = liquid & Rs.notna()
    S["ew_liquid"] = Rs.where(L.shift(1, fill_value=False)).mean(axis=1)
    vol60 = Rs.rolling(60, min_periods=40).std()

    def top(sig, asc=False):
        sig = sig.where(L)
        rk = sig.rank(axis=1, ascending=asc, method="first")
        pick = rk <= 10
        # 選股用 t 日資料，報酬實現在 t+1：以 t+1 日標記
        return nxt.where(pick).mean(axis=1).shift(1)
    S["naive_winners"] = top(Rs)
    S["naive_losers"] = top(Rs, asc=True)
    S["naive_lowvol"] = top(vol60, asc=True)
    ind = info.reindex(Rs.columns)
    for g, mem in SECTOR_GROUPS.items():
        cols = [c for c in Rs.columns if ind.get(c) in mem and True]
        if cols:
            S[f"sec_{g}"] = Rs[cols].where(liquid[cols].shift(1, fill_value=False)).mean(axis=1)
    df = pd.DataFrame(S).sort_index()
    return df.loc[df.index >= "2025-03-10"]


def streams_report(df=None):
    df = build_streams() if df is None else df
    cols = [c for c in STREAM_NAMES if c in df]
    d = df[cols]
    C = d.corr(min_periods=60)
    st = {c: stream_stats(d[c]) for c in cols}
    # 分散化：把「系統產生的流」與「基準流」分開看
    sys_cols = [c for c in ("close_top10", "preopen_top10_oc", "grey_top10") if c in d]
    common = d[cols].dropna()
    out = {"names": {c: STREAM_NAMES[c] for c in cols}, "cols": cols, "corr": np.round(C.values, 3).tolist(),
           "stats": st, "period": [str(d.index.min()), str(d.index.max())], "n_days_common": int(len(common)),
           "common_period": [str(common.index.min()), str(common.index.max())] if len(common) else None}
    n_eff = {}
    for label, cs in (("system", sys_cols), ("all", cols), ("sectors", [c for c in cols if c.startswith("sec_")]),
                      ("system+taiex+0050", sys_cols + [c for c in ("taiex", "etf0050") if c in d])):
        if len(cs) >= 2:
            sub = d[cs].dropna()
            if len(sub) >= 60:
                cov = sub.cov().values
                pr, en = eff_bets(sub.corr().values)
                n = len(cs)
                n_eff[label] = {"streams": cs, "n": n, "days": int(len(sub)), "pr": pr, "entropy": en,
                                "avg_corr": float((sub.corr().values.sum() - n) / (n * (n - 1))),
                                "div_ratio_equal": diversification_ratio(np.full(n, 1 / n), cov),
                                "div_ratio_invvol": diversification_ratio(w_inv_vol(np.sqrt(np.diag(cov))), cov),
                                "div_ratio_rp": diversification_ratio(w_risk_parity(cov), cov)}
    out["diversification"] = n_eff
    sr_avg = np.nanmean([s.get("sharpe") for c, s in st.items() if c in sys_cols and s.get("sharpe") is not None] or [0])
    rho = n_eff.get("system", {}).get("avg_corr", 0.0)
    out["holy_grail"] = {"sr_per_stream": float(sr_avg), "avg_corr": float(rho), "curve_actual_rho": holy_grail(sr_avg, max(rho, 0), 15),
                         "curve_rho0": holy_grail(sr_avg, 0.0, 15), "curve_rho30": holy_grail(sr_avg, 0.3, 15),
                         "note": "Dalio：15 條真正不相關(ρ=0)且各自有正夏普的報酬流，組合風險可降約 80%。本系統只有 3 條流，彼此平均相關約 0.35、"
                                 "與大盤也高度相關；而「收→收」流的高夏普是毛值（未扣成本，且含收盤鎖死漲停、實際買不到的股票），可成交的「開→收」流扣成本後為負，所以聖杯在這裡不成立。"}
    return out, df


# ------------------------------------------------------------------ 單日投組風險報告
def portfolio_report(picks, label="close", asof=None, lookback=120):
    """picks: list of dict(code,name,industry,prob?) → 風險報告 dict（以 asof 以前的資料估計）。"""
    con = db.connect()
    codes = [p["code"] for p in picks]
    asof = asof or con.execute("SELECT MAX(date) FROM prices").fetchone()[0]
    px = pd.read_sql_query(f"SELECT date,code,ret FROM prices WHERE date<=? AND date>=date(?, '-{int(lookback * 1.7)} day') "
                           f"AND code IN ({','.join('?' * len(codes))})", con, params=[asof, asof] + codes)
    idx = pd.read_sql_query("SELECT date,pct FROM index_daily WHERE name='TAIEX' AND date<=?", con, params=(asof,)).set_index("date").pct
    con.close()
    R = px.pivot(index="date", columns="code", values="ret").clip(-0.2, 0.2).reindex(columns=codes).tail(lookback)
    R = R.fillna(0)
    m = idx.reindex(R.index).fillna(0)
    n = len(codes)
    ws = weights_for(R.tail(60))
    cov = shrink_cov(R)
    corr = R.corr().values
    sd = np.sqrt(np.diag(cov))
    pr, en = eff_bets(corr)
    beta = (R.apply(lambda s: np.cov(s, m)[0, 1]) / m.var()).values if m.var() > 0 else np.zeros(n)
    inds = pd.Series([p.get("industry") or "" for p in picks])
    hhi_ind = float(((inds.value_counts() / n) ** 2).sum())
    out = {"asof": asof, "label": label, "n": n, "lookback_days": int(len(R)),
           "avg_pair_corr": float((corr.sum() - n) / (n * (n - 1))) if n > 1 else None,
           "eff_bets_pr": pr, "eff_bets_entropy": en, "industry_hhi": hhi_ind,
           "industries": inds.value_counts().to_dict(), "schemes": {}}
    for k, w in ws.items():
        pv = float(np.sqrt(w @ cov @ w))
        port = (R.values * w).sum(1)
        v, c = var_cvar(port)
        rc = w * (cov @ w) / (pv ** 2) if pv > 0 else np.full(n, 1 / n)
        out["schemes"][k] = {"name": SCHEMES[k], "vol_ann": pv * np.sqrt(ANN), "var95_1d": v, "cvar95_1d": c,
                             "mdd_hist": max_drawdown(port), "div_ratio": diversification_ratio(w, cov),
                             "beta": float((w * beta).sum()), "max_weight": float(w.max()),
                             "weights": {c_: round(float(x), 4) for c_, x in zip(codes, w)},
                             "risk_contrib": {c_: round(float(x), 4) for c_, x in zip(codes, rc)}}
    out["names"] = {p["code"]: p.get("name") for p in picks}
    out["stock_vol_ann"] = {c_: round(float(s * np.sqrt(ANN)), 3) for c_, s in zip(codes, sd)}
    out["stock_beta"] = {c_: round(float(b), 2) for c_, b in zip(codes, beta)}
    return out


def render_md(rep):
    L = [f"## 投組風險報告（{rep['label']}，資料至 {rep['asof']}，回看 {rep['lookback_days']} 日）", "",
         f"- 平均兩兩相關 **{rep['avg_pair_corr']:.2f}**；有效獨立賭注數 **{rep['eff_bets_pr']:.1f}**（參與比）/ {rep['eff_bets_entropy']:.1f}（熵）／ {rep['n']} 檔；"
         f"產業集中度 HHI {rep['industry_hhi']:.2f}（{rep['industries']}）", "",
         "| 加權 | 年化波動 | 1日 VaR95 | 1日 CVaR95 | 歷史最大回撤 | 分散化比率 | Beta | 最大權重 |", "|---|---|---|---|---|---|---|---|"]
    for k, s in rep["schemes"].items():
        f = lambda x, p=True: "NA" if x is None else (f"{x * 100:.2f}%" if p else f"{x:.2f}")
        L.append(f"| {s['name']} | {f(s['vol_ann'])} | {f(s['var95_1d'])} | {f(s['cvar95_1d'])} | {f(s['mdd_hist'])} | {f(s['div_ratio'], False)} | {f(s['beta'], False)} | {f(s['max_weight'])} |")
    L += ["", "> 風險估計用過去報酬（共變異數向等相關收縮 30%），僅描述；加權方式是否採用依護欄回測決定（見網站）。⚠️ 僅供研究，不構成投資建議。"]
    return "\n".join(L)


def _latest_pred(kind):
    from .predict import PRED_DIR
    import glob
    fs = sorted(glob.glob(str(PRED_DIR / ("preopen_*.json" if kind == "preopen" else "2*.json"))))
    for f in fs[::-1]:
        o = json.loads(open(f).read())
        if kind == "preopen" and o.get("void"):
            continue
        return o
    return None


def daily_report(write=True):
    """對最新收盤模型/盤前模型的 Top-N 做投組風險報告 → reports/risk/risk_<pred_date>_<kind>.json/.md"""
    out = {}
    RISK_DIR.mkdir(parents=True, exist_ok=True)
    for kind in ("close", "preopen"):
        o = _latest_pred(kind)
        if not o:
            continue
        top = o["predictions"][: o["top_n"]]
        rep = portfolio_report(top, label=kind, asof=o["pred_date"])
        rep["pred_date"], rep["target_date"] = o["pred_date"], o["target_date"]
        rep["picks"] = [{k: p.get(k) for k in ("rank", "code", "name", "industry", "prob")} for p in top]
        out[kind] = rep
        if write:
            (RISK_DIR / f"risk_{o['pred_date']}_{kind}.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
            (RISK_DIR / f"risk_{o['pred_date']}_{kind}.md").write_text(
                f"# {o['pred_date']} → {o['target_date']} {kind} Top-{len(top)} 投組風險\n\n" + render_md(rep) + "\n")
    return out


# ------------------------------------------------------------------ 加權/選股方式的護欄回測
def _picks_by_day(oos, w, top_n, pool=20):
    from . import weights as W
    days = {}
    for d, g in oos.groupby(level="date"):
        g = g[g.next_oc.notna()]
        if len(g) < 50:
            continue
        a = W.adjust(g.droplevel("date"), w, pool)
        days[d] = a.head(pool)
    return days


def eval_weighting(oos, w, top_n=10, lookback=60):
    """oos: walk-forward OOS (index date,code; prob,next_oc,next_excess,r1,industry,...)。回傳各方式的護欄指標。"""
    from . import model as M
    con = db.connect()
    start = (pd.to_datetime(oos.index.get_level_values("date").min()) - pd.Timedelta(days=int(lookback * 2))).strftime("%Y-%m-%d")
    px = pd.read_sql_query("SELECT date,code,ret FROM prices WHERE date>=?", con, params=(start,))
    con.close()
    R = px[px.code.str.fullmatch(r"[1-9]\d{3}")].pivot(index="date", columns="code", values="ret").clip(-0.2, 0.2)
    days = _picks_by_day(oos, w, top_n)
    rows = {k: [] for k in list(SCHEMES) + ["select_min_corr"]}
    ebets, avgc, dates = [], [], []
    for d, g in days.items():
        top = g.head(top_n)
        codes = list(top.index)
        hist = R.loc[:d].tail(lookback + 1).iloc[:-1] if False else R.loc[:d].tail(lookback)   # 含 d 日（收盤後已知）
        Rh = hist.reindex(columns=codes)
        if Rh.notna().sum().max() < 30:
            continue
        ws = weights_for(Rh)
        for k, wt in ws.items():
            rows[k].append((d, float((top.next_oc.values * wt).sum()), float((top.next_excess.values * wt).sum())))
        # 低相關選股：top-20 池挑 10
        pool = list(g.index)
        ch = select_min_corr(pool, R.loc[:d].tail(lookback), k=top_n)
        sel = g.loc[ch]
        rows["select_min_corr"].append((d, float(sel.next_oc.mean()), float(sel.next_excess.mean())))
        C = Rh.corr().values
        n = len(codes)
        avgc.append(float((np.nan_to_num(C).sum() - n) / (n * (n - 1))))
        ebets.append(eff_bets(C)[0])
        dates.append(d)
    res = {}
    for k, v in rows.items():
        if not v:
            continue
        r = pd.DataFrame(v, columns=["date", "oc", "ex"]).set_index("date")
        net = r.oc - M.COST_ROUNDTRIP
        half = len(r) // 2
        res[k] = {"name": SCHEMES.get(k, "低相關選股（top-20 池挑 10）"), "days": int(len(r)), "net_oc_bp": float(net.mean() * 1e4),
                  "excess_bp": float(r.ex.mean() * 1e4), "auc_oc": 0.0, "net_h1_bp": float(net.iloc[:half].mean() * 1e4),
                  "net_h2_bp": float(net.iloc[half:].mean() * 1e4), "vol_bp": float(net.std() * 1e4),
                  "sharpe": float(net.mean() / net.std() * np.sqrt(ANN)) if net.std() > 0 else None, "mdd": max_drawdown(net),
                  "period": f"{r.index.min()}~{r.index.max()}"}
    res["_picks_diag"] = {"avg_pair_corr": float(np.mean(avgc)) if avgc else None,
                          "eff_bets_pr": float(np.mean(ebets)) if ebets else None, "n_days": len(dates)}
    return res


def run_eval():
    """加權/選股方式 × (90 日護欄窗, 2025-07 起長窗)。checkpoint: models/risk_weight_eval.json"""
    from . import features as F, model as M, tune as T, weights as W
    path = MODEL_DIR / "risk_weight_eval.json"
    done = json.loads(path.read_text()) if path.exists() else {}
    X = F.build()
    w = W.load()
    lab = sorted(X[X.y.notna()].index.get_level_values("date").unique())
    start = lab[-T.EVAL_DAYS]
    feats = F.FEATURE_SETS[w.get("feature_set", "base")]
    cols = [c for c in ("r1", "industry", "log_value20", "gap_expect") if c in X]
    if "g90" not in done:
        done["g90"] = eval_weighting(T._oos(X, w, start), w)
        path.write_text(json.dumps(done, ensure_ascii=False, indent=1))
    if "long" not in done:
        oos, _ = M.walk_forward(X, step=42, start="2025-07-01", feats=feats, label="y", verbose=False, w=w)
        done["long"] = eval_weighting(oos.join(X[cols]), w)
        path.write_text(json.dumps(done, ensure_ascii=False, indent=1))
    ch = done["g90"].get("equal")
    for k, r in done["g90"].items():
        if k.startswith("_") or k == "equal":
            continue
        ok, gain, msg = T._verdict(ch, r)
        r["ok"], r["msg"] = ok, msg + "（排序未變、AUC 不適用）"
        lg = done["long"].get(k)
        if lg:
            e = done["long"]["equal"]
            lg["vs_equal_net_bp"] = lg["net_oc_bp"] - e["net_oc_bp"]
            lg["vs_equal_excess_bp"] = lg["excess_bp"] - e["excess_bp"]
    path.write_text(json.dumps(done, ensure_ascii=False, indent=1))
    return done


def decision_text(out):
    parts = []
    fe = out.get("feature_eval") or {}
    for r in fe.get("tune", []):
        parts.append(f"風險特徵（{'收盤後' if r['model'] == 'close' else '盤前'}模型）：{'短窗通過護欄但長窗矛盾→不啟用' if r.get('ok') else '未通過護欄→不啟用'}（{r.get('msg', '')}）。")
    we = out.get("weight_eval") or {}
    g = (we or {}).get("g90") or {}
    items = {k: v for k, v in g.items() if not k.startswith("_") and v.get("ok") is not None}
    on = [v["name"] for v in items.values() if v.get("ok") and v.get("enabled") is not False]
    half = [v["name"] for v in items.values() if v.get("ok") and v.get("enabled") is False]
    bad = [v["name"] for v in items.values() if not v.get("ok")]
    if items:
        parts.append("加權/選股：" + ("啟用 " + "、".join(on) + "；" if on else "沒有任何方式被啟用；") +
                     ("短窗通過但長窗反向、不啟用：" + "、".join(half) + "；" if half else "") +
                     ("未通過護欄：" + "、".join(bad) + "。" if bad else "") + "每日名單仍為等權 Top-10，加權只作為投組風險報告的參考。")
    dv = (out.get("diversification") or {}).get("system")
    if dv:
        parts.append(f"系統自己 3 條流平均相關 {dv['avg_corr']:.2f}、有效獨立賭注僅 {dv['pr']:.1f}/3；與大盤也高度連動，距離 Dalio 的 15 條不相關流很遠。")
    return " ".join(parts)


def payload():
    """docs/data/risk.json"""
    out, _ = streams_report()
    ev = MODEL_DIR / "risk_weight_eval.json"
    out["weight_eval"] = json.loads(ev.read_text()) if ev.exists() else None
    rf = MODEL_DIR / "risk_eval.json"
    if rf.exists():
        d = json.loads(rf.read_text())
        out["feature_eval"] = {"tune": [{"model": k.split("|")[0], "set": k.split("|")[2], **v} for k, v in d.items() if "|tune|" in k],
                               "long": [{"model": k.split("|")[0], "set": k.split("|")[2], **v} for k, v in d.items() if "|long|" in k]}
    for r in (out.get("feature_eval") or {}).get("tune", []):
        r["enabled"] = False                      # 短窗通過者也須長窗不矛盾才啟用（見 CHANGELOG）
        lg = [x for x in out["feature_eval"]["long"] if x["model"] == r["model"]]
        if r.get("ok"):
            r["msg"] += "；但長窗(2025-07起) AUC/RankIC 反而略降 → 短長矛盾，維持僅報告、不啟用"
    wg = (out.get("weight_eval") or {})
    for k, r in (wg.get("g90") or {}).items():
        if k.startswith("_") or not r.get("ok"):
            continue
        lg = (wg.get("long") or {}).get(k) or {}
        if lg.get("vs_equal_net_bp", 0) <= 0 or lg.get("vs_equal_excess_bp", 0) <= 0:
            r["enabled"] = False
            r["msg"] += f"；長窗(2025-07起) 淨 {lg.get('vs_equal_net_bp', 0):+.1f} bp、超額 {lg.get('vs_equal_excess_bp', 0):+.1f} bp 反向 → 不啟用"
    out["feature_names"] = {"+gs_risk": "+gs-quant 風險特徵(13)", "preopen_gs_risk": "盤前+gs-quant 風險特徵(13)"}
    out["decision"] = decision_text(out)
    try:
        out["today"] = daily_report(write=False)
    except Exception as e:
        print("daily_report failed:", e)
        out["today"] = {}
    out["generated"] = dt.datetime.now().isoformat(timespec="minutes")
    return out


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    if cmd == "eval":
        run_eval()
    elif cmd == "streams":
        o, df = streams_report()
        print(json.dumps({k: o[k] for k in ("period", "n_days_common", "diversification")}, ensure_ascii=False, indent=1))
    else:
        r = daily_report()
        for k, v in r.items():
            print(render_md(v))
