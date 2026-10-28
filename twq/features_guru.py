"""學術因子＋投資大師選股法則（point-in-time）。
價格類：12-1 動能 (Jegadeesh-Titman / Carhart)、52 週高點 (George-Hwang)、產業動能 (Moskowitz-Grinblatt)、
       特質波動 / 低波動 (Ang et al., BAB Frazzini-Pedersen)、MAX 彩券效應 (Bali-Cakici-Whitelaw)。
營收類（台股月營收）：營收創 12 月新高、營收 SUE（年增率相對自身歷史的標準化驚奇）。
財報類（MOPS 彙總報表，依法定公告期限才可用）：ROE、毛利/資產 (Novy-Marx)、E/P、B/P (Fama-French)、
       資產成長 (FF5 CMA)、EPS 年增 (PEAD/SUE 代理)、Greenblatt 神奇公式、Piotroski F 分數（6 項可得版）、
       Lynch PEG、Graham 數、現金殖利率 (Buffett/價值)、CANSLIM 條件數 (O'Neil)。
"""
import os
import numpy as np
import pandas as pd

from . import db

INFO = {
    "mom12_1": ("學術-動能", "12-1 月動能"), "mom6_1": ("學術-動能", "6-1 月動能"),
    "high52": ("學術-動能", "距 52 週高點"), "ind_mom": ("學術-動能", "產業 6 月動能"),
    "ivol60": ("學術-低波動", "60 日特質波動"), "max21": ("學術-低波動", "21 日最大單日報酬(MAX)"),
    "rev_high12": ("學術-營收", "月營收/近12月最高"), "rev_sue": ("學術-營收", "營收年增率驚奇(SUE)"),
    "roe_ttm": ("大師-品質", "ROE(近四季)"), "gpa": ("大師-品質", "毛利/總資產"),
    "ep": ("大師-價值", "盈餘殖利率 E/P"), "bp": ("大師-價值", "淨值市值比 B/P"),
    "asset_g": ("學術-投資", "總資產年成長"), "eps_yoy": ("學術-盈餘", "單季EPS年增/股價"),
    "magic_rank": ("大師-Greenblatt", "神奇公式排名"), "fscore": ("大師-Piotroski", "F 分數(0-6)"),
    "peg": ("大師-Lynch", "PEG"), "graham": ("大師-Graham", "Graham 數/股價"),
    "div_yield": ("大師-價值", "近一年現金殖利率"), "canslim_n": ("大師-O'Neil", "CANSLIM 條件數"),
}
PRICE_F = ["mom12_1", "mom6_1", "high52", "ind_mom", "ivol60", "max21"]
REV_F = ["rev_high12", "rev_sue"]
FUND_F = ["roe_ttm", "gpa", "ep", "bp", "asset_g", "eps_yoy", "div_yield"]
GURU_F = ["magic_rank", "fscore", "peg", "graham", "canslim_n"]
ALL_F = PRICE_F + REV_F + FUND_F + GURU_F
DEADLINE = {1: "05-15", 2: "08-14", 3: "11-14", 4: "03-31"}


def price_feats(adj, r, mk, ind):
    f = {}
    f["mom12_1"] = adj.shift(21) / adj.shift(250) - 1
    f["mom6_1"] = adj.shift(21) / adj.shift(125) - 1
    f["high52"] = adj / adj.rolling(250, min_periods=120).max() - 1
    r120 = adj / adj.shift(120) - 1
    if os.environ.get("TWQ_SECFIX") == "1":
        r120 = r120.where((r != 0).cummax())
    f["ind_mom"] = r120.T.groupby(ind).transform("mean").T
    m = mk.reindex(r.index)
    cov = (r.mul(m, axis=0).rolling(60, min_periods=30).mean()
           - r.rolling(60, min_periods=30).mean().mul(m.rolling(60, min_periods=30).mean(), axis=0))
    beta = cov.div(m.rolling(60, min_periods=30).var(ddof=0), axis=0)
    resid = r - beta.mul(m, axis=0)
    f["ivol60"] = resid.rolling(60, min_periods=30).std()
    f["max21"] = r.rolling(21).max()
    return f


def _asof_panel(events, days, cols, key="avail"):
    """events: DataFrame(code, avail(Timestamp), cols...). -> {col: wide frame days x code} as-of."""
    dd = pd.DataFrame({"date": pd.to_datetime(days)})
    out = []
    for code, g in events.sort_values(key).groupby("code"):
        m = pd.merge_asof(dd, g[[key] + cols], left_on="date", right_on=key)
        m["code"] = code
        out.append(m)
    if not out:
        return {}
    a = pd.concat(out)
    a["date"] = a["date"].dt.strftime("%Y-%m-%d")
    return {c: a.pivot(index="date", columns="code", values=c) for c in cols}


def rev_feats(rev, days, columns):
    rev = rev[rev.code.isin(columns)].sort_values(["code", "ym"]).copy()
    if rev.empty:
        return {}
    rev["avail"] = pd.to_datetime(rev.ym + "-01") + pd.offsets.MonthBegin(1) + pd.Timedelta(days=10)
    g = rev.groupby("code")
    rev["rev_high12"] = rev.rev / g.rev.transform(lambda s: s.rolling(12, min_periods=6).max())
    mu = g.yoy.transform(lambda s: s.shift(1).rolling(12, min_periods=6).mean())
    sd = g.yoy.transform(lambda s: s.shift(1).rolling(12, min_periods=6).std())
    rev["rev_sue"] = ((rev.yoy - mu) / sd.replace(0, np.nan)).clip(-10, 10)
    p = _asof_panel(rev, days, ["rev_high12", "rev_sue"])
    return {k: v.reindex(index=days, columns=columns) for k, v in p.items()}


def _fin_table():
    con = db.connect()
    try:
        q = pd.read_sql("SELECT * FROM fin_q", con)
        ex = pd.read_sql("SELECT * FROM exdiv", con)
    except Exception:
        q, ex = pd.DataFrame(), pd.DataFrame()
    con.close()
    return q, ex


def fund_feats(days, columns, close, f_base):
    q, ex = _fin_table()
    out = {}
    cl = close.reindex(index=days, columns=columns)
    if len(q):
        q = q.sort_values(["code", "year", "season"]).copy()
        flows = ["revenue", "gross", "op_income", "ni_parent", "eps"]
        prev = q.groupby(["code", "year"])[flows].shift(1)
        same_year_prev = q.groupby(["code", "year"]).season.shift(1) == q.season - 1
        for c in flows:   # YTD -> single quarter
            q[c + "_q"] = np.where(q.season == 1, q[c], np.where(same_year_prev, q[c] - prev[c], np.nan))
        g = q.groupby("code")
        for c in ("ni_parent", "gross", "op_income", "revenue", "eps"):
            q[c + "_ttm"] = g[c + "_q"].transform(lambda s: s.rolling(4, min_periods=4).sum())
        eq = q.equity_parent.fillna(q.equity)
        q["shares"] = (q.ni_parent / q.eps.replace(0, np.nan) * 1000).where(lambda s: s > 0)
        q["shares"] = g.shares.transform(lambda s: s.ffill())
        q["roe_ttm"] = q.ni_parent_ttm / eq.where(eq > 0)
        q["gpa"] = q.gross_ttm / q.assets
        q["asset_g"] = q.assets / g.assets.shift(4) - 1
        q["eps_q_ly"] = g.eps_q.shift(4)
        q["eps_diff"] = q.eps_q - q.eps_q_ly
        q["eps_growth"] = (q.eps_ttm / g.eps_ttm.shift(4) - 1).where(g.eps_ttm.shift(4) > 0)
        q["bvps"] = eq * 1000 / q.shares
        q["ev_debt"] = q.liab.fillna(0) * 1000 - q.cash.fillna(0) * 1000
        q["roc"] = q.op_income_ttm / (q.assets - q.cur_liab.fillna(0)).where(lambda s: s > 0)
        # Piotroski (6 items available from summary statements; no CFO / share issuance)
        roa = q.ni_parent_ttm / q.assets
        lev = q.liab / q.assets
        cr = q.cur_assets / q.cur_liab
        gm = q.gross_ttm / q.revenue_ttm
        at = q.revenue_ttm / q.assets
        sh = lambda s: s.groupby(q.code).shift(4)
        q["fscore"] = ((roa > 0).astype(float) + (roa > sh(roa)) + (lev < sh(lev)) + (cr > sh(cr)) + (gm > sh(gm)) + (at > sh(at)))
        q.loc[sh(roa).isna(), "fscore"] = np.nan
        q["avail"] = pd.to_datetime([(f"{y + 1}-" if s == 4 else f"{y}-") + DEADLINE[s] for y, s in zip(q.year, q.season)]) \
            + pd.Timedelta(days=1)
        cols = ["roe_ttm", "gpa", "asset_g", "eps_diff", "eps_ttm", "eps_growth", "bvps", "shares", "ev_debt", "roc",
                "fscore", "ni_parent_ttm", "op_income_ttm"]
        p = {k: v.reindex(index=days, columns=columns) for k, v in _asof_panel(q, days, cols).items()}
        # stale guard: older than ~200 days -> NaN
        av = _asof_panel(q.assign(av=q.avail), days, ["av"]).get("av")
        if av is not None:
            avw = av.reindex(index=days, columns=columns).apply(pd.to_datetime)
            dts = pd.Series(pd.to_datetime(days), index=days)
            stale = avw.rsub(dts, axis=0).apply(lambda c: c.dt.days) > 200
            p = {k: v.mask(stale) for k, v in p.items()}
        mcap = cl * p["shares"]
        out["roe_ttm"] = p["roe_ttm"].clip(-2, 2)
        out["gpa"] = p["gpa"].clip(-1, 3)
        out["asset_g"] = p["asset_g"].clip(-0.9, 5)
        out["eps_yoy"] = (p["eps_diff"] / cl).clip(-0.5, 0.5)
        out["ep"] = (p["eps_ttm"] / cl).clip(-1, 1)
        out["bp"] = (p["bvps"] / cl).clip(0, 10)
        ey = p["op_income_ttm"] * 1000 / (mcap + p["ev_debt"]).where(lambda s: s > 0)
        out["magic_rank"] = (ey.rank(axis=1, pct=True) + p["roc"].rank(axis=1, pct=True)) / 2
        out["fscore"] = p["fscore"]
        pe_ = cl / p["eps_ttm"].where(p["eps_ttm"] > 0)
        out["peg"] = (pe_ / (p["eps_growth"] * 100).where(p["eps_growth"] > 0)).clip(0, 20)
        gn = np.sqrt((22.5 * p["eps_ttm"] * p["bvps"]).where(lambda s: s > 0))
        out["graham"] = (gn / cl).clip(0, 10)
        rev_yoy = f_base.get("rev_yoy")
        cans = ((p["eps_diff"] > 0).astype(float) + (p["eps_growth"] > 0.25)
                + (f_base["dist_high60"] > -0.05) + ((f_base["vol_ratio5_20"] > 1.2) & (f_base["r5"] > 0))
                + (rev_yoy > 25 if rev_yoy is not None else 0))
        out["canslim_n"] = cans.where(p["eps_ttm"].notna())
    if len(ex):
        ex = ex[ex.code.isin(columns)]
        cash = ex[ex.kind.astype(str).str.contains("息")] if "kind" in ex else ex
        w = cash.pivot_table(index="date", columns="code", values="amount", aggfunc="sum")
        allx = pd.DataFrame(0.0, index=pd.to_datetime(sorted(set(days) | set(w.index))), columns=columns)
        w.index = pd.to_datetime(w.index)
        allx.loc[w.index, w.columns.intersection(columns)] = w[w.columns.intersection(columns)].fillna(0).values
        roll = allx.rolling("365D").sum()
        roll.index = roll.index.strftime("%Y-%m-%d")
        out["div_yield"] = (roll.reindex(index=days) / cl).clip(0, 0.5)
    return out
