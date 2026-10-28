"""Feature engineering. All features for date t use only information available after t's close."""
import datetime as dt

import numpy as np
import pandas as pd

from . import db
from .config import MIN_AVG_VALUE, MIN_PRICE

# feature -> (category, 中文說明)
FEATURE_INFO = {
    "r1": ("技術-動能", "當日報酬"), "r5": ("技術-動能", "5日報酬"), "r10": ("技術-動能", "10日報酬"),
    "r20": ("技術-動能", "20日報酬"), "r60": ("技術-動能", "60日報酬"),
    "ma5_ratio": ("技術-均線", "收盤/MA5"), "ma20_ratio": ("技術-均線", "收盤/MA20"),
    "ma60_ratio": ("技術-均線", "收盤/MA60"), "ma5_ma20": ("技術-均線", "MA5/MA20"),
    "rsi14": ("技術-震盪", "RSI14"), "vol20": ("技術-波動", "20日波動率"),
    "hl_range": ("技術-K線", "當日振幅"), "close_pos": ("技術-K線", "收盤在高低區間位置"),
    "gap": ("技術-K線", "開盤跳空"), "body": ("技術-K線", "K棒實體(收/開)"),
    "upper_shadow": ("技術-K線", "上影線"), "dist_high20": ("技術-位置", "距20日高點"),
    "dist_high60": ("技術-位置", "距60日高點"), "dist_low20": ("技術-位置", "距20日低點"),
    "limit_up": ("技術-K線", "漲停"), "limit_down": ("技術-K線", "跌停"),
    "up_days10": ("技術-動能", "10日上漲天數"),
    "vol_ratio5_20": ("量能", "5日/20日均量比"), "val_z": ("量能", "成交值Z分數"),
    "log_value20": ("量能", "20日平均成交值(log)"), "log_price": ("量能", "股價(log)"),
    "f_ratio": ("籌碼-外資", "外資買賣超/成交量"), "f5": ("籌碼-外資", "外資5日買賣超/均量"),
    "f20": ("籌碼-外資", "外資20日買賣超/均量"), "f_streak": ("籌碼-外資", "外資連買(賣)天數"),
    "t_ratio": ("籌碼-投信", "投信買賣超/成交量"), "t5": ("籌碼-投信", "投信5日買賣超/均量"),
    "t20": ("籌碼-投信", "投信20日買賣超/均量"), "t_streak": ("籌碼-投信", "投信連買(賣)天數"),
    "d_ratio": ("籌碼-自營", "自營商買賣超/成交量"), "d5": ("籌碼-自營", "自營商5日買賣超/均量"),
    "m_chg1": ("籌碼-融資券", "融資餘額日變化"), "m_chg5": ("籌碼-融資券", "融資餘額5日變化"),
    "s_chg5": ("籌碼-融資券", "融券餘額5日變化"), "s_m_ratio": ("籌碼-融資券", "券資比"),
    "m_to_vol": ("籌碼-融資券", "融資餘額/均量"),
    "mkt_r1": ("大盤", "加權指數當日報酬"), "mkt_r5": ("大盤", "加權指數5日報酬"),
    "mkt_ma20": ("大盤", "加權指數/MA20"), "mkt_ma60": ("大盤", "加權指數/MA60"),
    "mkt_vol20": ("大盤", "加權指數20日波動"), "otc_r1": ("大盤", "櫃買指數當日報酬"),
    "otc_r5": ("大盤", "櫃買指數5日報酬"), "breadth": ("大盤", "上漲家數比例"),
    "mkt_f_net": ("大盤", "外資全市場買賣超(標準化)"),
    "us_sox": ("美股", "費半前一晚報酬"), "us_tsm": ("美股", "台積電ADR前一晚報酬"),
    "us_ndx": ("美股", "那斯達克前一晚報酬"), "us_vix": ("美股", "VIX"),
    "rel_r1": ("相對強弱", "相對大盤當日"), "rel_r5": ("相對強弱", "相對大盤5日"),
    "sec_r1": ("產業", "產業當日平均報酬"), "sec_r5": ("產業", "產業5日平均報酬"),
    "rel_sec_r5": ("產業", "相對產業5日"),
    "rev_yoy": ("營收", "最新月營收年增率"), "rev_mom": ("營收", "最新月營收月增率"),
    "rev_yoy_chg": ("營收", "營收年增率變化"), "rev_yoy3": ("營收", "近3月營收年增率平均"),
    "rev_fresh": ("營收", "營收公布後天數"),
    "pe": ("評價", "本益比"), "dow": ("日曆", "星期"),
    "rk_r1": ("相對強弱", "當日報酬排名"), "rk_r5": ("相對強弱", "5日報酬排名"),
    "rk_r20": ("相對強弱", "20日報酬排名"), "rk_vr": ("量能", "量比排名"), "rk_f5": ("籌碼-外資", "外資5日排名"),
}
from . import features_ext as FX

BASE_FEATURES = list(FEATURE_INFO)
FEATURE_INFO.update(FX.INFO)
from . import flows as FL  # noqa: E402
FEATURE_INFO.update(FL.INFO)
from . import features_guru as GU  # noqa: E402
FEATURE_INFO.update(GU.INFO)
from . import features_risk as FR  # noqa: E402
FEATURE_INFO.update(FR.INFO)
from . import features_cmdty as CM  # noqa: E402
FEATURE_INFO.update(CM.INFO)
from . import features_ta as TA  # noqa: E402
FEATURE_INFO.update(TA.INFO)
FEATURE_SETS = {
    "base": BASE_FEATURES,
    "+futures": BASE_FEATURES + FX.FUT,
    "+global": BASE_FEATURES + FX.GLOBAL,
    "+events": BASE_FEATURES + FX.EVENTS + FX.STOCK_EXT,
    "all_base": BASE_FEATURES + FX.FUT + FX.GLOBAL + FX.EVENTS + FX.STOCK_EXT,
    "preopen": BASE_FEATURES + FX.FUT + FX.GLOBAL + FX.EVENTS + FX.STOCK_EXT + FX.PREOPEN,
    "+etf_index": BASE_FEATURES + [c for c in FL.ETF_F if c not in ("etf_w", "etf_n")],
    "+conf": BASE_FEATURES + FL.CONF_F,
    "+tech_events": BASE_FEATURES + FL.TECH_F,
    "+gov_funds": BASE_FEATURES + FL.NAT_F,
    "all_flow": BASE_FEATURES + [c for c in FL.ALL_F if c not in ("etf_w", "etf_n")],
    "+academic_px": BASE_FEATURES + GU.PRICE_F,
    "+rev_anom": BASE_FEATURES + GU.REV_F,
    "+fundamental": BASE_FEATURES + GU.FUND_F,
    "+guru": BASE_FEATURES + GU.GURU_F,
    "all_guru": BASE_FEATURES + GU.ALL_F,
    "+gs_risk": BASE_FEATURES + FR.RISK_F,
    "preopen_gs_risk": BASE_FEATURES + FX.FUT + FX.GLOBAL + FX.EVENTS + FX.STOCK_EXT + FX.PREOPEN + FR.RISK_F,
    "+cmdty": BASE_FEATURES + CM.CMDTY_F,
    "+cmdty_lite": BASE_FEATURES + CM.CMDTY_LITE,
    "preopen_cmdty": BASE_FEATURES + FX.FUT + FX.GLOBAL + FX.EVENTS + FX.STOCK_EXT + FX.PREOPEN + CM.CMDTY_F + CM.CMDTY_ON,
    "preopen_cmdty_lite": BASE_FEATURES + FX.FUT + FX.GLOBAL + FX.EVENTS + FX.STOCK_EXT + FX.PREOPEN + CM.CMDTY_LITE + CM.CMDTY_LITE_ON,
    "+ta": BASE_FEATURES + TA.TA_F,
    "+signal": BASE_FEATURES + TA.SIG_F,
    "+qr": BASE_FEATURES + TA.QR_F,
    "all_ml4t": BASE_FEATURES + TA.ALL_F,
    "preopen_flow": BASE_FEATURES + FX.FUT + FX.GLOBAL + FX.EVENTS + FX.STOCK_EXT + FX.PREOPEN +
                    [c for c in FL.ALL_F if c not in ("etf_w", "etf_n")],
}
_PRE = FEATURE_SETS["preopen_cmdty"]   # 盤前現行冠軍（含商品/匯率）
FEATURE_SETS.update({
    "preopen_ta": _PRE + TA.TA_F,
    "preopen_signal": _PRE + TA.SIG_F,
    "preopen_qr": _PRE + TA.QR_F,
    "preopen_ml4t": _PRE + TA.ALL_F,
})
FEATURES = BASE_FEATURES  # default (overridden by config of the trained model)


def category(feat):
    return FEATURE_INFO.get(feat, ("其他", feat))[0]


def label(feat):
    return FEATURE_INFO.get(feat, ("其他", feat))[1]


def _wide(df, col):
    return df.pivot(index="date", columns="code", values=col).sort_index()


def _streak(x: pd.DataFrame):
    s = np.sign(x.fillna(0)).values
    out = np.zeros_like(s)
    for i in range(len(s)):
        if i == 0:
            out[i] = s[i]
        else:
            same = (s[i] == np.sign(out[i - 1])) & (s[i] != 0)
            out[i] = np.where(same, out[i - 1] + s[i], s[i])
    return pd.DataFrame(out, index=x.index, columns=x.columns)


def load(start=None, end=None):
    con = db.connect()
    w = []
    if start:
        w.append(f"date>='{start}'")
    if end:
        w.append(f"date<='{end}'")
    where = (" WHERE " + " AND ".join(w)) if w else ""
    px = pd.read_sql(f"SELECT * FROM prices{where}", con)
    inst = pd.read_sql(f"SELECT * FROM inst{where}", con)
    mg = pd.read_sql(f"SELECT * FROM margin{where}", con)
    idx = pd.read_sql(f"SELECT * FROM index_daily{where}", con)
    us = pd.read_sql("SELECT * FROM us_daily", con)
    rev = pd.read_sql("SELECT * FROM revenue", con)
    info = pd.read_sql("SELECT * FROM stock_info", con)
    con.close()
    return px, inst, mg, idx, us, rev, info


def build(start=None, end=None, with_labels=True, min_hist=None, next_day=None):
    import os
    min_hist = min_hist or int(os.environ.get("TWQ_MIN_HIST", 40))
    px, inst, mg, idx, us, rev, info = load(start, end)
    px = px[px.code.str.fullmatch(r"[1-9]\d{3}")]
    # trading days = days where TAIEX exists and TWSE prices exist
    days = sorted(set(px.date) & set(idx[idx.name == "TAIEX"].date))
    px = px[px.date.isin(days)]
    close = _wide(px, "close").reindex(days)
    opn = _wide(px, "open").reindex(days)
    high = _wide(px, "high").reindex(days)
    low = _wide(px, "low").reindex(days)
    vol = _wide(px, "volume").reindex(days)
    val = _wide(px, "value").reindex(days)
    ret = _wide(px, "ret").reindex(days)
    pe = _wide(px, "pe").reindex(days)
    # fallback return when change missing
    cff = close.ffill()
    ret_fb = close / cff.shift(1) - 1
    ret = ret.fillna(ret_fb).clip(-0.5, 1.0)
    traded = (vol > 0) & close.notna()
    r = ret.where(traded, 0.0).fillna(0.0)
    adj = (1 + r).cumprod()
    prevclose = cff.shift(1)
    # reference price (ex-div adjusted prev close) = close/(1+ret)
    ref = (close / (1 + ret)).where(traded)

    f = {}
    f["r1"] = ret.where(traded)
    for n in (5, 10, 20, 60):
        f[f"r{n}"] = adj / adj.shift(n) - 1
    ma5, ma20, ma60 = adj.rolling(5).mean(), adj.rolling(20).mean(), adj.rolling(60).mean()
    f["ma5_ratio"] = adj / ma5 - 1
    f["ma20_ratio"] = adj / ma20 - 1
    f["ma60_ratio"] = adj / ma60 - 1
    f["ma5_ma20"] = ma5 / ma20 - 1
    up = r.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-r.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    f["rsi14"] = 100 * up / (up + dn)
    f["vol20"] = r.rolling(20).std()
    f["hl_range"] = (high - low) / ref
    f["close_pos"] = ((close - low) / (high - low)).where(high > low, 0.5)
    f["gap"] = opn / ref - 1
    f["body"] = close / opn - 1
    f["upper_shadow"] = (high - np.maximum(opn, close)) / ref
    f["dist_high20"] = adj / adj.rolling(20).max() - 1
    f["dist_high60"] = adj / adj.rolling(60).max() - 1
    f["dist_low20"] = adj / adj.rolling(20).min() - 1
    f["limit_up"] = (ret >= 0.095).astype(float)
    f["limit_down"] = (ret <= -0.095).astype(float)
    f["up_days10"] = (r > 0).rolling(10).sum()
    v0 = val.fillna(0)
    av5, av20 = v0.rolling(5).mean(), v0.rolling(20).mean()
    f["vol_ratio5_20"] = av5 / av20
    lv = np.log1p(v0)
    f["val_z"] = (lv - lv.rolling(20).mean()) / lv.rolling(20).std()
    f["log_value20"] = np.log1p(av20)
    f["log_price"] = np.log(cff)
    avgvol20 = vol.fillna(0).rolling(20).mean().replace(0, np.nan)

    # chips
    inst = inst[inst.code.isin(close.columns) & inst.date.isin(days)]
    fn = _wide(inst, "foreign_net").reindex(index=days, columns=close.columns)
    tn = _wide(inst, "trust_net").reindex(index=days, columns=close.columns)
    dn_ = _wide(inst, "dealer_net").reindex(index=days, columns=close.columns)
    have_inst = fn.notna().any(axis=1)
    fn0, tn0, dn0 = fn.fillna(0), tn.fillna(0), dn_.fillna(0)
    volnz = vol.replace(0, np.nan)
    f["f_ratio"] = fn0 / volnz
    f["f5"] = fn0.rolling(5).sum() / avgvol20
    f["f20"] = fn0.rolling(20).sum() / avgvol20
    f["f_streak"] = _streak(fn0)
    f["t_ratio"] = tn0 / volnz
    f["t5"] = tn0.rolling(5).sum() / avgvol20
    f["t20"] = tn0.rolling(20).sum() / avgvol20
    f["t_streak"] = _streak(tn0)
    f["d_ratio"] = dn0 / volnz
    f["d5"] = dn0.rolling(5).sum() / avgvol20
    for k in ("f_ratio", "f5", "f20", "f_streak", "t_ratio", "t5", "t20", "t_streak", "d_ratio", "d5"):
        f[k] = f[k].where(have_inst, np.nan, axis=0) if False else f[k]

    mg = mg[mg.code.isin(close.columns) & mg.date.isin(days)]
    mb = _wide(mg, "margin_bal").reindex(index=days, columns=close.columns)
    sb = _wide(mg, "short_bal").reindex(index=days, columns=close.columns)
    mbf, sbf = mb.ffill(limit=2), sb.ffill(limit=2)
    f["m_chg1"] = mbf / mbf.shift(1).replace(0, np.nan) - 1
    f["m_chg5"] = mbf / mbf.shift(5).replace(0, np.nan) - 1
    f["s_chg5"] = (sbf - sbf.shift(5)) / mbf.replace(0, np.nan)
    f["s_m_ratio"] = sbf / mbf.replace(0, np.nan)
    f["m_to_vol"] = mbf * 1000 / avgvol20

    # market
    ti = idx[idx.name == "TAIEX"].set_index("date").reindex(days)
    oi = idx[idx.name == "TPEX"].set_index("date").reindex(days)
    tclose = ti["close"]
    mkt_r1 = ti["pct"].fillna(tclose.pct_change())
    otc_r1 = oi["pct"]
    mkt = pd.DataFrame(index=days)
    mkt["mkt_r1"] = mkt_r1
    mkt["mkt_r5"] = tclose / tclose.shift(5) - 1
    mkt["mkt_ma20"] = tclose / tclose.rolling(20).mean() - 1
    mkt["mkt_ma60"] = tclose / tclose.rolling(60).mean() - 1
    mkt["mkt_vol20"] = mkt_r1.rolling(20).std()
    mkt["otc_r1"] = otc_r1
    mkt["otc_r5"] = oi["close"] / oi["close"].shift(5) - 1
    mkt["breadth"] = (f["r1"] > 0).sum(axis=1) / f["r1"].notna().sum(axis=1).replace(0, np.nan)
    tot_f = fn0.sum(axis=1)
    mkt["mkt_f_net"] = (tot_f - tot_f.rolling(60, min_periods=10).mean()) / tot_f.rolling(60, min_periods=10).std()
    # US: last US session that closed before Taiwan close of day t (US date < t)
    usw = us.pivot(index="date", columns="symbol", values="ret").sort_index()
    usc = us.pivot(index="date", columns="symbol", values="close").sort_index()
    for col, name in (("sox", "us_sox"), ("tsm_adr", "us_tsm"), ("nasdaq", "us_ndx")):
        mkt[name] = [usw[col][usw.index < d].iloc[-1] if col in usw and (usw.index < d).any() else np.nan for d in days]
    mkt["us_vix"] = [usc["vix"][usc.index < d].iloc[-1] if "vix" in usc and (usc.index < d).any() else np.nan for d in days]
    mkt["dow"] = [dt.date.fromisoformat(d).weekday() for d in days]

    # sector
    ind = info.set_index("code")["industry"].reindex(close.columns).fillna("未知")
    sec_r1 = f["r1"].T.groupby(ind).transform("mean").T
    sec_r5 = f["r5"].T.groupby(ind).transform("mean").T
    f["sec_r1"] = sec_r1
    f["sec_r5"] = sec_r5
    f["rel_sec_r5"] = f["r5"] - sec_r5
    f["rel_r1"] = f["r1"].sub(mkt["mkt_r1"], axis=0)
    f["rel_r5"] = f["r5"].sub(mkt["mkt_r5"], axis=0)
    f["pe"] = pe
    mk = mkt["mkt_r1"]
    rm = r.mul(mk, axis=0).rolling(60, min_periods=30).mean()
    cov = rm - r.rolling(60, min_periods=30).mean().mul(mk.rolling(60, min_periods=30).mean(), axis=0)
    f["beta60"] = cov.div(mk.rolling(60, min_periods=30).var(ddof=0), axis=0).clip(-1, 4)

    # revenue: month M available from day 11 of month M+1
    rev = rev[rev.code.isin(close.columns)].copy()
    rev["avail"] = pd.to_datetime(rev.ym + "-01") + pd.offsets.MonthBegin(1) + pd.Timedelta(days=10)
    rev = rev.sort_values(["code", "ym"])
    rev["yoy_prev"] = rev.groupby("code").yoy.shift(1)
    rev["yoy3"] = rev.groupby("code").yoy.transform(lambda s: s.rolling(3, min_periods=1).mean())
    rev_feats = {}
    if len(rev):
        dd = pd.DataFrame({"date": pd.to_datetime(days)})
        out = []
        for code, g in rev.groupby("code"):
            g = g.sort_values("avail")
            m = pd.merge_asof(dd, g[["avail", "yoy", "mom", "yoy_prev", "yoy3"]], left_on="date", right_on="avail")
            m["code"] = code
            out.append(m)
        rv = pd.concat(out)
        rv["date"] = rv["date"].dt.strftime("%Y-%m-%d")
        rv["rev_fresh"] = (pd.to_datetime(rv["date"]) - rv["avail"]).dt.days
        stale = rv["rev_fresh"] > 75
        rv.loc[stale, ["yoy", "mom", "yoy_prev", "yoy3", "rev_fresh"]] = np.nan
        rv["rev_yoy_chg"] = rv["yoy"] - rv["yoy_prev"]
        for a, b in (("yoy", "rev_yoy"), ("mom", "rev_mom"), ("rev_yoy_chg", "rev_yoy_chg"), ("yoy3", "rev_yoy3"),
                     ("rev_fresh", "rev_fresh")):
            rev_feats[b] = rv.pivot(index="date", columns="code", values=a).reindex(index=days, columns=close.columns)
    for b in ("rev_yoy", "rev_mom", "rev_yoy_chg", "rev_yoy3", "rev_fresh"):
        f[b] = rev_feats.get(b, pd.DataFrame(np.nan, index=days, columns=close.columns)).clip(-500, 2000) \
            if b != "rev_fresh" else rev_feats.get(b, pd.DataFrame(np.nan, index=days, columns=close.columns))

    # academic / guru factors (twq/features_guru.py); never break the pipeline
    try:
        ind_g = info.set_index("code")["industry"].reindex(close.columns).fillna("未知")
        f.update(GU.price_feats(adj, r, mkt_r1, ind_g))
        f.update(GU.rev_feats(rev, days, close.columns))
        f.update(GU.fund_feats(days, close.columns, close, f))
    except Exception as e:
        print("guru features unavailable:", e)
    try:  # gs-quant 風格風險/技術指標 (twq/features_risk.py)
        f.update(FR.risk_feats(adj, r, mkt_r1, ind))
    except Exception as e:
        print("risk features unavailable:", e)
    try:  # 第三批：技術指標庫 / 訊號處理 / 統計套利 (twq/features_ta.py)
        av20_ = val.fillna(0).rolling(20).mean()
        f.update(TA.ta_feats(adj, close, opn, high, low, vol, val, r, traded))
        f.update(TA.sig_feats(adj, r, traded))
        f.update(TA.qr_feats(adj, r, traded, f["beta60"], ind, av20_, mkt_r1))
        _reg = TA.mkt_regime(mkt_r1, tclose)
        for _c in _reg.columns:
            mkt[_c] = _reg[_c]
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("ta features unavailable:", e)
    for c in GU.ALL_F + FR.RISK_F + [c for c in TA.ALL_F if not c.startswith("qr_mkt")]:
        if c not in f:
            f[c] = pd.DataFrame(np.nan, index=days, columns=close.columns)

    # universe
    hist = traded.astype(int).rolling(60, min_periods=1).sum()
    univ = traded & (av20 >= MIN_AVG_VALUE) & (close >= MIN_PRICE) & (hist >= min_hist)
    # cross-sectional ranks within universe
    for src, k in (("r1", "rk_r1"), ("r5", "rk_r5"), ("r20", "rk_r20"), ("vol_ratio5_20", "rk_vr"), ("f5", "rk_f5")):
        f[k] = f[src].where(univ).rank(axis=1, pct=True)

    # stack
    stacked = {k: v.where(univ).stack(future_stack=True) for k, v in f.items()}
    X = pd.DataFrame(stacked)
    X = X[univ.stack(future_stack=True).reindex(X.index).fillna(False).values]
    X.index.names = ["date", "code"]
    X = X.join(mkt, on="date")
    try:
        X = X.join(FX.market(days, tclose, next_day), on="date")
    except Exception as e:  # extra tables missing -> NaN features
        print("ext features unavailable:", e)
        for c in FX.FUT + FX.GLOBAL + FX.EVENTS + FX.PREOPEN:
            X[c] = np.nan
    X["industry"] = ind.reindex(X.index.get_level_values("code")).values
    X["close"] = close.stack(future_stack=True).reindex(X.index).values
    X["value"] = val.stack(future_stack=True).reindex(X.index).values
    X = FX.stock_level(X)
    X = CM.add_features(X, days, r, mkt_r1, ind.reindex(r.columns))
    try:
        X = FL.add_features(X)
    except Exception as e:  # never break the pipeline
        print("flow features unavailable:", e)
        for c in FL.ALL_F:
            X[c] = np.nan

    if with_labels:
        nxt_ret = ret.where(traded).shift(-1)
        nxt_oc = (close / opn - 1).where(traded).shift(-1)
        nxt_mkt = mkt_r1.shift(-1)
        X["next_ret"] = nxt_ret.stack(future_stack=True).reindex(X.index).values
        X["next_oc"] = nxt_oc.stack(future_stack=True).reindex(X.index).values
        nm = nxt_mkt.reindex(X.index.get_level_values("date")).values
        X["next_mkt"] = nm
        X["next_excess"] = X["next_ret"] - nm
        med = X.groupby(level="date")["next_ret"].transform("median")
        X["next_xs"] = X["next_ret"] - med
        # label: beat the cross-sectional median next day (pure stock selection, market-neutral)
        X["y"] = (X["next_xs"] > 0).astype(float).where(X["next_ret"].notna())
        med_oc = X.groupby(level="date")["next_oc"].transform("median")
        X["y_oc"] = (X["next_oc"] > med_oc).astype(float).where(X["next_oc"].notna())
    X = X.replace([np.inf, -np.inf], np.nan).sort_index()
    return X


def trading_days():
    con = db.connect()
    d = [r[0] for r in con.execute(
        "SELECT DISTINCT p.date FROM prices p JOIN index_daily i ON i.date=p.date AND i.name='TAIEX' "
        "WHERE p.market='TWSE' ORDER BY 1")]
    con.close()
    return d


def next_trading_day(d: dt.date):
    """下一個交易日：人事行政總處日曆 + 證交所假日表 + 臨時休市記錄 + 實際資料（見 twq/calendar.py）。"""
    from . import calendar
    return calendar.next_trading_day(d)
