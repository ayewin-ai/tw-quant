"""Extra market-level features: futures, world markets, event risk, plus pre-open (overnight) variants.

Timing (prediction for target day T is made after Taiwan close of day t = previous trading day):
  base model  (run ~21:30 Taiwan on t):  TAIFEX day session t, 三大法人期貨 t, P/C t; Asian closes on t
               (Nikkei/KOSPI/HSI/Shanghai all close before 16:00 Taiwan); Europe/US/FX/commodities: last bar dated < t;
               GDELT 1.0 daily event files: UTC days < t (file for day d is published ~06:00 UTC d+1); Google News zh-TW: Taiwan day t-1 (day t counts are biased while the day is in progress); quakes before t 13:30 UTC.
  pre-open    (run ~07:00-08:30 Taiwan on T): adds TAIFEX night session (row dated T, 15:00 t -> 05:00 T),
               US/Europe/FX/commodity bars dated t, quakes before T 00:00 UTC.
"""
import datetime as dt

import numpy as np
import pandas as pd

from . import db

FUT = ["tx_day_ret", "tx_basis", "tx_vol_ratio", "fini_oi", "fini_oi_chg", "fini_oi_chg5", "trust_oi_chg",
       "pc_vol", "pc_oi", "pc_oi_chg"]
GLOBAL = ["nikkei_r", "kospi_r", "hsi_r", "shanghai_r", "nikkei_r5", "kospi_r5", "dax_r", "stoxx50_r",
          "es_r", "nq_r", "oil_r", "gold_r", "dxy_r", "us10y_chg", "usdtwd_r", "usdtwd_r5", "es_r5"]
EVENTS = ["g_taiwan_strait_z", "g_war_z", "g_sanctions_tariff_z", "g_disaster_z", "g_emergency_z",
          "g_taiwan_strait_tone", "g_war_tone", "n_地緣政治_z", "n_制裁關稅_z", "n_氣候災害_z", "n_突發事件_z",
          "quake_max72", "quake_days_since"]
STOCK_EXT = ["beta60", "geo_x_ship", "disaster_x_constr", "quake_x_semi", "risk_x_beta"]
PREOPEN = ["tx_night_ret", "tx_night_vol", "us_sox_on", "us_tsm_on", "us_ndx_on", "us_spx_on", "us_vix_on",
           "es_on", "dax_on", "usdtwd_on", "oil_on", "g_risk_on", "quake_max_on",
           "night_x_beta", "sox_x_tech", "tsm_x_semi", "gap_expect"]

INFO = {
    "tx_day_ret": ("期貨", "台指期日盤漲跌"), "tx_basis": ("期貨", "台指期基差(期/現-1)"),
    "tx_vol_ratio": ("期貨", "台指期成交量比"), "fini_oi": ("期貨", "外資台指期淨未平倉(萬口)"),
    "fini_oi_chg": ("期貨", "外資淨未平倉日變化"), "fini_oi_chg5": ("期貨", "外資淨未平倉5日變化"),
    "trust_oi_chg": ("期貨", "投信淨未平倉日變化"), "pc_vol": ("期貨", "選擇權P/C成交量比"),
    "pc_oi": ("期貨", "選擇權P/C未平倉比"), "pc_oi_chg": ("期貨", "P/C未平倉比變化"),
    "nikkei_r": ("國際股市", "日經當日"), "kospi_r": ("國際股市", "韓股當日"), "hsi_r": ("國際股市", "恆生當日"),
    "shanghai_r": ("國際股市", "上證當日"), "nikkei_r5": ("國際股市", "日經5日"), "kospi_r5": ("國際股市", "韓股5日"),
    "dax_r": ("國際股市", "德股(前一日)"), "stoxx50_r": ("國際股市", "歐洲50(前一日)"),
    "es_r": ("國際股市", "S&P期貨(前一日)"), "nq_r": ("國際股市", "那指期貨(前一日)"),
    "es_r5": ("國際股市", "S&P期貨5日"), "oil_r": ("國際股市", "原油(前一日)"), "gold_r": ("國際股市", "黃金(前一日)"),
    "dxy_r": ("國際股市", "美元指數(前一日)"), "us10y_chg": ("國際股市", "美10年殖利率變化"),
    "usdtwd_r": ("國際股市", "美元/台幣(前一日)"), "usdtwd_r5": ("國際股市", "美元/台幣5日"),
    "g_taiwan_strait_z": ("地緣政治", "GDELT台海新聞量Z"), "g_war_z": ("地緣政治", "GDELT戰爭新聞量Z"),
    "g_sanctions_tariff_z": ("地緣政治", "GDELT制裁關稅新聞量Z"), "g_disaster_z": ("氣候災害", "GDELT災害新聞量Z"),
    "g_emergency_z": ("突發事件", "GDELT突發事件新聞量Z"), "g_taiwan_strait_tone": ("地緣政治", "GDELT台海新聞語調"),
    "g_war_tone": ("地緣政治", "GDELT戰爭新聞語調"), "n_地緣政治_z": ("地緣政治", "中文新聞量:地緣政治Z"),
    "n_制裁關稅_z": ("地緣政治", "中文新聞量:制裁關稅Z"), "n_氣候災害_z": ("氣候災害", "中文新聞量:氣候災害Z"),
    "n_突發事件_z": ("突發事件", "中文新聞量:突發事件Z"), "quake_max72": ("氣候災害", "72小時內台灣附近最大地震規模"),
    "quake_days_since": ("氣候災害", "距上次M5+地震天數"),
    "beta60": ("技術-波動", "60日Beta"), "geo_x_ship": ("地緣政治", "地緣風險×航運/油電"),
    "disaster_x_constr": ("氣候災害", "災害新聞×營建/航運"), "quake_x_semi": ("氣候災害", "地震×半導體/光電"),
    "risk_x_beta": ("地緣政治", "事件風險×Beta"),
    "tx_night_ret": ("期貨夜盤", "台指期夜盤漲跌"), "tx_night_vol": ("期貨夜盤", "夜盤成交量比"),
    "us_sox_on": ("美股", "費半(預測後當晚)"), "us_tsm_on": ("美股", "台積電ADR(當晚)"),
    "us_ndx_on": ("美股", "那指(當晚)"), "us_spx_on": ("美股", "S&P500(當晚)"), "us_vix_on": ("美股", "VIX(當晚)"),
    "es_on": ("美股", "S&P期貨(當晚)"), "dax_on": ("國際股市", "德股(當日收盤)"), "usdtwd_on": ("國際股市", "美元/台幣(當晚)"),
    "oil_on": ("國際股市", "原油(當晚)"), "g_risk_on": ("地緣政治", "GDELT事件風險(當日)"),
    "quake_max_on": ("氣候災害", "開盤前24小時最大地震"), "night_x_beta": ("期貨夜盤", "夜盤×Beta(預期跳空)"),
    "sox_x_tech": ("美股", "費半×電子股"), "tsm_x_semi": ("美股", "台積電ADR×半導體"),
    "gap_expect": ("期貨夜盤", "預期開盤跳空(夜盤×Beta+產業)"),
}

TECH = {"半導體", "電子零組件", "電腦及週邊", "光電", "通信網路", "其他電子", "電子通路", "資訊服務"}
SEMI = {"半導體", "光電", "電子零組件"}
SHIP = {"航運", "油電燃氣"}
CONSTR = {"建材營造", "航運", "觀光餐旅"}


def _z(s, w=60):
    m, sd = s.rolling(w, min_periods=10).mean(), s.rolling(w, min_periods=10).std()
    return (s - m) / sd.replace(0, np.nan)


def _asof(series: pd.Series, days, strict):
    """Value of a date-indexed series as of each day (strict: date < day, else <= day)."""
    s = series.dropna().sort_index()
    idx = pd.to_datetime(s.index)
    out = []
    for d in pd.to_datetime(days):
        pos = idx.searchsorted(d, side="left" if strict else "right") - 1
        out.append(s.iloc[pos] if pos >= 0 else np.nan)
    return pd.Series(out, index=days)


def market(days, taiex_close: pd.Series, next_day=None):
    con = db.connect()
    q = lambda sql: pd.read_sql(sql, con)
    tx = q("SELECT * FROM taifex_tx")
    ti = q("SELECT * FROM taifex_inst")
    pcr = q("SELECT * FROM taifex_pcr").set_index("date")
    us = q("SELECT * FROM us_daily")
    gd = q("SELECT * FROM gdelt_daily")
    nc = q("SELECT * FROM news_counts")
    qk = q("SELECT * FROM quakes")
    con.close()
    M = pd.DataFrame(index=days)
    # ---- futures (day session t)
    day = tx[tx.session == "day"].set_index("date").reindex(days)
    night = tx[tx.session == "night"].set_index("date")
    M["tx_day_ret"] = day["change"] / (day["close"] - day["change"])
    M["tx_basis"] = day["settle"].fillna(day["close"]) / taiex_close.reindex(days) - 1
    M["tx_vol_ratio"] = day["volume"] / day["volume"].rolling(20, min_periods=5).mean()
    fo = ti[ti.who == "foreign"].set_index("date")["net_oi"].reindex(days)
    tr = ti[ti.who == "trust"].set_index("date")["net_oi"].reindex(days)
    M["fini_oi"] = fo / 1e4
    M["fini_oi_chg"] = fo.diff() / 1e4
    M["fini_oi_chg5"] = fo.diff(5) / 1e4
    M["trust_oi_chg"] = tr.diff() / 1e4
    M["pc_vol"] = pcr["pc_vol"].reindex(days)
    M["pc_oi"] = pcr["pc_oi"].reindex(days)
    M["pc_oi_chg"] = M["pc_oi"].diff()
    # ---- world markets
    ret = us.pivot(index="date", columns="symbol", values="ret")
    cls = us.pivot(index="date", columns="symbol", values="close")

    def r(sym, strict):
        return _asof(ret[sym], days, strict) if sym in ret else pd.Series(np.nan, index=days)

    def r5(sym, strict):
        if sym not in cls:
            return pd.Series(np.nan, index=days)
        c = cls[sym].dropna()
        return _asof(c / c.shift(5) - 1, days, strict)
    for s in ("nikkei", "kospi", "hsi", "shanghai"):  # Asian closes before 16:00 Taiwan on t
        M[f"{s}_r"] = r(s, strict=False)
    M["nikkei_r5"], M["kospi_r5"] = r5("nikkei", False), r5("kospi", False)
    for s, name in (("dax", "dax_r"), ("stoxx50", "stoxx50_r"), ("es_fut", "es_r"), ("nq_fut", "nq_r"),
                    ("oil", "oil_r"), ("gold", "gold_r"), ("dxy", "dxy_r"), ("usdtwd", "usdtwd_r")):
        M[name] = r(s, strict=True)
    M["es_r5"], M["usdtwd_r5"] = r5("es_fut", True), r5("usdtwd", True)
    if "us10y" in cls:
        c = cls["us10y"].dropna()
        M["us10y_chg"] = _asof(c.diff(), days, True)
    # ---- events
    gw = gd.pivot(index="date", columns="topic", values="volume").sort_index() if len(gd) else pd.DataFrame()
    gt = gd.pivot(index="date", columns="topic", values="tone").sort_index() if len(gd) else pd.DataFrame()
    for t in ("taiwan_strait", "war", "sanctions_tariff", "disaster", "emergency"):
        M[f"g_{t}_z"] = _asof(_z(gw[t]), days, True) if t in gw else np.nan
    for t in ("taiwan_strait", "war"):
        M[f"g_{t}_tone"] = _asof(gt[t], days, True) if t in gt else np.nan
    nw = nc.pivot(index="date", columns="topic", values="n").sort_index() if len(nc) else pd.DataFrame()
    for t in ("地緣政治", "制裁關稅", "氣候災害", "突發事件"):
        M[f"n_{t}_z"] = _asof(_z(nw[t].astype(float)), days, True) if t in nw else np.nan
    qt = pd.to_datetime(qk.time_utc, utc=True) if len(qk) else pd.Series([], dtype="datetime64[ns, UTC]")

    def qmax(cut, hours):
        if not len(qk):
            return 0.0
        m = (qt < cut) & (qt >= cut - pd.Timedelta(hours=hours))
        return float(qk.mag[m.values].max()) if m.any() else 0.0
    cuts = [pd.Timestamp(d + " 13:30", tz="UTC") for d in days]                      # 21:30 Taiwan on t
    M["quake_max72"] = [qmax(c, 72) for c in cuts]
    M["quake_days_since"] = [((c - qt[qt < c].max()).days if (qt < c).any() else 999) for c in cuts]
    # ---- pre-open (known by ~08:00 Taiwan on T, the next trading day)
    nxt = pd.Series(list(days[1:]) + [next_day], index=days)  # T for each t (None for last -> filled in predict)
    M["tx_night_ret"] = [(_night_ret(night, T)) for T in nxt]
    M["tx_night_vol"] = [(night["volume"].get(T, np.nan) if T else np.nan) for T in nxt]
    M["tx_night_vol"] = M["tx_night_vol"] / M["tx_night_vol"].rolling(20, min_periods=5).mean()
    for s, name in (("sox", "us_sox_on"), ("tsm_adr", "us_tsm_on"), ("nasdaq", "us_ndx_on"), ("spx", "us_spx_on"),
                    ("es_fut", "es_on"), ("dax", "dax_on"), ("usdtwd", "usdtwd_on"), ("oil", "oil_on")):
        # bars dated exactly t (US session t ends 04:00/05:00 Taiwan on T); missing = holiday (0) except latest day
        v = ret[s].reindex(days) if s in ret else pd.Series(np.nan, index=days)
        if len(v) > 1:
            v.iloc[:-1] = v.iloc[:-1].fillna(0)
        M[name] = v
    M["us_vix_on"] = _asof(cls["vix"], days, False) if "vix" in cls else np.nan
    M["g_risk_on"] = np.nan
    if len(gw):
        risk = sum(_z(gw[t]).fillna(0) for t in ("taiwan_strait", "war", "sanctions_tariff") if t in gw)
        M["g_risk_on"] = _asof(risk, days, True)  # GDELT daily file for t is only published 14:00 Taiwan on T
    M["quake_max_on"] = [qmax(pd.Timestamp(d, tz="UTC") + pd.Timedelta(days=1), 24) for d in days]  # to 08:00 TW T
    return M


def _night_ret(night, T):
    if T is None or T not in night.index:
        return np.nan
    row = night.loc[T]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    c, ch = row["close"], row["change"]
    return ch / (c - ch) if c and ch is not None and (c - ch) else np.nan


def stock_level(X: pd.DataFrame):
    ind = X["industry"].fillna("")
    tech = ind.isin(TECH).astype(float)
    semi = ind.isin(SEMI).astype(float)
    geo = X[["g_taiwan_strait_z", "g_war_z", "g_sanctions_tariff_z"]].fillna(0).sum(axis=1) + \
        X[["n_地緣政治_z", "n_制裁關稅_z"]].fillna(0).sum(axis=1)
    X["geo_x_ship"] = geo * ind.isin(SHIP)
    X["disaster_x_constr"] = (X["g_disaster_z"].fillna(0) + X["n_氣候災害_z"].fillna(0)) * ind.isin(CONSTR)
    X["quake_x_semi"] = X["quake_max72"].fillna(0) * semi
    X["risk_x_beta"] = geo * X["beta60"]
    X["night_x_beta"] = X["tx_night_ret"] * X["beta60"]
    X["sox_x_tech"] = X["us_sox_on"] * tech
    X["tsm_x_semi"] = X["us_tsm_on"] * semi
    X["gap_expect"] = X["night_x_beta"].fillna(0) + 0.5 * X["sox_x_tech"].fillna(0)
    return X
