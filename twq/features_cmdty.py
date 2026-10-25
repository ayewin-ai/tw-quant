"""商品期貨與匯率因子（Yahoo Finance 日K，twq/fetch_ext.WORLD）。

時間對齊（沒有偷看未來）
  收盤後模型（台灣 t 日 21:30 預測 t+1）：只用「日期 < t」的商品/匯率日K（= 美東 t-1 日已收，台灣 t 日開盤前已知）。
  盤前模型（台灣 T 日 ~07:00–08:30 預測 T）：加用「日期 = t」的日K（美東 t 日結算/紐約收盤，台灣 T 日清晨前已知）。
  商品=美東該日結算價（金屬 13:30 ET、能源 14:30 ET），FX=紐約 17:00 收盤，都早於台灣隔日開盤。
產業敏感度：以「該產業等權報酬 − 大盤」對「已知的商品/匯率報酬」做滾動 120 日單變量迴歸取 beta（只用 t 日以前資料），
  預期漲跌 = beta × 當期因子報酬，再依個股產業貼回（如 原油→塑膠/航運、銅→電器電纜、台幣貶→出口導向）。
"""
import numpy as np
import pandas as pd

from . import db
from .features_ext import _asof

# 新增（us_daily 的 symbol 名稱）
NEW_CM = ["brent", "natgas", "silver", "platinum", "copper", "corn", "soybean", "wheat"]
NEW_FX = ["usdjpy", "usdcny", "eurusd", "usdkrw"]
R5_SYMS = NEW_CM + NEW_FX + ["gold", "oil", "dxy"]          # usdtwd_r5 已存在於 features_ext
ON_SYMS = NEW_CM + NEW_FX + ["gold", "dxy"]                   # oil_on / usdtwd_on 已存在
COMP = {"oilx": ["oil", "brent"], "prec": ["gold", "silver", "platinum"], "agri": ["corn", "soybean", "wheat"],
        "asiafx": ["usdtwd", "usdjpy", "usdcny", "usdkrw"]}
BETA_F = {"oilx": "原油(WTI+Brent)", "natgas": "天然氣", "prec": "貴金屬(金銀鉑)", "copper": "銅",
          "agri": "農產(玉米黃豆小麥)", "usdtwd": "美元/台幣", "asiafx": "美元兌亞幣(台日人韓)", "dxy": "美元指數"}
BETA_WIN = 120

CM_STRICT = [f"cm_{s}_r1" for s in NEW_CM + NEW_FX] + [f"cm_{s}_r5" for s in R5_SYMS] + \
            [f"cm_{c}_r" for c in COMP]
CMB_STRICT = [f"cmb_{f}" for f in BETA_F]
CMDTY_F = CM_STRICT + CMB_STRICT
CM_ON = [f"cm_{s}_on" for s in ON_SYMS] + [f"cm_{c}_on" for c in COMP]
CMB_ON = [f"cmb_{f}_on" for f in BETA_F]
CMDTY_ON = CM_ON + CMB_ON
CMDTY_LITE = [f"cm_{c}_r" for c in COMP] + CMB_STRICT
CMDTY_LITE_ON = [f"cm_{c}_on" for c in COMP] + CMB_ON
ALL_F = CMDTY_F + CMDTY_ON

NAME = {"brent": "布蘭特原油", "natgas": "天然氣", "silver": "白銀", "platinum": "白金", "copper": "銅",
        "corn": "玉米", "soybean": "黃豆", "wheat": "小麥", "usdjpy": "美元/日圓", "usdcny": "美元/人民幣",
        "eurusd": "歐元/美元", "usdkrw": "美元/韓元", "gold": "黃金", "oil": "WTI原油", "dxy": "美元指數",
        "usdtwd": "美元/台幣"}
INFO = {}
for s in NEW_CM + NEW_FX:
    INFO[f"cm_{s}_r1"] = ("商品匯率", f"{NAME[s]}(前一日)")
for s in R5_SYMS:
    INFO[f"cm_{s}_r5"] = ("商品匯率", f"{NAME[s]}5日")
for s in ON_SYMS:
    INFO[f"cm_{s}_on"] = ("商品匯率", f"{NAME[s]}(當晚/盤前)")
for c, lab in (("oilx", "原油綜合"), ("prec", "貴金屬綜合"), ("agri", "農產綜合"), ("asiafx", "美元兌亞幣綜合")):
    INFO[f"cm_{c}_r"] = ("商品匯率", f"{lab}(前一日)")
    INFO[f"cm_{c}_on"] = ("商品匯率", f"{lab}(當晚/盤前)")
for f, lab in BETA_F.items():
    INFO[f"cmb_{f}"] = ("商品匯率-產業", f"{lab}×產業滾動beta(預期漲跌)")
    INFO[f"cmb_{f}_on"] = ("商品匯率-產業", f"{lab}×產業滾動beta(盤前預期漲跌)")


def _load():
    con = db.connect()
    us = pd.read_sql("SELECT date,symbol,close,ret FROM us_daily", con)
    con.close()
    return us.pivot(index="date", columns="symbol", values="ret"), us.pivot(index="date", columns="symbol", values="close")


def market(days):
    """日期層級欄位 (index=days)；回傳 (M, strict_factors, on_factors)。"""
    ret, cls = _load()
    nan = pd.Series(np.nan, index=days)

    def strict(s):
        return _asof(ret[s], days, True) if s in ret else nan

    def on(s):
        if s not in ret:
            return nan
        v = ret[s].reindex(days)
        if len(v) > 1:
            v.iloc[:-1] = v.iloc[:-1].fillna(0)
        return v

    def r5(s, st):
        if s not in cls:
            return nan
        c = cls[s].dropna()
        return _asof(c / c.shift(5) - 1, days, st)
    M = pd.DataFrame(index=days)
    for s in NEW_CM + NEW_FX:
        M[f"cm_{s}_r1"] = strict(s)
    for s in R5_SYMS:
        M[f"cm_{s}_r5"] = r5(s, True)
    for s in ON_SYMS:
        M[f"cm_{s}_on"] = on(s)
    fs, fo = {}, {}
    for c, mem in COMP.items():
        M[f"cm_{c}_r"] = pd.concat([strict(s) for s in mem], axis=1).mean(axis=1, skipna=True)
        M[f"cm_{c}_on"] = pd.concat([on(s) for s in mem], axis=1).mean(axis=1, skipna=True)
    for f in BETA_F:
        if f in COMP:
            fs[f], fo[f] = M[f"cm_{f}_r"], M[f"cm_{f}_on"]
        else:
            fs[f] = strict(f)
            fo[f] = on(f)
    return M, fs, fo


def sector_exp(days, r, mkt_r1, ind, fs, fo):
    """每個(日,產業)的 beta×因子 預期漲跌。回傳 dict name -> DataFrame(days x industry)。"""
    sec = r.T.groupby(ind).mean().T.sub(mkt_r1.reindex(r.index), axis=0)     # 產業超額報酬 (days x ind)
    out = {}
    for f in BETA_F:
        x = fs[f].reindex(sec.index)
        mp = BETA_WIN // 3
        cov = sec.mul(x, axis=0).rolling(BETA_WIN, min_periods=mp).mean() - \
            sec.rolling(BETA_WIN, min_periods=mp).mean().mul(x.rolling(BETA_WIN, min_periods=mp).mean(), axis=0)
        var = x.rolling(BETA_WIN, min_periods=mp).var(ddof=0).replace(0, np.nan)
        beta = cov.div(var, axis=0).clip(-3, 3)
        out[f"cmb_{f}"] = beta.mul(x, axis=0)
        out[f"cmb_{f}_on"] = beta.mul(fo[f].reindex(sec.index), axis=0)
        out[f"_beta_{f}"] = beta
    return out


def add_features(X, days, r, mkt_r1, ind):
    """把商品/匯率欄位併入 X (index date, code)。失敗時整組補 NaN，不影響主流程。"""
    try:
        M, fs, fo = market(days)
        X = X.join(M, on="date")
        se = sector_exp(days, r, mkt_r1, ind, fs, fo)
        di = pd.MultiIndex.from_arrays([X.index.get_level_values("date"), X["industry"].values])
        for k, df in se.items():
            if k.startswith("_beta_"):
                continue
            st = df.stack(future_stack=True)
            st.index.names = ["date", "industry"]
            X[k] = st.reindex(di).values
        return X
    except Exception as e:
        print("cmdty features unavailable:", e)
        for c in ALL_F:
            X[c] = np.nan
        return X
