"""網站「商品期貨與匯率因子」區塊資料 → docs/data/market_factors.json"""
import json

import numpy as np
import pandas as pd

from . import db, features_cmdty as CM
from .config import MODEL_DIR

LIST = [("商品-貴金屬", "gold", "黃金 GC=F"), ("商品-貴金屬", "silver", "白銀 SI=F"), ("商品-貴金屬", "platinum", "白金 PL=F"),
        ("商品-工業", "copper", "銅 HG=F"), ("商品-能源", "oil", "WTI 原油 CL=F"), ("商品-能源", "brent", "布蘭特 BZ=F"),
        ("商品-能源", "natgas", "天然氣 NG=F"), ("商品-農產", "corn", "玉米 ZC=F"), ("商品-農產", "soybean", "黃豆 ZS=F"),
        ("商品-農產", "wheat", "小麥 ZW=F"), ("匯率", "usdtwd", "美元/台幣 TWD=X"), ("匯率", "dxy", "美元指數 DX-Y.NYB"),
        ("匯率", "usdjpy", "美元/日圓 JPY=X"), ("匯率", "usdcny", "美元/人民幣 CNY=X"), ("匯率", "eurusd", "歐元/美元 EURUSD=X"),
        ("匯率", "usdkrw", "美元/韓元 KRW=X"), ("利率/風險", "us10y", "美10年殖利率 ^TNX"), ("利率/風險", "vix", "VIX")]
NAMES = {"close": "收盤後模型", "preopen": "盤前模型", "+cmdty": "＋商品匯率（全）", "+cmdty_lite": "＋商品匯率（精簡：綜合＋產業beta）",
         "preopen_cmdty": "＋商品匯率（全）", "preopen_cmdty_lite": "＋商品匯率（精簡）", "base": "基本（現行）", "preopen": "盤前（現行）"}


def moves():
    con = db.connect()
    us = pd.read_sql("SELECT date,symbol,close FROM us_daily", con)
    con.close()
    cls = us.pivot(index="date", columns="symbol", values="close")
    out = []
    for grp, s, lab in LIST:
        if s not in cls:
            continue
        c = cls[s].dropna()
        if len(c) < 22:
            continue
        out.append({"group": grp, "sym": s, "name": lab, "date": c.index[-1], "last": float(c.iloc[-1]),
                    "d1": float(c.iloc[-1] / c.iloc[-2] - 1), "d5": float(c.iloc[-1] / c.iloc[-6] - 1),
                    "d20": float(c.iloc[-1] / c.iloc[-21] - 1)})
    return out


def betas():
    """各產業對各因子的最新 120 日滾動 beta（產業超額報酬 vs 因子報酬）。"""
    con = db.connect()
    px = pd.read_sql("SELECT date,code,ret FROM prices WHERE date >= date((SELECT MAX(date) FROM prices),'-400 day')", con)
    info = pd.read_sql("SELECT code,industry FROM stock_info", con)
    idx = pd.read_sql("SELECT date,pct FROM index_daily WHERE name='TAIEX'", con).set_index("date").pct
    con.close()
    px = px[px.code.str.fullmatch(r"[1-9]\d{3}")]
    r = px.pivot(index="date", columns="code", values="ret").clip(-0.2, 0.2)
    days = list(r.index)
    ind = info.set_index("code").industry.reindex(r.columns).fillna("其他")
    M, fs, fo = CM.market(days)
    se = CM.sector_exp(days, r.fillna(0), idx.reindex(days), ind, fs, fo)
    n = ind.value_counts()
    keep = [i for i in n.index if n[i] >= 8 and i not in ("存託憑證", "其他")]
    rows = []
    for i in keep:
        row = {"industry": i, "n": int(n[i])}
        for f in CM.BETA_F:
            b = se[f"_beta_{f}"][i].iloc[-1]
            row[f] = None if pd.isna(b) else round(float(b), 3)
        rows.append(row)
    return rows


def payload():
    ev = MODEL_DIR / "cmdty_eval.json"
    d = json.loads(ev.read_text()) if ev.exists() else {}
    tune = [{"model": k.split("|")[0], "set": k.split("|")[2], **v} for k, v in d.items() if "|tune|" in k]
    champ = [{"model": k.split("|")[0], "set": k.split("|")[2], **v} for k, v in d.items() if "|champion|" in k]
    long_ = [{"model": k.split("|")[0], "set": k.split("|")[2], **v} for k, v in d.items() if "|long|" in k]
    passed = [f"{t['model']}:{t['set']}" for t in tune if t.get("ok")]
    try:
        bt = betas()
    except Exception as e:
        print("betas failed:", e)
        bt = []
    return {"moves": moves(), "betas": bt, "beta_names": CM.BETA_F, "beta_window": CM.BETA_WIN,
            "tune": tune, "champion": champ, "long": long_, "names": NAMES, "passed": passed,
            "decision": d.get("decision") or (("通過護欄並啟用：" + "、".join(passed)) if passed else
                         ("沒有任何一組通過護欄，維持只記錄、不影響每日名單。" if tune else "評估尚未完成。")),
            "sources": "Yahoo Finance chart API 日K（us_daily 表；3 年回補）。商品=美東該日結算價，FX=紐約收盤，皆早於台灣隔日開盤；"
                       "收盤後模型只用日期<t 的資料，盤前模型再加日期=t 的資料。"}
