"""Daily prediction: features as of DATE close -> next trading day picks with factor attributions."""
import datetime as dt
import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import db, features as F, model as M
from .config import PRED_DIR, PREDICT_REPORT_DIR, MODEL_DIR, TOP_N, DISCLAIMER, REPORT_DIR

PCT_FEATS = {"r1", "r5", "r10", "r20", "r60", "ma5_ratio", "ma20_ratio", "ma60_ratio", "ma5_ma20", "vol20",
             "hl_range", "gap", "body", "upper_shadow", "dist_high20", "dist_high60", "dist_low20", "f_ratio",
             "t_ratio", "d_ratio", "m_chg1", "m_chg5", "s_chg5", "s_m_ratio", "mkt_r1", "mkt_r5", "mkt_ma20",
             "mkt_ma60", "mkt_vol20", "otc_r1", "otc_r5", "breadth", "us_sox", "us_tsm", "us_ndx", "rel_r1",
             "rel_r5", "sec_r1", "sec_r5", "rel_sec_r5", "rk_r1", "rk_r5", "rk_r20", "rk_vr", "rk_f5"}


def fmt(feat, v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "NA"
    if feat in PCT_FEATS:
        return f"{v * 100:+.2f}%" if not feat.startswith("rk_") and feat != "breadth" else f"{v * 100:.0f}%"
    if feat in ("rev_yoy", "rev_mom", "rev_yoy_chg", "rev_yoy3"):
        return f"{v:+.1f}%"
    if feat in ("limit_up", "limit_down"):
        return "是" if v > 0 else "否"
    if feat == "log_value20":
        return f"{np.expm1(v) / 1e8:.1f}億"
    if feat == "log_price":
        return f"{np.exp(v):.1f}"
    return f"{v:.2f}"


def factor_list(contrib_row, x_row, n=5, sign=1):
    s = contrib_row.sort_values(ascending=(sign < 0))
    out = []
    for feat, c in s.head(n).items():
        if sign * c <= 0:
            break
        out.append({"feature": feat, "label": F.label(feat), "category": F.category(feat),
                    "value": None if pd.isna(x_row[feat]) else float(x_row[feat]), "value_str": fmt(feat, x_row[feat]),
                    "contrib": round(float(c), 4)})
    return out


def get_model(X, date: str):
    path = MODEL_DIR / f"model_{date}.txt"
    if path.exists():
        return lgb.Booster(model_file=str(path)), json.loads((MODEL_DIR / f"model_{date}.json").read_text())
    tr = X[(X.index.get_level_values("date") < date) & X.y.notna()]
    b = M.train(tr)
    b.save_model(str(path))
    meta = {"train_rows": int(len(tr)), "train_start": tr.index.get_level_values("date").min(),
            "train_end": tr.index.get_level_values("date").max(), "rounds": M.N_ROUNDS}
    (MODEL_DIR / f"model_{date}.json").write_text(json.dumps(meta))
    # keep only the 10 most recent models
    for old in sorted(MODEL_DIR.glob("model_*.txt"))[:-10]:
        old.unlink(missing_ok=True)
        old.with_suffix(".json").unlink(missing_ok=True)
    return b, meta


def market_context(X, date):
    row = X.loc[date].iloc[0]
    return {k: (None if pd.isna(row[k]) else float(row[k])) for k in
            ("mkt_r1", "mkt_r5", "mkt_ma20", "mkt_ma60", "otc_r1", "breadth", "us_sox", "us_tsm", "us_ndx", "us_vix")}


def run(date: dt.date = None, update=True, top=None):
    top = top or TOP_N
    if update:
        from . import fetch
        end = date or dt.date.today()
        fetch.update(end - dt.timedelta(days=7), end, verbose=False, extras=False)
        fetch.update_extras(end - dt.timedelta(days=40), end, verbose=False)
    tdays = F.trading_days()
    if date is None:
        ds = tdays[-1]
    else:
        ds = date.isoformat()
        if ds not in tdays:
            print(f"{ds} 沒有交易資料（休市或資料尚未公布），略過預測。")
            return None
    X = F.build(end=ds)
    if ds not in X.index.get_level_values("date"):
        print(f"{ds} 無可預測股票（資料不足）")
        return None
    booster, meta = get_model(X, ds)
    Xp = X.loc[[ds]].copy() if False else X.xs(ds, level="date", drop_level=False)
    prob = booster.predict(Xp[F.FEATURES])
    contrib, base = contributions_safe(booster, Xp)
    Xp["prob"] = prob
    Xp = Xp.sort_values("prob", ascending=False)
    Xp["rank"] = np.arange(1, len(Xp) + 1)
    con = db.connect()
    names = dict(con.execute("SELECT code,name FROM prices WHERE date=?", (ds,)).fetchall())
    con.close()
    target = F.next_trading_day(dt.date.fromisoformat(ds))
    rows = []
    for (d, code), r in Xp.iterrows():
        c = contrib.loc[(d, code)]
        grp = c.groupby(lambda f: F.category(f).split("-")[0]).sum().sort_values(ascending=False)
        rows.append({
            "code": code, "name": names.get(code, ""), "industry": r["industry"], "rank": int(r["rank"]),
            "prob": round(float(r["prob"]), 4), "close": float(r["close"]),
            "r1": None if pd.isna(r["r1"]) else float(r["r1"]),
            "factors_pos": factor_list(c, r, 5, 1), "factors_neg": factor_list(c, r, 5, -1),
            "category_contrib": {k: round(float(v), 4) for k, v in grp.items()},
        })
    out = {
        "pred_date": ds, "target_date": target.isoformat() if target else None,
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "universe_size": len(rows), "top_n": top, "model": meta, "market": market_context(X, ds),
        "base_value": float(np.mean(base)), "disclaimer": DISCLAIMER,
        "predictions": rows,
    }
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    (PRED_DIR / f"{ds}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    md = render(out)
    path = PREDICT_REPORT_DIR / f"predict_{ds}.md"
    path.write_text(md)
    print(md)
    print(f"\n已儲存：{PRED_DIR / (ds + '.json')}\n報告：{path}")
    return out


def contributions_safe(booster, Xp):
    c = booster.predict(Xp[F.FEATURES], pred_contrib=True)
    return pd.DataFrame(c[:, :-1], index=Xp.index, columns=F.FEATURES), c[:, -1]


def _pct(x, signed=True):
    return "NA" if x is None else (f"{x * 100:+.2f}%" if signed else f"{x * 100:.1f}%")


def render(o):
    m = o["market"]
    top = o["predictions"][: o["top_n"]]
    L = [f"# 台股隔日預測 — 依 {o['pred_date']} 收盤資料，預測 {o['target_date']}", "", o["disclaimer"], "",
         f"- 產生時間（box 本地時間 America/Bogota, UTC-5）：{o['generated_at']}",
         f"- 預測範圍：{o['universe_size']} 檔流動性足夠的上市櫃普通股（20日均成交值≥5,000萬、股價≥10元）",
         f"- 模型：LightGBM，訓練資料 {o['model']['train_start']} ~ {o['model']['train_end']}，共 {o['model']['train_rows']:,} 筆",
         "- 預測目標：隔日報酬「高於全體股票中位數」的機率（選股相對強弱，不是絕對漲跌；大盤方向另看市場狀態）", "",
         "## 市場狀態", "",
         f"- 加權指數：當日 {_pct(m['mkt_r1'])}，5日 {_pct(m['mkt_r5'])}，相對MA20 {_pct(m['mkt_ma20'])}，相對MA60 {_pct(m['mkt_ma60'])}",
         f"- 櫃買指數：當日 {_pct(m['otc_r1'])}；上漲家數比例 {_pct(m['breadth'], False)}",
         f"- 前一晚美股：費半 {_pct(m['us_sox'])}、台積電ADR {_pct(m['us_tsm'])}、那斯達克 {_pct(m['us_ndx'])}、VIX {('%.1f' % m['us_vix']) if m['us_vix'] else 'NA'}",
         "- 注意：預測產生後當晚的美股走勢尚未反映，請於開盤前自行留意（隔日檢討會分析其影響）。", "",
         f"## Top {o['top_n']} 看好名單", "",
         "| # | 代號 | 名稱 | 產業 | 跑贏機率 | 收盤 | 當日漲跌 | 主要看多因素 | 主要扣分因素 |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in top:
        pos = "；".join(f"{f['label']} {f['value_str']}" for f in r["factors_pos"][:3])
        neg = "；".join(f"{f['label']} {f['value_str']}" for f in r["factors_neg"][:2])
        L.append(f"| {r['rank']} | {r['code']} | {r['name']} | {r['industry']} | {r['prob']:.1%} | {r['close']:g} | "
                 f"{_pct(r['r1'])} | {pos} | {neg} |")
    L += ["", "## 各檔因素拆解（SHAP 貢獻，正值＝提高看多機率）", ""]
    for r in top:
        cats = "，".join(f"{k} {v:+.3f}" for k, v in list(r["category_contrib"].items())[:4])
        L.append(f"**{r['rank']}. {r['code']} {r['name']}**（{r['prob']:.1%}）— 類別貢獻：{cats}")
        for f in r["factors_pos"]:
            L.append(f"  - ＋ {f['label']}（{f['category']}）= {f['value_str']}，貢獻 {f['contrib']:+.3f}")
        for f in r["factors_neg"][:3]:
            L.append(f"  - － {f['label']}（{f['category']}）= {f['value_str']}，貢獻 {f['contrib']:+.3f}")
        L.append("")
    bottom = o["predictions"][-5:]
    L += ["## 最不看好 5 檔（供參考）", "", "| 代號 | 名稱 | 機率 | 主要看空因素 |", "|---|---|---|---|"]
    for r in reversed(bottom):
        neg = "；".join(f"{f['label']} {f['value_str']}" for f in r["factors_neg"][:3])
        L.append(f"| {r['code']} | {r['name']} | {r['prob']:.1%} | {neg} |")
    vpath = REPORT_DIR / "validation.md"
    try:
        v = json.loads((MODEL_DIR / "validation.json").read_text())
        L += ["", "## 模型歷史樣本外表現（walk-forward）", "",
              f"- 測試期 {v['test_period'][0]} ~ {v['test_period'][1]}：AUC {v['auc']:.3f}，"
              f"Top-{TOP_N} 勝過加權指數比例 {v[f'top{TOP_N}_precision_excess']:.1%}（全體基準 {v['base_rate_excess_up']:.1%}），"
              f"Top-{TOP_N} 隔日上漲比例 {v[f'top{TOP_N}_precision_abs_up']:.1%}", f"- 詳見 {vpath}"]
    except Exception:
        pass
    return "\n".join(L) + "\n"
