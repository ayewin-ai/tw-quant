"""Daily prediction: features as of DATE close -> next trading day picks with factor attributions."""
import datetime as dt
import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import db, features as F, model as M
from .config import PRED_DIR, PREDICT_REPORT_DIR, MODEL_DIR, TOP_N, DISCLAIMER, REPORT_DIR, PREDICT_SET, PREDICT_LABEL, LIMITUP_MODE

PCT_FEATS = {"r1", "r5", "r10", "r20", "r60", "ma5_ratio", "ma20_ratio", "ma60_ratio", "ma5_ma20", "vol20",
             "hl_range", "gap", "body", "upper_shadow", "dist_high20", "dist_high60", "dist_low20", "f_ratio",
             "t_ratio", "d_ratio", "m_chg1", "m_chg5", "s_chg5", "s_m_ratio", "mkt_r1", "mkt_r5", "mkt_ma20",
             "mkt_ma60", "mkt_vol20", "otc_r1", "otc_r5", "breadth", "us_sox", "us_tsm", "us_ndx", "rel_r1",
             "rel_r5", "sec_r1", "sec_r5", "rel_sec_r5", "rk_r1", "rk_r5", "rk_r20", "rk_vr", "rk_f5"}


PCT_FEATS |= {"tx_day_ret", "tx_basis", "nikkei_r", "kospi_r", "hsi_r", "shanghai_r", "nikkei_r5", "kospi_r5", "dax_r",
              "stoxx50_r", "es_r", "nq_r", "oil_r", "gold_r", "dxy_r", "usdtwd_r", "usdtwd_r5", "es_r5", "tx_night_ret",
              "us_sox_on", "us_tsm_on", "us_ndx_on", "us_spx_on", "es_on", "dax_on", "usdtwd_on", "oil_on",
              "night_x_beta", "sox_x_tech", "tsm_x_semi", "gap_expect"}


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


def get_model(X, date: str, fset=None, label=None, tag="model"):
    import os
    from . import weights as W
    w = W.load()
    fset = fset or (os.environ.get("TWQ_SET") or w.get("feature_set") or PREDICT_SET)
    label = label or PREDICT_LABEL
    feats = F.FEATURE_SETS[fset]
    path = MODEL_DIR / f"{tag}_{fset}_{label}_{date}_{W.model_hash(w)}.txt"
    if path.exists():
        return lgb.Booster(model_file=str(path)), json.loads(path.with_suffix(".json").read_text())
    tr = X[(X.index.get_level_values("date") < date) & X[label].notna()]
    b = M.train(tr, feats, label, w=w)
    b.save_model(str(path))
    meta = {"train_rows": int(len(tr)), "train_start": tr.index.get_level_values("date").min(),
            "train_end": tr.index.get_level_values("date").max(), "rounds": int(w["model"].get("num_rounds", M.N_ROUNDS)),
            "weights_version": w.get("version"), "recency_half_life_days": w.get("recency_half_life_days"),
            "feature_set": fset, "label": label, "n_features": len(feats)}
    path.with_suffix(".json").write_text(json.dumps(meta))
    # keep only the 10 most recent models
    for old in sorted(MODEL_DIR.glob(f"{tag}_*.txt"), key=lambda p: p.stat().st_mtime)[:-10]:
        old.unlink(missing_ok=True)
        old.with_suffix(".json").unlink(missing_ok=True)
    return b, meta


def market_context(X, date):
    row = X.loc[date].iloc[0]
    return {k: (None if pd.isna(row[k]) else float(row[k])) for k in
            ("mkt_r1", "mkt_r5", "mkt_ma20", "mkt_ma60", "otc_r1", "breadth", "us_sox", "us_tsm", "us_ndx", "us_vix")}


def run(date: dt.date = None, update=True, top=None, limitup=None):
    top = top or TOP_N
    limitup = limitup or LIMITUP_MODE
    if update:
        from . import fetch
        end = date or dt.date.today()
        fetch.update(end - dt.timedelta(days=7), end, verbose=False, extras=False)
        fetch.update_extras(end - dt.timedelta(days=40), end, verbose=False)
        from . import fetch_ext
        fetch_ext.update(end - dt.timedelta(days=10), end, verbose=False)
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
    target = F.next_trading_day(dt.date.fromisoformat(ds))
    booster, meta = get_model(X, ds)
    out = score(X, ds, target, booster, meta, top, limitup)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    (PRED_DIR / f"{ds}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    try:
        from . import archive
        archive.archive_prediction(PRED_DIR / f"{ds}.json")
    except Exception as e:
        print("archive error", e)
    md = render(out)
    path = PREDICT_REPORT_DIR / f"predict_{ds}.md"
    path.write_text(md)
    print(md)
    print(f"\n已儲存：{PRED_DIR / (ds + '.json')}\n報告：{path}")
    return out


def score(X, ds, target, booster, meta, top, limitup, mode="base"):
    feats = booster.feature_name()
    Xp = X.xs(ds, level="date", drop_level=False).copy()
    prob = booster.predict(Xp[feats])
    c = booster.predict(Xp[feats], pred_contrib=True)
    contrib, base = pd.DataFrame(c[:, :-1], index=Xp.index, columns=feats), c[:, -1]
    Xp["prob"] = prob
    Xp["limitup_flag"] = (Xp["r1"] >= 0.095)
    from . import weights as W
    w = W.load()
    if limitup == "exclude":
        w["score"]["limitup_penalty"] = 1.0
    elif limitup == "penalize":
        w["score"]["limitup_penalty"] = max(w["score"].get("limitup_penalty", 0), 0.05)
    Xp = W.adjust(Xp, w, top, preopen=(mode == "preopen"))   # tunable post-model adjustments (config/weights.json)
    meta = {**meta, "weights_version": w.get("version"), "score_adjust": w["score"],
            "liquidity_min_avg_value": w.get("liquidity_min_avg_value")}
    con = db.connect()
    names = dict(con.execute("SELECT code,name FROM prices WHERE date=?", (ds,)).fetchall())
    con.close()
    rows = []
    for (d, code), r in Xp.iterrows():
        c = contrib.loc[(d, code)]
        grp = c.groupby(lambda f: F.category(f).split("-")[0]).sum().sort_values(ascending=False)
        rows.append({
            "code": code, "name": names.get(code, ""), "industry": r["industry"], "rank": int(r["rank"]),
            "prob": round(float(r["prob"]), 4), "close": float(r["close"]),
            "r1": None if pd.isna(r["r1"]) else float(r["r1"]),
            "limitup": bool(r["limitup_flag"]),
            "factors_pos": factor_list(c, r, 5, 1), "factors_neg": factor_list(c, r, 5, -1),
            "category_contrib": {k: round(float(v), 4) for k, v in grp.items()},
        })
    out = {
        "pred_date": ds, "target_date": target.isoformat() if target else None,
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "universe_size": len(rows), "top_n": top, "model": meta, "market": market_context(X, ds),
        "base_value": float(np.mean(base)), "disclaimer": DISCLAIMER, "mode": mode, "limitup_mode": limitup,
        "overnight": overnight_context(X, ds),
        "predictions": rows,
    }
    return out


def overnight_context(X, ds):
    row = X.xs(ds, level="date").iloc[0]
    keys = ["tx_day_ret", "tx_basis", "fini_oi", "fini_oi_chg", "pc_oi", "nikkei_r", "kospi_r", "hsi_r",
            "tx_night_ret", "us_sox_on", "us_tsm_on", "us_ndx_on", "es_on", "usdtwd_on", "g_risk_on", "quake_max72",
            "quake_max_on", "oil_on", "cm_gold_on", "cm_silver_on", "cm_platinum_on", "cm_copper_on", "cm_brent_on",
            "cm_natgas_on", "cm_dxy_on", "cm_usdjpy_on", "cm_usdcny_on", "cm_usdkrw_on", "cm_eurusd_on"]
    return {k: (None if k not in row or pd.isna(row[k]) else float(row[k])) for k in keys}


def run_preopen(target: dt.date = None, update=True, top=None, limitup=None):
    """Pre-open re-scoring for target day T using TAIFEX night session + US/Europe close + overnight events."""
    top = top or TOP_N
    limitup = limitup or LIMITUP_MODE
    from zoneinfo import ZoneInfo
    tdays = F.trading_days()
    tw_today = dt.datetime.now(ZoneInfo("Asia/Taipei")).date()
    from . import calendar
    calendar.ensure_fresh()
    calendar.retarget_predictions()
    if target is None:
        # 目標＝台灣『今天』若是交易日且尚無收盤資料；否則（休市日/已有資料）取最後資料日之後的下一交易日。
        if tw_today.isoformat() > tdays[-1] and calendar.is_trading_day(tw_today):
            target = tw_today
        else:
            target = calendar.next_trading_day(dt.date.fromisoformat(tdays[-1]))
            if tw_today.isoformat() > tdays[-1] and target > tw_today:
                print(f"台灣今天 {tw_today} 休市（{calendar.status(tw_today)['why']}），不產生盤前預測；"
                      f"下一個交易日是 {target}。")
                return None
    elif not calendar.is_trading_day(target):
        print(f"目標日 {target} 休市（{calendar.status(target)['why']}），不產生盤前預測。")
        return None
    t = max(d for d in tdays if d < target.isoformat())
    if update:
        from . import fetch, fetch_ext
        fetch_ext.init()
        for fn in (lambda: fetch_ext.taifex_tx(dt.date.fromisoformat(t), target),
                   lambda: fetch_ext.world("5d"), lambda: fetch.us_markets("5d"),
                   lambda: fetch_ext.quakes(dt.date.fromisoformat(t) - dt.timedelta(days=3), target),
                   lambda: fetch_ext.gdelt_events(dt.date.fromisoformat(t) - dt.timedelta(days=3), dt.date.fromisoformat(t))):
            try:
                fn()
            except Exception as e:
                print("preopen update warning:", e)
    X = F.build(end=t, next_day=target.isoformat())
    booster, meta = get_model(X, t, fset=W.load().get("preopen_feature_set") or "preopen", label="y_oc", tag="preopen")
    out = score(X, t, target, booster, meta, top, limitup, mode="preopen")
    if out["overnight"].get("tx_night_ret") is None:
        print("⚠️ 尚未取得台指期夜盤資料（夜盤約台灣 05:00 結束），夜盤特徵為缺值。")
    (PRED_DIR / f"preopen_{target.isoformat()}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    try:
        from . import archive
        archive.archive_prediction(PRED_DIR / f"preopen_{target.isoformat()}.json")
    except Exception as e:
        print("archive error", e)
    md = render(out)
    path = PREDICT_REPORT_DIR / f"preopen_{target.isoformat()}.md"
    path.write_text(md)
    print(md)
    print(f"\n盤前報告：{path}")
    return out





def _overnight_lines(on, pre):
    g = lambda k: on.get(k)
    L = [f"- 台指期：日盤 {_pct(g('tx_day_ret'))}、基差 {_pct(g('tx_basis'))}、外資淨未平倉 "
         f"{(g('fini_oi') or 0) * 1e4:,.0f} 口（日變化 {(g('fini_oi_chg') or 0) * 1e4:+,.0f}）、P/C 未平倉比 {g('pc_oi') or 'NA'}",
         f"- 亞洲：日經 {_pct(g('nikkei_r'))}、韓股 {_pct(g('kospi_r'))}、恆生 {_pct(g('hsi_r'))}",
         f"- 事件：72 小時內台灣附近最大地震 M{g('quake_max72') or 0:.1f}"]
    if pre:
        L.insert(1, f"- **夜盤與隔夜**：台指期夜盤 {_pct(g('tx_night_ret'))}、費半 {_pct(g('us_sox_on'))}、台積電ADR {_pct(g('us_tsm_on'))}、"
                    f"那指 {_pct(g('us_ndx_on'))}、S&P期貨 {_pct(g('es_on'))}、美元/台幣 {_pct(g('usdtwd_on'))}、GDELT 事件風險 {g('g_risk_on') or 'NA'}")
        L.insert(2, f"- **隔夜商品**：黃金 {_pct(g('cm_gold_on'))}、白銀 {_pct(g('cm_silver_on'))}、白金 {_pct(g('cm_platinum_on'))}、"
                    f"銅 {_pct(g('cm_copper_on'))}、WTI {_pct(g('oil_on'))}、布蘭特 {_pct(g('cm_brent_on'))}、天然氣 {_pct(g('cm_natgas_on'))}")
        L.insert(3, f"- **隔夜匯率**：美元/台幣 {_pct(g('usdtwd_on'))}、美元指數 {_pct(g('cm_dxy_on'))}、美元/日圓 {_pct(g('cm_usdjpy_on'))}、"
                    f"美元/人民幣 {_pct(g('cm_usdcny_on'))}、美元/韓元 {_pct(g('cm_usdkrw_on'))}、歐元/美元 {_pct(g('cm_eurusd_on'))}"
                    "（商品=美東結算價、匯率=紐約收盤）")
    else:
        L.append("- 注意：預測後當晚的美股與台指期夜盤尚未反映；請看開盤前的「盤前更新預測」(twq preopen)。")
    L.append("- ⚠️漲停追高：前一日漲停的股票隔日常開高或買不到，歷史上是主要的誤判來源之一。")
    return L


def _pct(x, signed=True):
    return "NA" if x is None else (f"{x * 100:+.2f}%" if signed else f"{x * 100:.1f}%")


def render(o):
    m = o["market"]
    top = o["predictions"][: o["top_n"]]
    pre = o.get("mode") == "preopen"
    title = (f"# 盤前更新預測 — {o['target_date']} 開盤前（含夜盤、美股收盤、隔夜事件）" if pre else
             f"# 台股隔日預測 — 依 {o['pred_date']} 收盤資料，預測 {o['target_date']}")
    L = [title, "", o["disclaimer"], "",
         f"- 產生時間（box 本地時間 America/Bogota, UTC-5）：{o['generated_at']}",
         f"- 預測範圍：{o['universe_size']} 檔流動性足夠的上市櫃普通股（20日均成交值≥5,000萬、股價≥10元）",
         f"- 模型：LightGBM，訓練資料 {o['model']['train_start']} ~ {o['model']['train_end']}，共 {o['model']['train_rows']:,} 筆",
         (f"- 預測目標：{o['target_date']} 「開盤→收盤」報酬高於全體中位數的機率（盤前可執行版本）" if pre else
          "- 預測目標：隔日報酬「高於全體股票中位數」的機率（選股相對強弱，不是絕對漲跌；大盤方向另看市場狀態）"),
         f"- 特徵組：{o['model'].get('feature_set', 'base')}（{o['model'].get('n_features', '')} 個特徵）；漲停股處理：{o.get('limitup_mode', 'flag')}", "",
         "## 市場狀態", "",
         f"- 加權指數：當日 {_pct(m['mkt_r1'])}，5日 {_pct(m['mkt_r5'])}，相對MA20 {_pct(m['mkt_ma20'])}，相對MA60 {_pct(m['mkt_ma60'])}",
         f"- 櫃買指數：當日 {_pct(m['otc_r1'])}；上漲家數比例 {_pct(m['breadth'], False)}",
         f"- 前一晚美股：費半 {_pct(m['us_sox'])}、台積電ADR {_pct(m['us_tsm'])}、那斯達克 {_pct(m['us_ndx'])}、VIX {('%.1f' % m['us_vix']) if m['us_vix'] else 'NA'}",
         *(_overnight_lines(o.get("overnight") or {}, pre)), "",
         f"## Top {o['top_n']} 看好名單", "",
         "| # | 代號 | 名稱 | 產業 | 跑贏機率 | 收盤 | 當日漲跌 | 主要看多因素 | 主要扣分因素 |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in top:
        pos = "；".join(f"{f['label']} {f['value_str']}" for f in r["factors_pos"][:3])
        neg = "；".join(f"{f['label']} {f['value_str']}" for f in r["factors_neg"][:2])
        flag = " ⚠️漲停追高" if r.get("limitup") else ""
        L.append(f"| {r['rank']} | {r['code']} | {r['name']}{flag} | {r['industry']} | {r['prob']:.1%} | {r['close']:g} | "
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
