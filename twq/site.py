"""Build static-site data (docs/data/*.json) from DB, models and reports."""
import datetime as dt
import json
import shutil

import numpy as np
import pandas as pd

from . import db
from .config import ROOT, MODEL_DIR, REPORT_DIR, PRED_DIR, TOP_N, DISCLAIMER

DOCS = ROOT / "docs"
OUT = DOCS / "data"


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if (o is None or np.isnan(o) or np.isinf(o)) else round(float(o), 6)
    if isinstance(o, np.integer):
        return int(o)
    return o


def _dump(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(_clean(obj), ensure_ascii=False, separators=(",", ":")))


def _calendar_payload():
    import json as _j
    from zoneinfo import ZoneInfo
    from . import calendar as cal
    today = dt.datetime.now(ZoneInfo("Asia/Taipei")).date()
    days, lo, hi, _ = cal._db_info()
    st = cal.status(today)
    nxt = cal.next_trading_day(today if not st["open"] else today)  # 今天（台灣）之後的下一個交易日
    ov = _j.loads(cal.OVERRIDES.read_text()) if cal.OVERRIDES.exists() else {}
    unsched = []  # 資料區間內『行事曆說開市但沒資料』= 臨時休市（颱風假等）
    d = dt.date.fromisoformat(lo)
    while d <= dt.date.fromisoformat(hi):
        if d.weekday() < 5 and cal.official_closed(d) is None and d.isoformat() not in days:
            unsched.append(d.isoformat())
        d += dt.timedelta(days=1)
    ovmap = {c["date"]: c for c in ov.get("closures", [])}
    chk = {str(y): cal.cross_check(y) for y in (today.year, today.year + 1)}
    _dump("calendar.json", {
        "today_tw": today.isoformat(), "today_open": st["open"], "today_why": st["why"], "today_source": st["source"],
        "last_data_date": hi, "next_trading_day": nxt.isoformat() if nxt else None,
        "upcoming_closures": cal.upcoming_closures(today, 120),
        "unscheduled_closures": [{"date": x, "reason": ovmap.get(x, {}).get("reason", "資料區間內無成交資料（颱風假或證交所自訂休市，未逐一查證原因）"),
                                  "source": ovmap.get(x, {}).get("source", "推定")} for x in unsched],
        "cross_check": chk,
        "sources": ["人事行政總處 政府行政機關辦公日曆表（data.gov.tw；公開鏡像 ruyut/TaiwanCalendar）",
                    "臺灣證券交易所 市場開休市日期（openapi.twse.com.tw /holidaySchedule/holidaySchedule）",
                    "臨時休市（颱風假等）：依證交所公告事後記入 config/calendar_overrides.json，其餘以實際成交資料為準"]})


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {"generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "disclaimer": DISCLAIMER,
            "top_n": TOP_N}

    # coverage
    con = db.connect()
    cov = {}
    for t in ("prices", "inst", "margin"):
        n, lo, hi, nd = con.execute(f"SELECT COUNT(*),MIN(date),MAX(date),COUNT(DISTINCT date) FROM {t}").fetchone()
        cov[t] = {"rows": n, "start": lo, "end": hi, "days": nd}
    n, lo, hi = con.execute("SELECT COUNT(*),MIN(ym),MAX(ym) FROM revenue").fetchone()
    cov["revenue"] = {"rows": n, "start": lo, "end": hi}
    n, lo, hi = con.execute("SELECT COUNT(*),MIN(date),MAX(date) FROM us_daily").fetchone()
    cov["us_daily"] = {"rows": n, "start": lo, "end": hi}
    cov["stocks"] = con.execute("SELECT COUNT(DISTINCT code) FROM prices").fetchone()[0]
    con.close()
    meta["coverage"] = cov

    # validation
    vpath = MODEL_DIR / "validation.json"
    val = json.loads(vpath.read_text()) if vpath.exists() else {}
    _dump("validation.json", val)

    # equity curves
    bpath = REPORT_DIR / "backtest_daily.csv"
    if bpath.exists():
        b = pd.read_csv(bpath)
        _dump("equity.json", {
            "dates": b.date.tolist(),
            "cc_gross": b.cc_equity.tolist(), "cc_net": b.cc_net_equity.tolist(),
            "oc_gross": b.oc_equity.tolist(), "oc_net": b.oc_net_equity.tolist(), "taiex": b.mkt_equity.tolist(),
        })

    # walk-forward daily top-N hit rate (out of sample)
    for p in (MODEL_DIR / "oos_predictions.parquet", MODEL_DIR / "oos_predictions.csv.gz"):
        if p.exists():
            o = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p, dtype={"code": str})
            if "date" not in o.columns:
                o = o.reset_index()
            o["rank"] = o.groupby("date").prob.rank(ascending=False, method="first")
            tp = o[(o["rank"] <= TOP_N) & o.next_excess.notna()]
            d = tp.groupby("date").agg(hit_ex=("next_excess", lambda s: (s > 0).mean()),
                                       hit_up=("next_ret", lambda s: (s > 0).mean())).reset_index()
            d["cum_hit_ex"] = d.hit_ex.expanding().mean()
            d["cum_hit_up"] = d.hit_up.expanding().mean()
            d["roll20_hit_ex"] = d.hit_ex.rolling(20, min_periods=5).mean()
            base = o[o.next_excess.notna()].groupby("date").next_excess.apply(lambda s: (s > 0).mean())
            d["base_ex"] = base.reindex(d.date).expanding().mean().values
            _dump("hitrate_oos.json", d.to_dict(orient="list"))
            break

    # live performance log
    if (REPORT_DIR / "performance_log.csv").exists():
        pf = pd.read_csv(REPORT_DIR / "performance_log.csv").sort_values("pred_date")
        pf["cum_hit_ex"] = pf.top_hit_excess.expanding().mean()
        pf["cum_hit_up"] = pf.top_hit_abs.expanding().mean()
        pf["cum_excess"] = (1 + pf.top_avg_excess).cumprod() - 1
        _dump("performance_live.json", pf.to_dict(orient="list"))

    # error patterns
    if (REPORT_DIR / "review_log.csv").exists():
        lg = pd.read_csv(REPORT_DIR / "review_log.csv", dtype={"code": str})
        ex = lg.assign(cause=lg.causes.fillna("").str.split(";")).explode("cause")
        tab = ex.groupby(["cause", "kind"]).size().unstack(fill_value=0)
        _dump("error_patterns.json", {"causes": tab.index.tolist(),
                                       "wrong_pick": tab.get("wrong_pick", pd.Series(0, index=tab.index)).tolist(),
                                       "missed_winner": tab.get("missed_winner", pd.Series(0, index=tab.index)).tolist()})

    ab = MODEL_DIR / "ablation.json"
    if ab.exists():
        from .model import SET_NAMES
        d = json.loads(ab.read_text())
        _dump("ablation.json", {"rows": list(d.values()), "names": SET_NAMES})
        for f in MODEL_DIR.glob("ablation_daily_*.csv"):
            b = pd.read_csv(f)
            key = f.stem.replace("ablation_daily_", "")
            _dump(f"ablation_eq_{key.replace('+', 'p')}.json", {"dates": b.date.tolist(), "oc_net": b.oc_net_equity.tolist(),
                                                               "taiex": b.mkt_equity.tolist()})
    ge = MODEL_DIR / "guru_eval.json"
    if ge.exists():
        from .guru_eval import site_payload
        _dump("guru_eval.json", site_payload(json.loads(ge.read_text())))
    pre = sorted(PRED_DIR.glob("preopen_*.json"))
    if pre:
        o = json.loads(pre[-1].read_text())
        top = [{k: r.get(k) for k in ("rank", "code", "name", "industry", "prob", "close", "r1", "limitup")} |
               {"pos": [f"{f['label']} {f['value_str']}" for f in r["factors_pos"][:3]]} for r in o["predictions"][: o["top_n"]]]
        _dump("latest_preopen.json", {"target_date": o["target_date"], "pred_date": o["pred_date"],
                                      "void": o.get("void", False), "void_reason": o.get("void_reason"),
                                      "overnight": o.get("overnight"), "model": o["model"], "top": top})
    try:
        from . import archive
        comp = archive.write_comparison()
        if comp is not None:
            c = comp.reset_index()
            _dump("pred_history.json", {"columns": list(c.columns),
                                        "rows": c.astype(object).where(c.notna(), None).values.tolist()})
    except Exception as e:
        print("pred_history error", e)
    try:
        from . import tune, weights as W
        if tune.LOG.exists():
            lg = pd.read_csv(tune.LOG)
            ch = lg[lg.decision == "champion"].drop_duplicates("date", keep="last")
            keys = ["score.limitup_penalty", "score.gap_risk_penalty", "score.max_per_industry",
                    "liquidity_min_avg_value", "recency_half_life_days"]
            hist = []
            for r in ch.itertuples():
                f = W.HIST / f"weights_v{int(r.version_before)}.json"
                ww = json.loads(f.read_text()) if f.exists() else W.load()
                hist.append({"date": r.date, "version": int(r.version_before), "net_oc_bp": r.net_oc_bp,
                             "excess_bp": r.excess_bp, "auc_oc": r.auc_oc, **{k: W.get(ww, k) for k in keys}})
            dec = lg[lg.decision != "champion"].tail(40).iloc[::-1]
            _dump("tuning.json", {"history": hist, "keys": keys,
                                  "decisions": dec[["date", "decision", "key", "old", "new", "reason"]].astype(object)
                                  .where(dec[["date", "decision", "key", "old", "new", "reason"]].notna(), None).to_dict("records"),
                                  "current": W.load()})
    except Exception as e:
        print("tuning site data error", e)
    # latest prediction
    preds = sorted(p for p in PRED_DIR.glob("*.json") if not p.name.startswith("preopen_"))
    if preds:
        o = json.loads(preds[-1].read_text())
        top = []
        for r in o["predictions"][: o["top_n"]]:
            top.append({k: r.get(k) for k in ("rank", "code", "name", "industry", "prob", "close", "r1", "limitup")} |
                       {"pos": [f"{f['label']} {f['value_str']}" for f in r["factors_pos"][:3]],
                        "neg": [f"{f['label']} {f['value_str']}" for f in r["factors_neg"][:2]]})
        _dump("latest_prediction.json", {"pred_date": o["pred_date"], "target_date": o["target_date"],
                                         "universe_size": o["universe_size"], "market": o["market"],
                                         "model": o["model"], "top": top, "overnight": o.get("overnight"),
                                         "history": [p.stem for p in preds][-60:]})
    # latest review (parse structured data from logs)
    reviews = sorted((REPORT_DIR / "review").glob("review_*.md"))
    if reviews:
        pdate = reviews[-1].stem.replace("review_", "")
        pf = pd.read_csv(REPORT_DIR / "performance_log.csv", dtype={"pred_date": str})
        perf = pf[pf.pred_date == pdate].iloc[0].to_dict() if (pf.pred_date == pdate).any() else {}
        lg = pd.read_csv(REPORT_DIR / "review_log.csv", dtype={"code": str, "pred_date": str})
        cases = lg[lg.pred_date == pdate].to_dict(orient="records")
        # top-N actual results
        topres = []
        pj = PRED_DIR / f"{pdate}.json"
        if pj.exists() and perf.get("target_date"):
            o = json.loads(pj.read_text())
            con = db.connect()
            act = dict(con.execute("SELECT code, ret FROM prices WHERE date=?", (perf["target_date"],)).fetchall())
            con.close()
            for r in o["predictions"][: o["top_n"]]:
                ret = act.get(r["code"])
                topres.append({"rank": r["rank"], "code": r["code"], "name": r["name"], "industry": r["industry"],
                               "prob": r["prob"], "ret": ret,
                               "excess": None if ret is None else ret - (perf.get("taiex") or 0)})
        _dump("latest_review.json", {"pred_date": pdate, "perf": perf, "top": topres, "cases": cases,
                                     "markdown": reviews[-1].name})
        # copy markdown reports for direct viewing
    rep = DOCS / "reports"
    rep.mkdir(parents=True, exist_ok=True)
    for sub in ("predict", "review"):
        (rep / sub).mkdir(exist_ok=True)
        for f in sorted((REPORT_DIR / sub).glob("*.md"))[-90:]:
            shutil.copy2(f, rep / sub / f.name)
    for f in ("validation.md", "error_patterns.md"):
        if (REPORT_DIR / f).exists():
            shutil.copy2(REPORT_DIR / f, rep / f)
    meta["reports"] = {sub: sorted(p.name for p in (rep / sub).glob("*.md")) for sub in ("predict", "review")}
    try:
        from . import grey
        _dump("grey.json", grey.site_payload())
    except Exception as e:  # 影子模式失敗不影響網站
        print("grey site payload failed:", e)
    try:
        from . import lookup
        lookup.build()
    except Exception as e:  # 查詢資料失敗不影響網站
        print("lookup.json failed:", e)
    try:
        _calendar_payload()
    except Exception as e:
        print("calendar.json failed:", e)
    try:
        from . import market_factors
        _dump("market_factors.json", market_factors.payload())
    except Exception as e:
        print("market_factors.json failed:", e)
    try:
        from . import risk as _risk
        _dump("risk.json", _risk.payload())
    except Exception as e:
        print("risk.json failed:", e)
    _dump("meta.json", meta)
    print(f"site data written to {OUT}")
