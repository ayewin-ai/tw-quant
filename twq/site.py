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

    # latest prediction
    preds = sorted(PRED_DIR.glob("*.json"))
    if preds:
        o = json.loads(preds[-1].read_text())
        top = []
        for r in o["predictions"][: o["top_n"]]:
            top.append({k: r[k] for k in ("rank", "code", "name", "industry", "prob", "close", "r1")} |
                       {"pos": [f"{f['label']} {f['value_str']}" for f in r["factors_pos"][:3]],
                        "neg": [f"{f['label']} {f['value_str']}" for f in r["factors_neg"][:2]]})
        _dump("latest_prediction.json", {"pred_date": o["pred_date"], "target_date": o["target_date"],
                                         "universe_size": o["universe_size"], "market": o["market"],
                                         "model": o["model"], "top": top,
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
        for f in sorted((REPORT_DIR / sub).glob("*.md"))[-60:]:
            shutil.copy2(f, rep / sub / f.name)
    for f in ("validation.md", "error_patterns.md"):
        if (REPORT_DIR / f).exists():
            shutil.copy2(REPORT_DIR / f, rep / f)
    meta["reports"] = {sub: sorted(p.name for p in (rep / sub).glob("*.md")) for sub in ("predict", "review")}
    _dump("meta.json", meta)
    print(f"site data written to {OUT}")
