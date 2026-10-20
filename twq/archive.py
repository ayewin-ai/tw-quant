"""Permanent, git-committed archive of every prediction and its next-day outcome (append-only).

- reports/archive/predictions/{close|preopen}_{date}.csv.gz : every stock's rank / prob / top factors
  (close_{pred_date} = close-time list for the next trading day; preopen_{target_date} = pre-open re-score)
- reports/archive/prediction_vs_actual/YYYY-MM.csv (by target month) : one row per (kind, pred_date, code) once the
  target day has data; rows are only ever appended, never rewritten. Sharded by month because the full history
  grows ~60 MB/year and GitHub rejects files > 100 MB.
- reports/archive/daily_summary.csv : one row per (kind, pred_date) with Top-N hit rate, avg return, excess, AUC.
"""
import gzip
import json
import shutil

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import db
from .config import PRED_DIR, REPORT_DIR

ARCH = REPORT_DIR / "archive"
PRED_ARCH = ARCH / "predictions"
PVA = ARCH / "prediction_vs_actual"
SUMMARY = ARCH / "daily_summary.csv"
PVA_COLS = ["kind", "pred_date", "target_date", "code", "name", "industry", "rank", "prob", "in_top",
            "ret_cc", "gap", "ret_oc", "taiex_cc", "bm_gap", "bm_oc", "bm_cc", "universe_oc",
            "excess_cc", "excess_gap", "excess_oc", "excess_oc_univ", "up", "beat", "basis"]
BENCH = "0050"  # gap / intraday benchmark: TAIEX has no tradable open print, 0050 (tracks TW50, ~cap-weighted) does


def _kind_key(o, stem):
    return ("preopen", o["target_date"]) if stem.startswith("preopen_") else ("close", o["pred_date"])


def _factors(fs, n=3):
    return "；".join(f"{f['label']}={f['value_str']}({f['contrib']:+.3f})" for f in (fs or [])[:n])


def archive_prediction(path, o=None, stem=None):
    """Write the compact CSV for one prediction JSON (idempotent; never deletes)."""
    o = o if o is not None else json.loads(path.read_text())
    kind, key = _kind_key(o, stem or path.stem)
    PRED_ARCH.mkdir(parents=True, exist_ok=True)
    out = PRED_ARCH / f"{kind}_{key}.csv.gz"
    if out.exists() and path is not None and path.stat().st_mtime <= out.stat().st_mtime:
        return out
    rows = [{"pred_date": o["pred_date"], "target_date": o["target_date"], "code": r["code"], "name": r["name"],
             "industry": r.get("industry"), "rank": r["rank"], "prob": round(r["prob"], 5), "r1": r.get("r1"),
             "close": r.get("close"), "limitup": r.get("limitup"),
             "feature_set": o.get("model", {}).get("feature_set", "base"), "label": o.get("model", {}).get("label", "y"),
             "top_pos": _factors(r.get("factors_pos")), "top_neg": _factors(r.get("factors_neg"))}
            for r in o["predictions"]]
    with open(out, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:  # deterministic bytes
        gz.write(pd.DataFrame(rows).to_csv(index=False).encode("utf-8"))
    return out


def archive_all():
    n = 0
    for p in sorted(PRED_DIR.glob("*.json")):
        archive_prediction(p)
        n += 1
    for p in sorted(PRED_DIR.glob("*.json.gz")):
        stem = p.name[:-len(".json.gz")]
        with gzip.open(p, "rt", encoding="utf-8") as f:
            o = json.load(f)
        kind, key = _kind_key(o, stem)
        if not (PRED_ARCH / f"{kind}_{key}.csv.gz").exists():
            archive_prediction(None, o, stem)
            n += 1
    return n


def _actuals(target):
    con = db.connect()
    px = pd.read_sql("SELECT code, ret, open, close FROM prices WHERE date=?", con, params=(target,)).set_index("code")
    t = con.execute("SELECT pct FROM index_daily WHERE date=? AND name='TAIEX'", (target,)).fetchone()
    con.close()
    return px, (t[0] if t else None)


def _bench(target):
    con = db.connect()
    r = con.execute("SELECT open, close, ret FROM prices WHERE date=? AND code=?", (target, BENCH)).fetchone()
    con.close()
    if not r or not r[0] or r[2] is None:
        return None, None, None
    o, c, cc = r
    prev = c / (1 + cc)
    return o / prev - 1, c / o - 1, cc


def update_vs_actual(top_n=10):
    """Append outcomes for archived predictions whose target day now has price data. Append-only."""
    ARCH.mkdir(parents=True, exist_ok=True)
    done = set()
    PVA.mkdir(parents=True, exist_ok=True)
    for f in PVA.glob("*.csv"):
        old = pd.read_csv(f, usecols=["kind", "pred_date"], dtype=str).drop_duplicates()
        done |= set(map(tuple, old.values))
    sums = []
    added = 0
    for f in sorted(PRED_ARCH.glob("*.csv.gz")):
        kind = f.name.split("_")[0]
        p = pd.read_csv(f, dtype={"code": str})
        if p.empty:
            continue
        pdte, tgt = p.pred_date.iloc[0], p.target_date.iloc[0]
        if (kind, pdte) in done:
            continue
        px, taiex = _actuals(tgt)
        if px.empty:
            continue
        d = p.rename(columns={"close": "pred_close"}).join(px, on="code", how="inner")
        d = d[d.ret.notna()].copy()
        if d.empty:
            continue
        d["ret_cc"] = d.ret
        d["ret_oc"] = d.close / d.open - 1
        d["gap"] = d.open / (d.close / (1 + d.ret)) - 1          # (1+cc) = (1+gap)(1+oc)
        for c in ("ret_oc", "gap"):
            d.loc[~np.isfinite(d[c]), c] = np.nan
        bg, bo, bc = _bench(tgt)
        d["taiex_cc"], d["bm_gap"], d["bm_oc"], d["bm_cc"] = taiex, bg, bo, bc
        d["universe_oc"] = d.ret_oc.mean()
        d["excess_cc"] = d.ret_cc - (taiex or 0)
        d["excess_gap"] = d.gap - (bg if bg is not None else np.nan)
        d["excess_oc"] = d.ret_oc - (bo if bo is not None else np.nan)
        d["excess_oc_univ"] = d.ret_oc - d.universe_oc
        basis = "oc" if kind == "preopen" else "cc"
        r, ex = (d.ret_oc, d.excess_oc_univ) if basis == "oc" else (d.ret_cc, d.excess_cc)
        d["up"] = (r > 0).astype(int)
        d["beat"] = (ex > 0).astype(int)
        d["basis"] = basis
        d["in_top"] = (d["rank"] <= top_n).astype(int)
        d["kind"] = kind
        out = d[PVA_COLS].round(5)
        shard = PVA / f"{tgt[:7]}.csv"
        out.to_csv(shard, mode="a", header=not shard.exists(), index=False)
        added += len(out)
        top = d[d.in_top == 1]
        y = (r > r.median()).astype(int)
        try:
            auc = roc_auc_score(y, d.prob)
        except ValueError:
            auc = np.nan
        sums.append({"kind": kind, "pred_date": pdte, "target_date": tgt, "basis": basis, "n": len(d),
                     "top_n": len(top), "top_up": top.up.mean(), "top_beat": top.beat.mean(),
                     "top_avg_ret": (top.ret_oc if basis == "oc" else top.ret_cc).mean(),
                     "top_avg_excess": (top.excess_oc if basis == "oc" else top.excess_cc).mean(),
                     "top_avg_ret_cc": top.ret_cc.mean(), "top_avg_ret_oc": top.ret_oc.mean(),
                     "top_avg_gap": top.gap.mean(), "top_beat_taiex_cc": (top.excess_cc > 0).mean(),
                     "top_ex_gap": top.excess_gap.mean(), "top_ex_oc": top.excess_oc.mean(), "top_ex_cc": top.excess_cc.mean(),
                     "bm_gap": bg, "bm_oc": bo, "bm_cc": bc,
                     "universe_avg_ret": r.mean(), "taiex_cc": taiex, "auc": auc,
                     "rank_ic": d.prob.corr(r, method="spearman")})
    if sums:
        s = pd.DataFrame(sums).round(5)
        if SUMMARY.exists():  # append-only, but never append a (kind, pred_date) that is already recorded
            have = set(map(tuple, pd.read_csv(SUMMARY, usecols=["kind", "pred_date"], dtype=str).values))
            s = s[[(k, d) not in have for k, d in zip(s.kind, s.pred_date)]]
        if not s.empty:
            s.to_csv(SUMMARY, mode="a", header=not SUMMARY.exists(), index=False)
    return added


def read_summary():
    """daily_summary.csv with one row per (kind, pred_date): older runs appended some days twice; keep the first."""
    return pd.read_csv(SUMMARY).drop_duplicates(["kind", "pred_date"], keep="first")


def gzip_old(keep_days=7):
    """Keep every full JSON locally; gzip those older than keep_days that are already reviewed/archived."""
    files = sorted(p for p in PRED_DIR.glob("*.json") if not p.name.startswith("_"))
    for p in files[:-keep_days * 2] if len(files) > keep_days * 2 else []:
        o = json.loads(p.read_text())
        kind, key = _kind_key(o, p.stem)
        if not (PRED_ARCH / f"{kind}_{key}.csv.gz").exists():
            continue
        rev = REPORT_DIR / "review" / f"review_{p.stem}.md"
        if not rev.exists():
            continue
        with open(p, "rb") as f, gzip.open(str(p) + ".gz", "wb") as g:
            shutil.copyfileobj(f, g)
        p.unlink()


COMP = ARCH / "daily_comparison.csv"


def write_comparison():
    """Per-target-date table (close-time vs pre-open), derived from the append-only daily_summary.csv."""
    from .model import COST_ROUNDTRIP
    if not SUMMARY.exists():
        return None
    s = read_summary()
    cols = {"top_up": "top10_up", "top_beat": "top10_beat", "top_beat_taiex_cc": "top10_beat_taiex",
            "top_avg_ret": "top10_avg_ret", "top_avg_excess": "top10_avg_excess", "auc": "auc", "rank_ic": "rank_ic",
            "top_avg_gap": "top10_gap", "top_avg_ret_oc": "top10_intraday", "top_avg_ret_cc": "top10_total",
            "top_ex_gap": "top10_ex_gap", "top_ex_oc": "top10_ex_intraday", "top_ex_cc": "top10_ex_total"}
    out = None
    for kind in ("close", "preopen"):
        k = s[s.kind == kind].set_index("target_date")[list(cols) + ["pred_date"]].rename(columns=cols)
        if kind == "preopen":
            k["top10_net_ret"] = k["top10_avg_ret"] - COST_ROUNDTRIP
        k = k.add_prefix(f"{kind}_")
        out = k if out is None else out.join(k, how="outer")
    g = s.groupby("target_date")
    out = out.join(g.taiex_cc.first().rename("taiex_ret")).join(g.bm_gap.first().rename("bm0050_gap")) \
        .join(g.bm_oc.first().rename("bm0050_intraday")).join(g.bm_cc.first().rename("bm0050_total")).sort_index().round(5)
    out.index.name = "target_date"
    out.to_csv(COMP)
    return out


def run():
    a = archive_all()
    b = update_vs_actual()
    try:
        gzip_old()
    except Exception as e:
        print("gzip_old error", e)
    write_comparison()
    print(f"archive: {a} prediction files archived, {b} outcome rows appended -> {PVA}", flush=True)
    return b
