"""ml4t_eval 的補充分析：截斷不變性稽核、meta-labeling、IC 衰減、校準度、特徵 IC 表。結果寫入 models/ml4t_eval.json（key 前綴 extra|）。"""
import sys

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

from . import features as F, model as M, tune as T, weights as W
from . import ml4t_eval as E
from .config import TOP_N

META_FEATS = ["prob", "rank_day", "vol20", "hl_range", "close_pos", "gap", "r1", "limit_up", "log_value20", "rel_r1",
              "mkt_r1", "mkt_vol20", "breadth", "f_ratio", "t_ratio", "ta_atr", "ta_cmf", "sg_hurst", "qr_resid_z", "gap_expect"]
NON_FEATURE = {"next_ret", "next_oc", "next_mkt", "next_excess", "next_xs", "y", "y_oc"}


# ------------------------------------------------------------------ 1. truncation invariance (look-ahead audit)
def audit_truncation(n_dates=25, back=30):
    full = F.build()
    days = sorted(full.index.get_level_values("date").unique())
    cut = days[-back]
    part = F.build(end=cut)
    cols = [c for c in full.columns if c not in NON_FEATURE and not c.startswith("next_")
            and pd.api.types.is_numeric_dtype(full[c])]
    chk = [d for d in days if d <= cut][-n_dates:]
    a = full.loc[full.index.get_level_values("date").isin(chk), cols]
    b = part.loc[part.index.get_level_values("date").isin(chk), cols]
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]
    diff = (a.astype(float) - b.astype(float)).abs()
    both_nan = a.isna() & b.isna()
    mism = ((diff > 1e-9) | (a.isna() ^ b.isna())) & ~both_nan
    per = mism.sum()
    bad = per[per > 0].sort_values(ascending=False)
    res = {"cut_date": cut, "check_dates": [chk[0], chk[-1]], "n_rows": int(len(common)), "n_features": len(cols),
           "rows_only_full": int(len(a.index.difference(b.index))) if False else None,
           "n_features_mismatch": int(len(bad)),
           "mismatch": {k: int(v) for k, v in bad.head(30).items()},
           "mismatch_rate": {k: round(float(v) / len(common), 5) for k, v in bad.head(30).items()}}
    E._save("extra|truncation", res)
    print("truncation:", res["n_rows"], "rows,", res["n_features"], "features, mismatching:", res["n_features_mismatch"], dict(list(res["mismatch"].items())[:15]))
    return res


# ------------------------------------------------------------------ 2. meta-labeling
def meta(model, pool=30, X=None):
    oos = pd.read_pickle(E.CACHE / f"oos_long_{model}.pkl")
    X = X if X is not None else F.build()
    w = W.load()
    cols = [c for c in META_FEATS if c in X.columns and c not in oos.columns]
    d = oos.join(X[cols])
    d["rank_day"] = d.groupby(level="date").prob.rank(ascending=False, method="first")
    d["pct_day"] = d["rank_day"] / d.groupby(level="date").prob.transform("size")
    d = d[d.rank_day <= pool].copy()
    d["xs_oc"] = d["y_oc"]   # label: beat median open->close
    feats = [c for c in META_FEATS if c in d.columns]
    days = sorted(d.index.get_level_values("date").unique())
    warm = 80
    params = dict(objective="binary", learning_rate=0.05, num_leaves=7, min_child_samples=100, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=20.0, verbose=-1, seed=7, num_threads=4)
    parts = []
    for i in range(warm, len(days), 21):
        tr = d.loc[days[0]:days[i - 1]]
        tr = tr[tr.y_oc.notna()]
        te = d.loc[days[i]:days[min(i + 20, len(days) - 1)]].copy()
        b = lgb.train(params, lgb.Dataset(tr[feats], label=tr.y_oc), num_boost_round=150)
        te["meta_p"] = b.predict(te[feats])
        parts.append(te)
    t = pd.concat(parts)
    t = t[t.next_oc.notna()]
    rows = []
    for dd, g in t.groupby(level="date"):
        prim = g[g.rank_day <= TOP_N]
        mt = g.sort_values("meta_p", ascending=False).head(TOP_N)
        comb = g.assign(cr=g.rank_day.rank() + g.meta_p.rank(ascending=False)).sort_values("cr").head(TOP_N)
        rows.append({"date": dd, "prim_oc": prim.next_oc.mean(), "meta_oc": mt.next_oc.mean(), "comb_oc": comb.next_oc.mean(),
                     "pool_oc": g.next_oc.mean(), "prim_ex": prim.next_excess.mean(), "meta_ex": mt.next_excess.mean(),
                     "comb_ex": comb.next_excess.mean()})
    r = pd.DataFrame(rows).set_index("date")
    cost = M.COST_ROUNDTRIP
    h = len(r) // 2
    out = {"model": model, "pool": pool, "days": len(r), "period": [r.index.min(), r.index.max()],
           "meta_auc_in_pool": float(roc_auc_score(t.y_oc.dropna(), t.meta_p[t.y_oc.notna()])),
           "prim_auc_in_pool": float(roc_auc_score(t.y_oc.dropna(), -t.rank_day[t.y_oc.notna()]))}
    for k in ("prim", "meta", "comb"):
        net = r[f"{k}_oc"] - cost
        out[k] = {"net_oc_bp": float(net.mean() * 1e4), "excess_bp": float(r[f"{k}_ex"].mean() * 1e4),
                  "h1": float(net.iloc[:h].mean() * 1e4), "h2": float(net.iloc[h:].mean() * 1e4),
                  "oc_up": float((r[f"{k}_oc"] > 0).mean())}
    out["pool_oc_bp"] = float(r.pool_oc.mean() * 1e4)
    # 配對差的 t 值（meta/comb 相對主模型，逐日開→收報酬差）與「最近 90 日」子窗
    for k in ("meta", "comb"):
        dd = (r[f"{k}_oc"] - r["prim_oc"])
        out[f"{k}_t"] = float(dd.mean() / dd.std() * np.sqrt(len(dd))) if dd.std() > 0 else 0.0
        s90 = r.iloc[-T.EVAL_DAYS:]
        h9 = len(s90) // 2
        dn9 = ((s90[f"{k}_oc"] - s90["prim_oc"]).mean()) * 1e4
        d1 = ((s90[f"{k}_oc"] - s90["prim_oc"]).iloc[:h9].mean()) * 1e4
        d2 = ((s90[f"{k}_oc"] - s90["prim_oc"]).iloc[h9:].mean()) * 1e4
        out[f"{k}_90"] = {"dnet_bp": float(dn9), "dh1": float(d1), "dh2": float(d2)}
        out[f"{k}_enable_90"] = bool(dn9 >= T.MIN_GAIN_BP and d1 > 0 and d2 > 0)
    for k in ("meta", "comb"):
        dn = out[k]["net_oc_bp"] - out["prim"]["net_oc_bp"]
        ok = dn >= T.MIN_GAIN_BP and out[k]["h1"] > out["prim"]["h1"] and out[k]["h2"] > out["prim"]["h2"] \
            and out[k]["excess_bp"] >= out["prim"]["excess_bp"] - T.MAX_EXCESS_LOSS_BP
        out[f"{k}_enable"] = bool(ok)
        out[f"{k}_msg"] = f"開→收淨 {dn:+.1f} bp/日、前/後半 {out[k]['h1'] - out['prim']['h1']:+.1f}/{out[k]['h2'] - out['prim']['h2']:+.1f} bp"
    E._save(f"extra|meta|{model}", out)
    print("meta", model, {k: out[k] for k in ("prim", "meta", "comb")}, out["meta_msg"], out["comb_msg"],
          {k: (out[f"{k}_t"], out[f"{k}_90"], out[f"{k}_enable_90"]) for k in ("meta", "comb")})
    return out


# ------------------------------------------------------------------ 3. IC decay + gap/intraday split
def ic_decay(X=None):
    from . import db
    oos = pd.read_pickle(E.CACHE / "oos_long_close.pkl")
    con = db.connect()
    px = pd.read_sql("SELECT date, code, ret FROM prices", con)
    con.close()
    ret = px.pivot(index="date", columns="code", values="ret").sort_index().fillna(0).clip(-0.5, 1.0)
    adj = (1 + ret).cumprod()
    res = {}
    for k in range(1, 6):
        fwd = (adj.shift(-k) / adj - 1).stack()
        fwd.index.names = ["date", "code"]
        d = oos[["prob"]].join(fwd.rename("f"), how="inner").dropna()
        ic = d.groupby(level="date").apply(lambda g: g.prob.corr(g.f, method="spearman"))
        res[f"c2c_{k}d"] = {"ic": float(ic.mean()), "ir": float(ic.mean() / ic.std() * np.sqrt(252)), "n": int(len(ic))}
    gap = (1 + oos.next_ret) / (1 + oos.next_oc) - 1
    for name, s in (("gap_next", gap), ("intraday_next", oos.next_oc)):
        d = pd.DataFrame({"prob": oos.prob, "f": s}).dropna()
        ic = d.groupby(level="date").apply(lambda g: g.prob.corr(g.f, method="spearman"))
        res[name] = {"ic": float(ic.mean()), "ir": float(ic.mean() / ic.std() * np.sqrt(252)), "n": int(len(ic))}
    E._save("extra|ic_decay_close", res)
    print("ic decay", res)
    return res


# ------------------------------------------------------------------ 4. calibration
def calibration():
    out = {}
    for model, lab in (("close", "y"), ("preopen", "y_oc")):
        oos = pd.read_pickle(E.CACHE / f"oos_long_{model}.pkl")
        d = oos[oos[lab].notna()]
        p, y = d.prob.clip(1e-4, 1 - 1e-4).values, d[lab].values
        bins = pd.qcut(p, 10, labels=False, duplicates="drop")
        rel = pd.DataFrame({"p": p, "y": y, "b": bins}).groupby("b").agg(p=("p", "mean"), y=("y", "mean"), n=("y", "size"))
        ece = float((rel.n / rel.n.sum() * (rel.p - rel.y).abs()).sum())
        brier = float(np.mean((p - y) ** 2))
        lg = np.log(p / (1 - p))
        lr = LogisticRegression(C=1e6).fit(lg[:, None], y)
        order = np.argsort(d.index.get_level_values("date").values, kind="stable")
        half = len(order) // 2
        a, b = order[:half], order[half:]
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p[a], y[a])
        base_b = float(np.mean((y[b].mean() - y[b]) ** 2))
        out[model] = {"label": lab, "n": int(len(d)), "brier": brier, "brier_const": float(np.mean((y.mean() - y) ** 2)),
                      "ece": ece, "logit_slope": float(lr.coef_[0][0]), "logit_intercept": float(lr.intercept_[0]),
                      "brier_h2_raw": float(np.mean((p[b] - y[b]) ** 2)), "brier_h2_iso": float(np.mean((iso.predict(p[b]) - y[b]) ** 2)),
                      "brier_h2_const": base_b, "base_rate": float(y.mean()), "mean_prob": float(p.mean()),
                      "reliability": [{"p": round(float(r.p), 4), "y": round(float(r.y), 4), "n": int(r.n)} for r in rel.itertuples()]}
        print("calib", model, {k: out[model][k] for k in ("brier", "brier_const", "ece", "logit_slope", "brier_h2_raw", "brier_h2_iso")})
    E._save("extra|calibration", out)
    return out


# ------------------------------------------------------------------ 5. feature IC table
def ic_table(X=None):
    X = X if X is not None else F.build()
    X = X.copy()
    X["xs_oc"] = X["next_oc"] - X.groupby(level="date")["next_oc"].transform("median")
    feats = [c for c in F.FEATURE_SETS["preopen_ml4t"] if c in X.columns]
    out = {}
    for tgt, nm in (("next_xs", "cc"), ("xs_oc", "oc")):
        ic = E._ic_series(X, feats, tgt)
        s = pd.DataFrame({"ic": ic.mean(), "icir": ic.mean() / ic.std() * np.sqrt(252), "t": ic.mean() / ic.std() * np.sqrt(len(ic)),
                          "hit": (ic > 0).mean()})
        out[nm] = {k: {c: round(float(v), 4) for c, v in row.items()} for k, row in s.iterrows()}
    new = set(F.TA.ALL_F)
    E._save("extra|ic_table", out)
    for nm in out:
        s = pd.DataFrame(out[nm]).T
        s["new"] = s.index.isin(new)
        print(nm, "top |t| new:", s[s.new].reindex(s[s.new].t.abs().sort_values(ascending=False).index).head(8)[["ic", "t"]].round(3).to_dict("index"))
        print(nm, "top |t| old:", s[~s.new].reindex(s[~s.new].t.abs().sort_values(ascending=False).index).head(5)[["ic", "t"]].round(3).to_dict("index"))
    return out


if __name__ == "__main__":
    what = sys.argv[1:] or ["trunc", "ic", "calib", "meta", "decay"]
    X = None
    if "trunc" in what:
        audit_truncation()
    if "calib" in what:
        calibration()
    if "decay" in what:
        ic_decay()
    if any(k in what for k in ("meta", "ic")):
        X = F.build()
    if "meta" in what:
        for m in ("close", "preopen"):
            meta(m, X=X)
    if "ic" in what:
        ic_table(X)
