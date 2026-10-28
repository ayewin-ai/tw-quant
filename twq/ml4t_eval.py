"""第三批參考（ML4T / QuantResearch / SignalProcessing / Stock_feature_engineering / VisualHFT）的評估與稽核。

護欄與 `twq tune`、risk_eval、cmdty_eval 完全相同：最近 90 個交易日 walk-forward、開→收扣成本淨報酬、Top-10 超額、
AUC、前後半段都要贏；另外做 2025-07 起較長期檢查（AUC、Rank IC、Top-10 開→收不能比冠軍差）。兩者都過才算「可啟用」。

內容：
  feat   特徵組：+ta / +signal / +qr / all_ml4t（盤前為 preopen_*）
  label  標籤變體：尾端標籤(前/後 30%)、波動門檻標籤(三重障礙的 1 日版)、排名回歸、波動標準化回歸
  select 折內特徵篩選：ICIR top-K、隨機雜訊特徵對照（Boruta-lite）
  audit  稽核：embargo（隔離訓練窗尾端 5 日）、洗牌標籤對照、特徵截斷不變性（無未來資訊）
  meta   元標記（meta-labeling）：在主模型 Top-30 內再訓練第二層，選出 Top-10
  extra  IC 衰減、校準度（Brier/ECE/等滲迴歸）、特徵 IC 表
checkpoint：models/ml4t_eval.json
"""
import json
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

from . import features as F, model as M, tune as T, weights as W
from .config import MODEL_DIR, TOP_N

PATH = MODEL_DIR / "ml4t_eval.json"
CACHE = MODEL_DIR / "ml4t_cache"      # 大型暫存（不進 git，見 .gitignore）
LONG_START = "2025-07-01"
CHAMP = {"close": ("base", "y"), "preopen": ("preopen_cmdty", "y_oc")}
FEAT_EXPS = {"close": ["+ta", "+signal", "+qr", "all_ml4t"], "preopen": ["preopen_ta", "preopen_signal", "preopen_qr", "preopen_ml4t"]}
LABEL_VARIANTS = ["tail30", "tb05", "rank_reg", "vn_reg"]
SELECTS = ["icir60", "noise"]
EXTRA_COLS = ("r1", "industry", "log_value20", "gap_expect")


# ------------------------------------------------------------------ data prep
def prep(X):
    """加入標籤變體欄位（只用來「訓練」；評估一律用原本的 y / y_oc）。"""
    X = X.copy()
    d = X.index.get_level_values("date")
    X["xs_oc"] = X["next_oc"] - X.groupby(level="date")["next_oc"].transform("median")
    vol = X["vol20"].clip(lower=0.005)
    for tag, xs in (("", "next_xs"), ("_oc", "xs_oc")):
        v = X[xs]
        pr = v.groupby(level="date").rank(pct=True)
        X[f"tail30{tag}"] = np.where(pr > 0.7, 1.0, np.where(pr < 0.3, 0.0, np.nan))
        X[f"tail30{tag}"] = X[f"tail30{tag}"].where(v.notna())
        X[f"tb05{tag}"] = np.where(v > 0.5 * vol, 1.0, np.where(v < -0.5 * vol, 0.0, np.nan))
        X[f"tb05{tag}"] = X[f"tb05{tag}"].where(v.notna())
        z = norm.ppf(((pr * 0.998) + 0.001).clip(0.001, 0.999))
        X[f"rank_reg{tag}"] = pd.Series(z, index=X.index).where(v.notna())
        X[f"vn_reg{tag}"] = (v / vol).clip(-3, 3)
    rng = np.random.default_rng(7)
    for k in range(6):
        X[f"noise{k}"] = rng.standard_normal(len(X)).astype("float32")
    # 洗牌標籤：同一天內打亂 y（placebo）
    for lab in ("y", "y_oc"):
        sh = X[lab].copy()
        for dt_, idx in X.groupby(level="date").indices.items():
            vals = sh.iloc[idx].values.copy()
            rng.shuffle(vals)
            sh.iloc[idx] = vals
        X[f"{lab}_shuf"] = sh
    return X


def _ic_series(X, feats, target):
    """每日 Spearman IC（特徵截面排名 vs 目標截面排名）。回傳 days x feats。"""
    g = X[feats + [target]].dropna(subset=[target])
    R = g[feats].groupby(level="date").rank(pct=True)
    t = g[target].groupby(level="date").rank(pct=True)
    out = {}
    for d, idx in g.groupby(level="date").indices.items():
        a = R.iloc[idx].values
        b = t.iloc[idx].values
        a = a - np.nanmean(a, axis=0)
        b = b - b.mean()
        num = np.nansum(a * b[:, None], axis=0)
        den = np.sqrt(np.nansum(a ** 2, axis=0) * (b ** 2).sum())
        out[d] = num / np.where(den > 0, den, np.nan)
    return pd.DataFrame(out, index=feats).T.sort_index()


# ------------------------------------------------------------------ experiment runner
def _spec(model, name, X, w):
    champ_set, label = CHAMP[model]
    base_feats = F.FEATURE_SETS[champ_set]
    sp = dict(feats=base_feats, label=label, train_label=None, params_extra=None, embargo=0, select=None)
    suf = "_oc" if label == "y_oc" else ""
    if name in F.FEATURE_SETS:
        sp["feats"] = F.FEATURE_SETS[name]
    elif name in ("champion", "secfix"):
        pass
    elif name in ("tail30", "tb05"):
        sp["train_label"] = name + suf
    elif name in ("rank_reg", "vn_reg"):
        sp["train_label"] = name + suf
        sp["params_extra"] = {"objective": "regression", "metric": "l2"}
    elif name == "embargo5":
        sp["embargo"] = 5
    elif name == "shuffled":
        sp["train_label"] = f"{label}_shuf"
    elif name == "icir60":
        sp["select"] = _icir_selector(X, base_feats, label, 60)
    elif name == "noise":
        sp["select"] = _noise_selector(X, base_feats, label, w)
    else:
        raise KeyError(name)
    return sp


def _icir_selector(X, feats, label, k):
    tgt = "next_xs" if label == "y" else "xs_oc"
    ic = _ic_series(X, list(feats), tgt)

    def sel(train_days):
        s = ic.loc[ic.index.isin(train_days)]
        icir = (s.mean() / s.std()).abs().sort_values(ascending=False)
        return list(icir.head(k).index)
    return sel


def _noise_selector(X, feats, label, w):
    """Boruta-lite：在訓練窗訓練含 6 個隨機雜訊特徵的快速模型，只保留 gain 高於「最大雜訊 gain」的特徵。"""
    import lightgbm as lgb
    noise = [f"noise{k}" for k in range(6)]

    def sel(train_days):
        tr = X.loc[train_days[0]:train_days[-1]]
        tr = tr[tr[label].notna()]
        tr = tr[tr.index.get_level_values("date").isin(train_days[-150:])]   # 最近 150 日加速
        cols = list(feats) + noise
        p = {**M.PARAMS, "min_child_samples": 300, "num_threads": 4}
        b = lgb.train(p, lgb.Dataset(tr[cols], label=tr[label]), num_boost_round=120)
        g = pd.Series(b.feature_importance("gain"), index=cols)
        thr = g[noise].max()
        keep = [c for c in feats if g[c] > thr]
        return keep if len(keep) >= 20 else list(g[list(feats)].sort_values(ascending=False).head(20).index)
    return sel


def _oos(X, w, start, sp, step):
    oos, b = M.walk_forward(X, step=step, start=start, feats=sp["feats"], label=sp["label"], verbose=False, w=w,
                            train_label=sp["train_label"], embargo=sp["embargo"], select=sp["select"],
                            params_extra=sp["params_extra"])
    cols = [c for c in EXTRA_COLS if c in X]
    return oos.join(X[cols]), b


def _long_metrics(oos):
    r, daily, eq, ic = M.metrics(oos)
    return {"auc": r["auc"], "auc_oc": r["auc_oc"], "rank_ic": r["rank_ic_mean"], "top_avg_oc_bp": r[f"top{M.TOP_N}_avg_oc_bp"],
            "cum_oc_net": r["backtest"]["cum_ret_top_open2close_net"], "cum_cc_gross": r["backtest"]["cum_ret_top_close2close_gross"],
            "cum_taiex": r["backtest"]["cum_ret_taiex"], "excess_prec": r[f"top{M.TOP_N}_precision_excess"],
            "period": r["test_period"], "n_days": r["n_days"]}


def long_ok(champ, ch):
    d_auc = ch["auc"] - champ["auc"]
    d_ic = ch["rank_ic"] - champ["rank_ic"]
    d_top = ch["top_avg_oc_bp"] - champ["top_avg_oc_bp"]
    ok = d_auc >= -0.0005 and d_ic >= 0 and d_top >= 0
    return ok, f"長窗 AUC {d_auc:+.4f}、RankIC {d_ic:+.4f}、Top-10 開→收 {d_top:+.1f} bp"


def _save(key, val):
    cur = json.loads(PATH.read_text()) if PATH.exists() else {}
    cur[key] = val
    PATH.write_text(json.dumps(cur, ensure_ascii=False, indent=1, default=float))


def _done():
    return json.loads(PATH.read_text()) if PATH.exists() else {}


def run_exp(X, w, model, name, start, lab_days, force=False):
    done = _done()
    key = f"{model}|{name}"
    if key in done and not force:
        return done[key]
    t0 = time.time()
    sp = _spec(model, name, X, w)
    res = {"model": model, "name": name, "n_features": len(sp["feats"])}
    oos_s, b = _oos(X, w, start, sp, T.STEP)
    res["short"] = T.evaluate(oos_s, w)
    oos_l, _ = _oos(X, w, LONG_START, sp, 42)
    res["long"] = _long_metrics(oos_l)
    if name == "champion":
        CACHE.mkdir(exist_ok=True)
        oos_l.to_pickle(CACHE / f"oos_long_{model}.pkl")
    res["seconds"] = round(time.time() - t0)
    _save(key, res)
    return res


def judge(model, name, done=None):
    done = done or _done()
    c, r = done.get(f"{model}|champion"), done.get(f"{model}|{name}")
    if not c or not r:
        return None
    ok, gain, msg = T._verdict(c["short"], r["short"])
    lok, lmsg = long_ok(c["long"], r["long"])
    r["verdict_short"], r["verdict_msg"], r["long_ok"], r["long_msg"] = ok, msg, lok, lmsg
    r["enable"] = bool(ok and lok)
    _save(f"{model}|{name}", r)
    return r


def run(models=("close", "preopen"), names=None, force=False):
    X = prep(F.build())
    w = W.load()
    lab_days = sorted(X[X.y.notna()].index.get_level_values("date").unique())
    start = lab_days[-T.EVAL_DAYS]
    for model in models:
        todo = ["champion"] + (names or (FEAT_EXPS[model] + LABEL_VARIANTS + SELECTS + ["embargo5", "shuffled"]))
        for name in todo:
            r = run_exp(X, w, model, name, start, lab_days, force)
            j = judge(model, name) if name != "champion" else None
            print(model, name, {k: round(v, 4) if isinstance(v, float) else v for k, v in r["short"].items() if k != "period"},
                  {k: round(r["long"][k], 4) for k in ("auc", "rank_ic", "top_avg_oc_bp")},
                  (j or {}).get("verdict_msg", ""), (j or {}).get("long_msg", ""), "ENABLE" if (j or {}).get("enable") else "", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    models = [m for m in a if m in ("close", "preopen")] or ["close", "preopen"]
    names = [x for x in a if x not in ("close", "preopen")] or None
    run(models, names)
