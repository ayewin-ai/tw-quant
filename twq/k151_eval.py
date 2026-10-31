"""151 Trading Strategies 因子組的 ablation + tune 護欄評估（收盤後模型與盤前模型各一）。checkpoint: models/k151_eval.json

護欄與 `twq tune` 完全相同：最近 90 個交易日 walk-forward、開→收扣成本淨報酬、Top-10 超額、AUC、前後半段都要贏，
另外做 2025-07 起的較長期檢查。全部不通過就維持只記錄、不啟用。
"""
import json

from . import features as F, model as M, tune as T, weights as W
from .config import MODEL_DIR

PATH = MODEL_DIR / "k151_eval.json"
LONG_START = "2025-07-01"
G = ["mom", "rev", "ma", "sr", "rot", "all"]
TARGETS = {   # model -> (champion set, challengers, label)
    "close": ("base", [f"+k151_{g}" for g in G], "y"),
    "preopen": ("preopen_cmdty", [f"preopen_k151_{g}" for g in G], "y_oc"),
}


def _oos(X, w, start, feats, label, step):
    oos, b = M.walk_forward(X, step=step, start=start, feats=feats, label=label, verbose=False, w=w)
    cols = [c for c in ("r1", "industry", "log_value20", "gap_expect") if c in X]
    return oos.join(X[cols]), b


def run(only=None):
    done = json.loads(PATH.read_text()) if PATH.exists() else {}
    X = F.build()
    w = W.load()
    lab_days = sorted(X[X.y.notna()].index.get_level_values("date").unique())
    start = lab_days[-T.EVAL_DAYS]

    def save():
        PATH.write_text(json.dumps(done, ensure_ascii=False, indent=1))
    for model, (champ_set, chal, label) in TARGETS.items():
        if only and model not in only:
            continue
        cset = champ_set
        if model == "close":
            cset = w.get("feature_set", "base")
        k = f"{model}|champion|{cset}"
        if k not in done:
            oos, _ = _oos(X, w, start, F.FEATURE_SETS[cset], label, T.STEP)
            done[k] = T.evaluate(oos, w); save()
        champ = done[k]
        for name in chal:
            k = f"{model}|tune|{name}"
            if k not in done:
                oos, b = _oos(X, w, start, F.FEATURE_SETS[name], label, T.STEP)
                ch = T.evaluate(oos, w)
                ok, gain, msg = T._verdict(champ, ch)
                imp = sorted(zip(b.feature_importance("gain"), b.feature_name()), reverse=True)
                done[k] = {**ch, "ok": ok, "msg": msg, "champion": cset,
                           "top_new": [n for _, n in imp[:25] if n not in F.BASE_FEATURES][:8]}
                save()
                print(k, ok, msg, flush=True)
        for name in [cset] + chal:
            k = f"{model}|long|{name}"
            if k in done:
                continue
            oos, b = M.walk_forward(X, step=42, start=LONG_START, feats=F.FEATURE_SETS[name], label=label, verbose=False)
            r, daily, eq, ic = M.metrics(oos)
            done[k] = {"auc": r["auc"], "auc_oc": r["auc_oc"], "rank_ic": r["rank_ic_mean"],
                       "top_avg_oc_bp": r[f"top{M.TOP_N}_avg_oc_bp"], "cum_oc_net": r["backtest"]["cum_ret_top_open2close_net"],
                       "cum_cc_gross": r["backtest"]["cum_ret_top_close2close_gross"], "cum_taiex": r["backtest"]["cum_ret_taiex"],
                       "excess_prec": r[f"top{M.TOP_N}_precision_excess"], "period": r["test_period"], "n_days": r["n_days"]}
            save()
            print(k, {x: done[k][x] for x in ("auc", "auc_oc", "rank_ic", "top_avg_oc_bp", "cum_oc_net")}, flush=True)
    return done


if __name__ == "__main__":
    import sys
    run(sys.argv[1:] or None)
