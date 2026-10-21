"""學術/大師因子組的 ablation + tune 護欄評估。checkpoint: models/guru_eval.json"""
import json
import sys

from . import features as F, model as M, tune as T, weights as W
from .config import MODEL_DIR

SETS = ["+academic_px", "+rev_anom", "+fundamental", "+guru", "all_guru"]
PATH = MODEL_DIR / "guru_eval.json"
LONG_START = "2025-07-01"


def run():
    done = json.loads(PATH.read_text()) if PATH.exists() else {}
    X = F.build()
    w = W.load()
    lab_days = sorted(X[X.y.notna()].index.get_level_values("date").unique())
    start = lab_days[-T.EVAL_DAYS]
    if "champion" not in done:
        done["champion"] = T.evaluate(T._oos(X, w, start), w)
        PATH.write_text(json.dumps(done, ensure_ascii=False, indent=1))
    champ = done["champion"]
    for name in ["base"] + SETS:
        feats = F.FEATURE_SETS[name]
        for label in ("y", "y_oc"):
            k = f"long|{name}|{label}"
            if k in done:
                continue
            oos, b = M.walk_forward(X, step=42, start=LONG_START, feats=feats, label=label, verbose=False)
            r, daily, eq, ic = M.metrics(oos)
            imp = sorted(zip(b.feature_importance("gain"), b.feature_name()), reverse=True)
            done[k] = {"auc": r["auc"], "auc_oc": r["auc_oc"], "rank_ic": r["rank_ic_mean"],
                       "top_avg_oc_bp": r[f"top{M.TOP_N}_avg_oc_bp"], "cum_oc_net": r["backtest"]["cum_ret_top_open2close_net"],
                       "cum_cc_gross": r["backtest"]["cum_ret_top_close2close_gross"], "cum_taiex": r["backtest"]["cum_ret_taiex"],
                       "excess_prec": r[f"top{M.TOP_N}_precision_excess"], "period": r["test_period"], "n_days": r["n_days"],
                       "top_feats": [n for _, n in imp[:10]],
                       "new_in_top20": [n for _, n in imp[:20] if n not in F.BASE_FEATURES]}
            PATH.write_text(json.dumps(done, ensure_ascii=False, indent=1))
            print(k, {x: done[k][x] for x in ("auc", "auc_oc", "rank_ic", "top_avg_oc_bp", "cum_oc_net")}, flush=True)
        if name == "base":
            continue
        k = f"tune|{name}"
        if k not in done:
            cw = W.set_(w, "feature_set", name)
            ch = T.evaluate(T._oos(X, cw, start), cw)
            ok, gain, msg = T._verdict(champ, ch)
            done[k] = {**ch, "ok": ok, "msg": msg}
            PATH.write_text(json.dumps(done, ensure_ascii=False, indent=1))
            print(k, ok, msg, flush=True)
    return done


if __name__ == "__main__":
    run()
