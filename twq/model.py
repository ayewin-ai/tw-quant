"""LightGBM model: walk-forward validation, training, contributions."""
import json
import datetime as dt

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import features as F
from .config import MODEL_DIR, REPORT_DIR, TOP_N, DISCLAIMER

PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_child_samples=300,
              feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0,
              verbose=-1, seed=42, num_threads=4)
N_ROUNDS = 300
COST_ROUNDTRIP = 0.001425 * 2 * 0.6 + 0.003  # 手續費(6折)x2 + 證交稅0.3%


def train(X: pd.DataFrame, feats=None):
    feats = feats or F.FEATURES
    d = X[X.y.notna()]
    ds = lgb.Dataset(d[feats], label=d["y"])
    return lgb.train(PARAMS, ds, num_boost_round=N_ROUNDS)


def contributions(booster, Xp: pd.DataFrame, feats=None):
    feats = feats or F.FEATURES
    c = booster.predict(Xp[feats], pred_contrib=True)
    return pd.DataFrame(c[:, :-1], index=Xp.index, columns=feats), c[:, -1]


def walk_forward(X, min_train_days=250, step=21, start=None):
    days = sorted(X.index.get_level_values("date").unique())
    lab_days = [d for d in days if X.loc[d].y.notna().any()]
    first = min_train_days if start is None else max(min_train_days // 2, next(i for i, d in enumerate(lab_days) if d >= start))
    out = []
    for i in range(first, len(lab_days), step):
        train_days = lab_days[:i]          # labels of these days are known before test day i closes
        test_days = lab_days[i:i + step]
        tr = X.loc[train_days[0]:train_days[-1]]
        te = X.loc[test_days[0]:test_days[-1]]
        b = train(tr)
        p = b.predict(te[F.FEATURES])
        o = te[["next_ret", "next_oc", "next_mkt", "next_excess", "next_xs", "y"]].copy()
        o["prob"] = p
        out.append(o)
        print(f"  fold train {train_days[0]}~{train_days[-1]} ({len(tr)}) test {test_days[0]}~{test_days[-1]}", flush=True)
    return pd.concat(out), b


def metrics(oos: pd.DataFrame, top=TOP_N):
    o = oos[oos.y.notna()].copy()
    res = {}
    res["n_obs"] = int(len(o))
    res["n_days"] = int(o.index.get_level_values("date").nunique())
    res["test_period"] = [o.index.get_level_values("date").min(), o.index.get_level_values("date").max()]
    res["base_rate_beat_median"] = float(o.y.mean())
    res["base_rate_excess_up"] = float((o.next_excess > 0).mean())
    res["auc"] = float(roc_auc_score(o.y, o.prob))
    res["auc_vs_taiex_excess"] = float(roc_auc_score((o.next_excess > 0).astype(int), o.prob))
    res["accuracy@0.5"] = float(((o.prob > 0.5) == (o.y > 0)).mean())
    o["rank"] = o.groupby(level="date").prob.rank(ascending=False, method="first")
    ic = o.groupby(level="date").apply(lambda g: g.prob.corr(g.next_excess, method="spearman"))
    res["rank_ic_mean"] = float(ic.mean())
    res["rank_ic_ir"] = float(ic.mean() / ic.std() * np.sqrt(252)) if ic.std() > 0 else None
    tp = o[o["rank"] <= top]
    res[f"top{top}_precision_beat_median"] = float((tp.y > 0).mean())
    res[f"top{top}_precision_excess"] = float((tp.next_excess > 0).mean())
    res[f"top{top}_precision_abs_up"] = float((tp.next_ret > 0).mean())
    res["universe_abs_up_rate"] = float((o.next_ret > 0).mean())
    res[f"top{top}_avg_excess_bp"] = float(tp.next_excess.mean() * 1e4)
    o["dec"] = o.groupby(level="date").prob.transform(lambda s: pd.qcut(s.rank(method="first"), 10, labels=False))
    dec = o.groupby("dec").next_excess.mean() * 1e4
    res["decile_avg_excess_bp"] = {int(k): round(float(v), 1) for k, v in dec.items()}

    # backtest: equal-weight top-N each day
    daily = tp.groupby(level="date").agg(cc=("next_ret", "mean"), oc=("next_oc", "mean"), mkt=("next_mkt", "mean"))
    codes = tp.reset_index().groupby("date").code.apply(set)
    prev = None
    turn = []
    for d, s in codes.items():
        turn.append(1.0 if prev is None else 1 - len(s & prev) / len(s))
        prev = s
    daily["turnover"] = turn
    daily["cc_net"] = daily.cc - daily.turnover * COST_ROUNDTRIP
    daily["oc_net"] = daily.oc - COST_ROUNDTRIP  # open->close is always a full round trip
    eq = (1 + daily[["cc", "cc_net", "oc", "oc_net", "mkt"]]).cumprod()
    res["backtest"] = {
        "days": int(len(daily)),
        "cum_ret_top_close2close_gross": float(eq.cc.iloc[-1] - 1),
        "cum_ret_top_close2close_net": float(eq.cc_net.iloc[-1] - 1),
        "cum_ret_top_open2close_gross": float(eq.oc.iloc[-1] - 1),
        "cum_ret_top_open2close_net": float(eq.oc_net.iloc[-1] - 1),
        "cum_ret_taiex": float(eq.mkt.iloc[-1] - 1),
        "avg_turnover": float(daily.turnover.mean()),
        "daily_win_rate_vs_taiex(gross c2c)": float((daily.cc > daily.mkt).mean()),
        "sharpe_c2c_gross": float(daily.cc.mean() / daily.cc.std() * np.sqrt(252)),
        "sharpe_taiex": float(daily.mkt.mean() / daily.mkt.std() * np.sqrt(252)),
    }
    return res, daily, eq, ic


def validate(start=None):
    print("building features...", flush=True)
    X = F.build()
    print("rows", len(X), "days", X.index.get_level_values('date').nunique(), flush=True)
    oos, last_model = walk_forward(X, start=start)
    res, daily, eq, ic = metrics(oos)
    # naive baseline for comparison: 1-day reversal (buy yesterday's biggest losers)
    base = oos.copy()
    base["prob"] = -X.loc[oos.index, "r1"].fillna(0).values
    bres, _, _, _ = metrics(base)
    res["baseline_reversal"] = {k: bres[k] for k in ("auc", "rank_ic_mean", f"top{TOP_N}_precision_excess",
                                                      f"top{TOP_N}_avg_excess_bp")}
    res["baseline_reversal"]["cum_ret_top_open2close_net"] = bres["backtest"]["cum_ret_top_open2close_net"]
    res["baseline_reversal"]["cum_ret_top_close2close_gross"] = bres["backtest"]["cum_ret_top_close2close_gross"]
    imp = pd.Series(last_model.feature_importance("gain"), index=F.FEATURES).sort_values(ascending=False)
    res["top_features_gain"] = {k: round(float(v), 1) for k, v in imp.head(15).items()}
    res["generated"] = dt.datetime.now().isoformat(timespec="seconds")
    (MODEL_DIR / "validation.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
    daily.join(eq.add_suffix("_equity")).to_csv(REPORT_DIR / "backtest_daily.csv")
    oos.to_parquet(MODEL_DIR / "oos_predictions.parquet") if _has_parquet() else oos.to_csv(MODEL_DIR / "oos_predictions.csv.gz")
    _write_md(res)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return res


def _has_parquet():
    try:
        import pyarrow  # noqa
        return True
    except ImportError:
        return False


def _write_md(res):
    b = res["backtest"]
    pct = lambda x: f"{x * 100:.2f}%"
    lines = [
        "# 模型驗證報告 (Walk-forward, 樣本外)", "", DISCLAIMER, "",
        f"- 產生時間：{res['generated']}",
        f"- 測試期間：{res['test_period'][0]} ~ {res['test_period'][1]}（{res['n_days']} 個交易日，{res['n_obs']:,} 筆股票-日）",
        "- 方法：LightGBM 二元分類，目標＝隔日報酬是否高於當日全體股票中位數（選股能力，與大盤漲跌無關）；"
        "擴張視窗，每 21 個交易日重新訓練，只用當時已知資料。績效另以「勝過加權指數」與「絕對上漲」衡量。", "",
        "## 分類指標", "",
        "| 指標 | 數值 |", "|---|---|",
        f"| AUC（跑贏中位數） | {res['auc']:.4f} |",
        f"| AUC（勝過加權指數） | {res['auc_vs_taiex_excess']:.4f} |",
        f"| 準確率 (門檻0.5，跑贏中位數) | {pct(res['accuracy@0.5'])} |",
        f"| Top-{TOP_N} 跑贏中位數比例 | {pct(res[f'top{TOP_N}_precision_beat_median'])} |",
        f"| 基準：勝過大盤比例 | {pct(res['base_rate_excess_up'])} |",
        f"| 每日 Rank IC 平均 | {res['rank_ic_mean']:.4f} |",
        f"| Top-{TOP_N} 勝過大盤比例 (precision) | {pct(res[f'top{TOP_N}_precision_excess'])} |",
        f"| Top-{TOP_N} 隔日上漲比例 | {pct(res[f'top{TOP_N}_precision_abs_up'])} |",
        f"| 全體股票隔日上漲比例 (基準) | {pct(res['universe_abs_up_rate'])} |",
        f"| Top-{TOP_N} 平均超額報酬 | {res[f'top{TOP_N}_avg_excess_bp']:.1f} bp |", "",
        "## 對照組：單純「買前一天跌最多」(1日反轉)", "",
        f"- AUC {res['baseline_reversal']['auc']:.4f}，Rank IC {res['baseline_reversal']['rank_ic_mean']:.4f}，"
        f"Top-{TOP_N} 勝過大盤 {pct(res['baseline_reversal'][f'top{TOP_N}_precision_excess'])}，"
        f"平均超額 {res['baseline_reversal'][f'top{TOP_N}_avg_excess_bp']:.1f} bp，"
        f"收盤→收盤累積(毛) {pct(res['baseline_reversal']['cum_ret_top_close2close_gross'])}，"
        f"開盤→收盤累積(扣成本) {pct(res['baseline_reversal']['cum_ret_top_open2close_net'])}", "",
        "## 十分位（0=機率最低, 9=最高）平均隔日超額報酬 (bp)", "",
        "| " + " | ".join(str(k) for k in res["decile_avg_excess_bp"]) + " |",
        "|" + "---|" * len(res["decile_avg_excess_bp"]),
        "| " + " | ".join(str(v) for v in res["decile_avg_excess_bp"].values()) + " |", "",
        f"## 簡易回測：每日等權買進 Top-{TOP_N}", "",
        "| 版本 | 累積報酬 |", "|---|---|",
        f"| 收盤→隔日收盤（毛） | {pct(b['cum_ret_top_close2close_gross'])} |",
        f"| 收盤→隔日收盤（扣成本，換手 {b['avg_turnover']:.0%}） | {pct(b['cum_ret_top_close2close_net'])} |",
        f"| 隔日開盤→收盤（可實際執行，毛） | {pct(b['cum_ret_top_open2close_gross'])} |",
        f"| 隔日開盤→收盤（扣成本） | {pct(b['cum_ret_top_open2close_net'])} |",
        f"| 加權指數 | {pct(b['cum_ret_taiex'])} |", "",
        f"每日勝過大盤比例 {pct(b['daily_win_rate_vs_taiex(gross c2c)'])}；Sharpe（毛）{b['sharpe_c2c_gross']:.2f} vs 加權指數 {b['sharpe_taiex']:.2f}。", "",
        "**誠實說明**：收盤→隔日收盤版本假設能以預測日收盤價買進，但資料要收盤後才公布，實務上做不到；"
        "較接近實際的是「隔日開盤買、收盤賣」。成本以手續費 6 折 ×2 + 證交稅 0.3% 估算，未計滑價與漲停買不到的情況。", "",
        f"**結論**：可執行版本（隔日開盤買→收盤賣，扣成本）累積 {pct(b['cum_ret_top_open2close_net'])}，"
        f"{'沒有' if b['cum_ret_top_open2close_net'] <= b['cum_ret_taiex'] else '有'}勝過加權指數 {pct(b['cum_ret_taiex'])}。"
        "收盤→收盤與開盤→收盤的巨大落差代表：模型抓到的幾乎都是「隔夜跳空」（收盤後到隔日開盤），"
        "這段報酬在收盤後才拿到資料時無法取得。排名有統計上的區辨力（十分位單調），但目前不構成可交易的優勢。", "",
        "## 重要特徵 (最後一期模型, gain)", "",
    ]
    for k, v in res["top_features_gain"].items():
        lines.append(f"- {k}（{F.label(k)}，{F.category(k)}）：{v}")
    (REPORT_DIR / "validation.md").write_text("\n".join(lines) + "\n")
