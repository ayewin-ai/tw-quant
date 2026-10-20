"""Guarded daily self-adjustment (`twq tune`).

1. Read recent review findings (review logs incl. pre-open, gap vs intraday attribution).
2. Turn them into a few small candidate changes to config/weights.json (bounded, limited step size).
3. Evaluate champion vs each challenger with walk-forward on the last EVAL_DAYS trading days
   (open->close after costs, Top-10 excess vs TAIEX, AUC), never on just yesterday.
4. Accept at most one change per day, only if it is better by a margin on several metrics and in both halves.
5. Roll back if the live results of the current version are clearly worse than the previous version's.
Every decision -> reports/tuning_log.csv + reports/CHANGELOG_model.md.
"""
import datetime as dt
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import features as F, model as M, weights as W
from .config import MODEL_DIR, REPORT_DIR, TOP_N

EVAL_DAYS = 90
STEP = 30
LOG = REPORT_DIR / "tuning_log.csv"
CHANGELOG = REPORT_DIR / "CHANGELOG_model.md"
MIN_GAIN_BP = 3.0          # open->close net, bp/day
MAX_EXCESS_LOSS_BP = 2.0   # Top-10 close->close excess may not fall more than this
MAX_AUC_LOSS = 0.002
ROLLBACK_DAYS = 10
ROLLBACK_MARGIN = 0.003    # 30 bp/day worse live Top-10 excess than the previous version


# ------------------------------------------------------------------ findings
def findings(days=20):
    out = {"n_days": 0, "causes": {}, "pre_causes": {}, "intraday_bp": None, "gap_bp": None, "notes": []}
    try:
        lg = pd.read_csv(REPORT_DIR / "review_log.csv", dtype={"pred_date": str})
        recent = sorted(lg.pred_date.unique())[-days:]
        w = lg[lg.pred_date.isin(recent) & (lg.kind == "wrong_pick")]
        out["n_days"] = len(recent)
        out["causes"] = w.causes.fillna("").str.split(";").explode().value_counts().to_dict()
        out["n_wrong"] = int(len(w))
    except Exception as e:
        out["notes"].append(f"review_log 無法讀取：{e}")
    try:
        lp = pd.read_csv(REPORT_DIR / "review_log_preopen.csv", dtype={"pred_date": str})
        out["pre_causes"] = lp[lp.kind == "wrong_pick"].causes.fillna("").str.split(";").explode().value_counts().to_dict()
    except Exception:
        pass
    try:
        c = pd.read_csv(REPORT_DIR / "archive" / "daily_comparison.csv").tail(days)
        out["intraday_bp"] = float(c.close_top10_intraday.mean() * 1e4)
        out["gap_bp"] = float(c.close_top10_gap.mean() * 1e4)
        out["live_days"] = int(c.close_top10_intraday.notna().sum())
    except Exception:
        pass
    return out


def candidates(w, f):
    """Small, bounded changes aimed at the detected problems. Returns [(key, new_value, reason_zh)]."""
    c = f["causes"]
    nw = max(f.get("n_wrong", 0), 1)
    share = lambda *ks: sum(c.get(k, 0) for k in ks) / nw
    cands = []

    def bump(key, sign=1, reason=""):
        lo, hi, step = W.BOUNDS[key]
        cur = W.get(w, key)
        new = type(lo)(min(max(cur + sign * step, lo), hi))
        if new != cur:
            cands.append((key, new, reason))

    if share("追高反轉", "漲停/題材") >= 0.15 or (c.get("開高走低/跳空回補", 0) + f["pre_causes"].get("開高走低/跳空回補", 0)) >= 3:
        bump("score.limitup_penalty", +1, f"近期誤判中「追高反轉/漲停」占 {share('追高反轉', '漲停/題材'):.0%}：提高前日漲停扣分")
    if (f.get("intraday_bp") is not None and f["intraday_bp"] < 0) or share("追高反轉") >= 0.1:
        bump("score.gap_risk_penalty", +1, f"收盤後名單盤中(開→收)平均 {f.get('intraday_bp') or 0:+.0f} bp、跳空 {f.get('gap_bp') or 0:+.0f} bp："
                                            "報酬集中在拿不到的跳空，提高「前日大漲」扣分")
    if share("產業輪動", "大盤系統性") >= 0.3:
        cur = W.get(w, "score.max_per_industry")
        if cur > 3:
            cands.append(("score.max_per_industry", max(3, cur - W.BOUNDS["score.max_per_industry"][2]),
                          f"誤判中「產業輪動/大盤系統性」占 {share('產業輪動', '大盤系統性'):.0%}：限制 Top-10 同產業檔數以分散"))
    if share("籌碼反轉", "個股特有/無法解釋") >= 0.2:
        bump("liquidity_min_avg_value", +1, "誤判中「籌碼反轉/個股特有」偏多：提高流動性門檻，避開小型股雜訊")
    if share("個股特有/無法解釋", "大盤系統性") >= 0.3 or not cands:
        hl = w.get("recency_half_life_days")
        new = 365 if not hl else (180 if hl > 180 else None)
        if new != hl:
            cands.append(("recency_half_life_days", new, "市場型態變化/無法解釋偏多：訓練樣本加入時間衰減（較重視近期）"))
    # relaxation candidates so penalties can also come back down when they stop helping
    for key in ("score.limitup_penalty", "score.gap_risk_penalty"):
        if W.get(w, key) > 0 and not any(k == key for k, _, _ in cands):
            bump(key, -1, "檢查既有扣分是否仍有幫助（可回調）")
    return cands[:4]


# ------------------------------------------------------------------ evaluation
def _oos(X, w, start):
    key = f"tune_oos_{W.model_hash(w)}_{start}_{X.index.get_level_values('date').max()}.pkl"
    p = MODEL_DIR / key
    if p.exists():
        return pd.read_pickle(p)
    feats = F.FEATURE_SETS[w.get("feature_set", "base")]
    oos, _ = M.walk_forward(X, step=STEP, start=start, feats=feats, label="y", verbose=False, w=w)
    cols = [c for c in ("r1", "industry", "log_value20", "gap_expect") if c in X]
    oos = oos.join(X[cols])
    oos.to_pickle(p)
    for old in sorted(MODEL_DIR.glob("tune_oos_*.pkl"), key=lambda q: q.stat().st_mtime)[:-6]:
        old.unlink(missing_ok=True)
    return oos


def evaluate(oos, w):
    rows, scored = [], []
    for d, g in oos.groupby(level="date"):
        g = g[g.next_oc.notna()]
        if len(g) < 50:
            continue
        a = W.adjust(g.droplevel("date"), w, TOP_N)
        t = a[a["rank"] <= TOP_N]
        rows.append({"date": d, "oc": t.next_oc.mean(), "ex": t.next_excess.mean()})
        scored.append(a[["score", "y_oc"]])
    r = pd.DataFrame(rows).set_index("date")
    s = pd.concat(scored)
    s = s[s.y_oc.notna()]
    half = len(r) // 2
    net = r.oc - M.COST_ROUNDTRIP
    return {"days": len(r), "net_oc_bp": net.mean() * 1e4, "excess_bp": r.ex.mean() * 1e4,
            "auc_oc": float(roc_auc_score(s.y_oc, s.score)),
            "net_h1_bp": net.iloc[:half].mean() * 1e4, "net_h2_bp": net.iloc[half:].mean() * 1e4,
            "period": f"{r.index.min()}~{r.index.max()}"}


def _verdict(champ, ch):
    d_net = ch["net_oc_bp"] - champ["net_oc_bp"]
    d_ex = ch["excess_bp"] - champ["excess_bp"]
    d_auc = ch["auc_oc"] - champ["auc_oc"]
    h1 = ch["net_h1_bp"] - champ["net_h1_bp"]
    h2 = ch["net_h2_bp"] - champ["net_h2_bp"]
    why = []
    if d_net < MIN_GAIN_BP:
        why.append(f"開→收淨報酬只改善 {d_net:+.1f} bp/日（門檻 +{MIN_GAIN_BP:.0f}）")
    if d_ex < -MAX_EXCESS_LOSS_BP:
        why.append(f"Top-10 超額變差 {d_ex:+.1f} bp")
    if d_auc < -MAX_AUC_LOSS:
        why.append(f"AUC 變差 {d_auc:+.4f}")
    if h1 <= 0 or h2 <= 0:
        why.append(f"前後半段不一致（{h1:+.1f} / {h2:+.1f} bp）")
    ok = not why
    msg = (f"開→收淨 {d_net:+.1f} bp/日、超額 {d_ex:+.1f} bp、AUC {d_auc:+.4f}、前/後半 {h1:+.1f}/{h2:+.1f} bp"
           + ("" if ok else "；不採用：" + "；".join(why)))
    return ok, d_net, msg


# ------------------------------------------------------------------ rollback
def rollback_check(w):
    if not LOG.exists():
        return None
    lg = pd.read_csv(LOG)
    acc = lg[lg.decision.isin(["accepted", "rollback"])]
    if acc.empty:
        return None
    last = acc.iloc[-1]
    try:
        s = pd.read_csv(REPORT_DIR / "archive" / "daily_summary.csv")
    except Exception:
        return None
    s = s[s.kind == "close"].sort_values("pred_date")
    after = s[s.pred_date > last.date].tail(ROLLBACK_DAYS)
    before = s[s.pred_date <= last.date].tail(ROLLBACK_DAYS)
    if len(after) < ROLLBACK_DAYS or len(before) < ROLLBACK_DAYS:
        return None
    a, b = after.top_avg_excess.mean(), before.top_avg_excess.mean()
    if a < b - ROLLBACK_MARGIN and a < 0:
        prev = W.HIST / f"weights_v{int(last.version_before)}.json"
        if prev.exists():
            return json.loads(prev.read_text()), f"上線後 {ROLLBACK_DAYS} 日 Top-10 超額 {a:+.2%}/日，明顯差於前一版 {b:+.2%}/日"
    return None


# ------------------------------------------------------------------ main
def _log(rows):
    df = pd.DataFrame(rows)
    df.to_csv(LOG, mode="a", header=not LOG.exists(), index=False)


HEAD = ("# 模型調整紀錄（CHANGELOG）\n\n"
        "自動調權（`twq tune`）與程式修正（commit 訊息前綴 `auto-fix:`）都記在這裡，最新在最上面。"
        "本紀錄僅供研究，不構成投資建議。\n\n<!-- entries -->\n")


def _changelog(lines):
    body = ""
    if CHANGELOG.exists():
        t = CHANGELOG.read_text()
        body = t.split("<!-- entries -->\n", 1)[1] if "<!-- entries -->\n" in t else t
    CHANGELOG.write_text(HEAD + "\n".join(lines) + "\n\n" + body)


def record_fix(summary, files=""):
    """Used by the daily routine agent after an `auto-fix:` commit."""
    _changelog([f"## {dt.date.today().isoformat()} 程式修正（auto-fix）", "", f"- {summary}" + (f"（{files}）" if files else "")])


def run(dry=False):
    t0 = time.time()
    today = dt.date.today().isoformat()
    if not dry and LOG.exists():
        lg = pd.read_csv(LOG)
        if ((lg.date == today) & lg.decision.isin(["champion"])).any():
            print(f"今天（{today}）已經調整/評估過，略過（每日最多一次）。", flush=True)
            return []
    w = W.load()
    if not W.PATH.exists():
        W.save(w)
    f = findings()
    X = F.build()
    lab_days = sorted(X[X.y.notna()].index.get_level_values("date").unique())
    start = lab_days[-EVAL_DAYS]
    champ_oos = _oos(X, w, start)
    champ = evaluate(champ_oos, w)
    rows = [{"date": today, "decision": "champion", "key": "", "old": "", "new": "", "version_before": w["version"],
             "version_after": w["version"], **{k: round(v, 4) if isinstance(v, float) else v for k, v in champ.items()},
             "reason": f"現行版本 v{w['version']} 近 {champ['days']} 日樣本外表現"}]
    md = [f"## {today}（v{w['version']}）", "",
          f"- 檢視近 {f['n_days']} 次檢討{'（樣本仍少，只用來挑候選；是否採用一律看近 ' + str(EVAL_DAYS) + ' 日 walk-forward）' if f['n_days'] < 10 else ''}：誤判原因 " + ("、".join(f"{k}×{v}" for k, v in list(f["causes"].items())[:6]) or "無") +
          f"；收盤後名單近期 跳空 {f.get('gap_bp') or 0:+.0f} bp / 盤中 {f.get('intraday_bp') or 0:+.0f} bp",
          f"- 現行版本樣本外（{champ['period']}，{champ['days']} 日）：開→收扣成本 {champ['net_oc_bp']:+.1f} bp/日、"
          f"Top-10 超額 {champ['excess_bp']:+.1f} bp/日、AUC(開→收) {champ['auc_oc']:.4f}"]
    new_w, decision_note = None, None
    rb = rollback_check(w)
    if rb:
        prev, why = rb
        new_w = {**prev, "version": w["version"] + 1, "updated": today}
        rows.append({"date": today, "decision": "rollback", "key": "all", "old": f"v{w['version']}", "new": f"v{prev['version']}",
                     "version_before": w["version"], "version_after": new_w["version"], "reason": why})
        md.append(f"- ⏪ **回滾**到 v{prev['version']} 的設定：{why}")
    else:
        best = None
        for key, val, reason in candidates(w, f):
            cw = W.set_(w, key, val)
            oos = champ_oos if W.model_hash(cw) == W.model_hash(w) else _oos(X, cw, start)
            ch = evaluate(oos, cw)
            ok, gain, msg = _verdict(champ, ch)
            rows.append({"date": today, "decision": "candidate_ok" if ok else "rejected", "key": key,
                         "old": W.get(w, key), "new": val, "version_before": w["version"], "version_after": w["version"],
                         **{k: round(v, 4) if isinstance(v, float) else v for k, v in ch.items()},
                         "reason": f"{reason}｜{msg}"})
            md.append(f"- {'✅ 候選通過' if ok else '❌ 不採用'} `{key}`：{W.get(w, key)} → {val}。{reason}。{msg}")
            if ok and (best is None or gain > best[0]):
                best = (gain, key, val, cw, reason, ch)
        if best:
            gain, key, val, cw, reason, ch = best
            new_w = {**cw, "version": w["version"] + 1, "updated": today}
            rows.append({"date": today, "decision": "accepted", "key": key, "old": W.get(w, key), "new": val,
                         "version_before": w["version"], "version_after": new_w["version"],
                         **{k: round(v, 4) if isinstance(v, float) else v for k, v in ch.items()},
                         "reason": f"{reason}｜開→收淨報酬 +{gain:.1f} bp/日"})
            md.append(f"- **採用** `{key}` = {val} → 新版本 v{new_w['version']}（每日最多一項變更）")
        else:
            md.append("- 今天沒有任何候選調整通過門檻，維持現行設定。")
    md.append(f"- 耗時 {time.time() - t0:.0f} 秒")
    if not dry:
        if new_w:
            W.save(new_w)
        _log(rows)
        _changelog(md)
    print("\n".join(md), flush=True)
    return rows
