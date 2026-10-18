"""Next-day review: compare predictions vs actual, root-cause analysis, cumulative error log."""
import datetime as dt
import json
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import db, news as N, features as F
from .config import PRED_DIR, REVIEW_REPORT_DIR, REPORT_DIR, DISCLAIMER

TECH_INDUSTRIES = {"半導體", "電子零組件", "電腦及週邊", "光電", "通信網路", "其他電子", "電子通路", "資訊服務"}
LOG = REPORT_DIR / "review_log.csv"
PERF = REPORT_DIR / "performance_log.csv"
PATTERNS = REPORT_DIR / "error_patterns.md"

SUGGEST = {
    "大盤系統性": "加入「大盤方向」子模型或風險開關（台指期夜盤、VIX、外資期貨未平倉），大盤轉弱時降低部位；目前模型只預測相對強弱。",
    "美股隔夜": "加入開盤前更新：用預測日當晚美股（費半、台積電ADR、那指）重新計分，或改用台指期夜盤作為特徵。",
    "產業輪動": "加入類股指數動能、族群資金流（產業法人買賣超合計）、同族群同步度等特徵。",
    "新聞/事件": "加入新聞情緒分數（標題關鍵字或語言模型分類）與 MOPS 重大訊息公告旗標。",
    "法說會": "加入法說會行事曆（MOPS 法人說明會公告）作為事件特徵，法說前後波動加大。",
    "營收公布": "加入營收公布日旗標與「營收驚喜」（本月 YoY 相對過去 3 個月趨勢的偏離）。",
    "除權息": "加入除權息日旗標、殖利率與歷史填息率。",
    "追高反轉": "加入過熱/反轉因子（連續漲停、乖離過大、處置股名單、當沖比率），並對前一日漲停股降權。",
    "籌碼反轉": "加入法人買超持續性、券商分點（主力）資料與借券賣出資料。",
    "漲停/題材": "加入題材熱度（新聞量突增、族群同時漲停數）與注意股名單。",
    "法人同步買超": "隔日法人大買是同時發生的資訊，可考慮預測「法人是否持續買超」作為中介特徵。",
    "個股特有/無法解釋": "可能為隨機雜訊；持續累積觀察，若集中在特定產業或股價區間再加特徵。",
}


def _load_actual(target):
    con = db.connect()
    px = pd.read_sql("SELECT code,name,ret,close,open,high,low,value FROM prices WHERE date=?", con, params=(target,))
    inst = pd.read_sql("SELECT code,foreign_net,trust_net,dealer_net,total_net FROM inst WHERE date=?", con, params=(target,))
    idx = dict(con.execute("SELECT name,pct FROM index_daily WHERE date=?", (target,)).fetchall())
    info = pd.read_sql("SELECT code,industry FROM stock_info", con)
    exd = pd.read_sql("SELECT code,kind,amount FROM exdiv WHERE date=?", con, params=(target,))
    con.close()
    return px.set_index("code"), inst.set_index("code"), idx, info.set_index("code").industry, exd.set_index("code")


def _us_between(pred_date, target):
    """US sessions after Taiwan close of pred_date and before Taiwan open of target."""
    con = db.connect()
    df = pd.read_sql("SELECT date,symbol,ret FROM us_daily WHERE date>=? AND date<?", con, params=(pred_date, target))
    con.close()
    if df.empty:
        return {}, []
    out = {s: float((1 + g.ret.fillna(0)).prod() - 1) for s, g in df.groupby("symbol")}
    return out, sorted(df.date.unique())


def pending_predictions():
    done = {p.stem.replace("review_", "") for p in REVIEW_REPORT_DIR.glob("review_*.md")}
    tdays = set(F.trading_days())
    out = []
    for p in sorted(PRED_DIR.glob("*.json")):
        if p.stem in done:
            continue
        o = json.loads(p.read_text())
        t = o.get("target_date")
        # target may have shifted (e.g. typhoon day): use first trading day after pred_date with data
        later = sorted(d for d in tdays if d > o["pred_date"])
        if later:
            out.append((p.stem, later[0]))
    return out


def run(pred_date: dt.date = None, update=True, news=True):
    if update:
        from . import fetch
        end = dt.date.today()
        fetch.update(end - dt.timedelta(days=7), end, verbose=False, extras=False)
        fetch.us_markets("3mo")
    if pred_date:
        later = sorted(d for d in F.trading_days() if d > pred_date.isoformat())
        if not later:
            print("目標交易日尚無資料，稍後再檢討。")
            return
        jobs = [(pred_date.isoformat(), later[0])]
    else:
        jobs = pending_predictions()
        if not jobs:
            print("沒有待檢討的預測（或目標日資料尚未公布）。")
    for pd_, tgt in jobs:
        review_one(pd_, tgt, news=news)


def classify(kind, s, ctx, drivers, news_items, inst_row):
    """Return list of (cause, detail)."""
    causes = []
    mkt, us = ctx["mkt"], ctx["us"]
    sox, tsm, ndx = us.get("sox"), us.get("tsm_adr"), us.get("nasdaq")
    is_tech = s["industry"] in TECH_INDUSTRIES
    sign = -1 if kind == "wrong_pick" else 1
    exc, sec_exc = s["excess"], s["sector_excess"]
    if mkt is not None and sign * mkt >= 0.01:
        causes.append(("大盤系統性", f"加權指數 {mkt:+.2%}"))
    if sox is not None:
        hit = (sign * sox >= 0.02 and is_tech) or (ndx is not None and sign * ndx >= 0.015) or \
              (tsm is not None and sign * tsm >= 0.025 and is_tech)
        if hit:
            causes.append(("美股隔夜", f"預測後美股：費半 {sox:+.2%}、台積電ADR {(tsm or 0):+.2%}、那指 {(ndx or 0):+.2%}"))
    if sec_exc is not None and sign * sec_exc >= 0.01 and abs(sec_exc) >= 0.4 * abs(exc):
        causes.append(("產業輪動", f"{s['industry']} 產業平均超額 {sec_exc:+.2%}，占個股超額 {exc:+.2%} 的大部分"))
    if s.get("exdiv"):
        causes.append(("除權息", s["exdiv"]))
    neg_tags = {"利空事件", "處置/注意股"}
    pos_tags = {"利多事件", "訂單/產品", "法人報告/評等", "AI/半導體題材"}
    wrong_dir = pos_tags if kind == "wrong_pick" else neg_tags   # tags that cannot explain this direction
    tags = Counter(t for n in (news_items or []) for t in n["tags"] if t not in wrong_dir)
    if tags.get("法說會"):
        causes.append(("法說會", "新聞提及法說會/股東會"))
    if tags.get("營收") or (ctx["target_day"] <= 10 and s.get("rev_note")):
        causes.append(("營收公布", "新聞提及營收" if tags.get("營收") else f"月營收公布期間；{s.get('rev_note')}"))
    other = [t for t in tags if t not in ("法說會", "營收")]
    if other:
        causes.append(("新聞/事件", "、".join(f"{t}×{c}" for t, c in tags.items() if t in other)))
    dcat = Counter(F.category(d["feature"]).split("-")[0] for d in drivers[:5])
    if kind == "wrong_pick":
        if (dcat.get("技術", 0) + dcat.get("相對強弱", 0) >= 3) and (s["pred_r1"] or 0) >= 0.04:
            causes.append(("追高反轉", f"模型主要依技術動能看多，預測日已漲 {s['pred_r1']:+.2%}"))
        if dcat.get("籌碼", 0) >= 2 and inst_row is not None and (inst_row.get("total_net") or 0) < 0:
            causes.append(("籌碼反轉", f"看多主因為籌碼，但隔日三大法人轉賣超 {inst_row['total_net'] / 1000:,.0f} 張"))
    else:
        if s["ret"] >= 0.095:
            causes.append(("漲停/題材", f"目標日漲停 {s['ret']:+.2%}"))
        if inst_row is not None and s.get("avg_vol") and (inst_row.get("total_net") or 0) > 0.1 * s["avg_vol"]:
            causes.append(("法人同步買超", f"目標日三大法人買超 {inst_row['total_net'] / 1000:,.0f} 張"))
    if not causes:
        causes.append(("個股特有/無法解釋", f"個股特有部分 {s['idio']:+.2%}"))
    return causes


def review_one(pred_date, target, news=True):
    o = json.loads((PRED_DIR / f"{pred_date}.json").read_text())
    preds = pd.DataFrame(o["predictions"]).set_index("code")
    px, inst, idx, industry, exd = _load_actual(target)
    mkt = idx.get("TAIEX")
    otc = idx.get("TPEX")
    us, us_dates = _us_between(pred_date, target)
    df = preds.join(px[["ret", "close", "open", "value"]].rename(columns={"close": "t_close", "open": "t_open"}), how="left")
    df = df[df.ret.notna()].copy()
    df["excess"] = df.ret - (mkt or 0)
    df["sec_mean"] = df.groupby("industry").ret.transform("mean")
    df["sector_excess"] = df.sec_mean - (mkt or 0)
    df["idio"] = df.ret - df.sec_mean
    df["y"] = (df.ret > df.ret.median()).astype(int)
    top_n = o["top_n"]
    top = df[df["rank"] <= top_n].sort_values("rank")
    ctx = {"mkt": mkt, "us": us, "target_day": dt.date.fromisoformat(target).day}
    # metrics
    try:
        auc = float(roc_auc_score(df.y, df.prob))
    except ValueError:
        auc = None
    ic = float(df.prob.corr(df.excess, method="spearman"))
    dec = pd.qcut(df.prob.rank(method="first"), 10, labels=False)
    top_dec, bot_dec = df.excess[dec == 9].mean(), df.excess[dec == 0].mean()
    breadth = float((df.ret > 0).mean())
    perf = {"pred_date": pred_date, "target_date": target, "n_universe": len(df),
            "top_hit_abs": float((top.ret > 0).mean()), "top_hit_excess": float((top.excess > 0).mean()),
            "top_avg_ret": float(top.ret.mean()), "top_avg_excess": float(top.excess.mean()),
            "universe_avg_ret": float(df.ret.mean()), "auc": auc, "rank_ic": ic,
            "top_decile_excess": float(top_dec), "bottom_decile_excess": float(bot_dec),
            "taiex": mkt, "otc": otc, "breadth": breadth,
            "us_sox": us.get("sox"), "us_tsm": us.get("tsm_adr")}

    # monthly revenue context
    con = db.connect()
    rev = pd.read_sql("SELECT code,ym,yoy FROM revenue WHERE ym=(SELECT MAX(ym) FROM revenue)", con).set_index("code")
    avg_vol = dict(con.execute("SELECT code, AVG(volume) FROM prices WHERE date<=? AND date>=date(?, '-30 day') "
                               "GROUP BY code", (pred_date, pred_date)).fetchall())
    con.close()

    def stock_info(code, r):
        e = exd.loc[code] if code in exd.index else None
        if isinstance(e, pd.DataFrame):
            e = e.iloc[0]
        return {"code": code, "name": r["name"], "industry": r["industry"], "rank": int(r["rank"]),
                "prob": float(r["prob"]), "ret": float(r.ret), "excess": float(r.excess),
                "sector_excess": float(r.sector_excess), "idio": float(r.idio), "pred_r1": r.get("r1"),
                "exdiv": f"目標日除{e['kind']}（{e['amount']}元）" if e is not None else None,
                "rev_note": (f"最新月營收YoY {rev.loc[code, 'yoy']:+.1f}%（{rev.loc[code, 'ym']}）"
                             if code in rev.index and pd.notna(rev.loc[code, "yoy"]) else None),
                "avg_vol": avg_vol.get(code)}

    analyses = []
    wrong = top[(top.excess < 0) | (top.ret < 0)]
    missed = df[(df["rank"] > top_n)].sort_values("excess", ascending=False)
    missed = missed[missed.excess > 0.03].head(10)
    major_drops = top[top.ret <= -0.03]
    targets = [("wrong_pick", c, r) for c, r in wrong.iterrows()] + [("missed_winner", c, r) for c, r in missed.iterrows()]
    t0 = dt.date.fromisoformat(pred_date)
    t1 = dt.date.fromisoformat(target)
    for kind, code, r in targets:
        s = stock_info(code, r)
        drivers = r["factors_pos"] if kind == "wrong_pick" else r["factors_neg"]
        items = N.headlines(r["name"], code, t0, t1) if news else []
        ir = inst.loc[code].to_dict() if code in inst.index else None
        causes = classify(kind, s, ctx, drivers, items, ir)
        analyses.append({"kind": kind, **s, "drivers": drivers, "news": items or [], "causes": causes,
                         "inst_target": ir, "major": kind == "wrong_pick" and code in major_drops.index})

    # sectors
    sec = df.groupby("industry").agg(n=("ret", "size"), ret=("ret", "mean"), ex=("excess", "mean"),
                                     prob=("prob", "mean")).query("n>=3").sort_values("ex")
    md = render(o, perf, top, analyses, sec, us, us_dates, df)
    path = REVIEW_REPORT_DIR / f"review_{pred_date}.md"
    path.write_text(md)
    _append_logs(perf, analyses, pred_date, target)
    write_patterns()
    print(md)
    print(f"\n檢討報告：{path}\n累積紀錄：{LOG}\n錯誤模式統計：{PATTERNS}")


def _p(x, signed=True):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "NA"
    return f"{x * 100:+.2f}%" if signed else f"{x * 100:.1f}%"


def render(o, perf, top, analyses, sec, us, us_dates, df):
    L = [f"# 預測檢討 — {perf['pred_date']} 的預測 vs {perf['target_date']} 實際結果", "", DISCLAIMER, "",
         "## 總結", "",
         f"- Top {o['top_n']}：上漲 {int(round(perf['top_hit_abs'] * len(top)))}/{len(top)}（{_p(perf['top_hit_abs'], False)}），"
         f"勝過大盤 {int(round(perf['top_hit_excess'] * len(top)))}/{len(top)}（{_p(perf['top_hit_excess'], False)}）",
         f"- Top {o['top_n']} 平均報酬 {_p(perf['top_avg_ret'])}，平均超額 {_p(perf['top_avg_excess'])}；全體平均 {_p(perf['universe_avg_ret'])}",
         f"- 全體 {perf['n_universe']} 檔：AUC(跑贏中位數) {perf['auc']:.3f}，Rank IC {perf['rank_ic']:+.3f}；"
         f"最高十分位超額 {_p(perf['top_decile_excess'])} vs 最低十分位 {_p(perf['bottom_decile_excess'])}",
         f"- 市場：加權 {_p(perf['taiex'])}、櫃買 {_p(perf['otc'])}、上漲家數比例 {_p(perf['breadth'], False)}",
         f"- 預測後美股（{', '.join(us_dates) or '無'}）：費半 {_p(us.get('sox'))}、台積電ADR {_p(us.get('tsm_adr'))}、"
         f"那指 {_p(us.get('nasdaq'))}、S&P500 {_p(us.get('spx'))}", "",
         "## Top 名單逐檔結果", "",
         "| # | 代號 | 名稱 | 產業 | 機率 | 實際報酬 | 超額 | 產業超額 | 個股特有 | 結果 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for c, r in top.iterrows():
        res = "✅" if r.excess > 0 and r.ret > 0 else ("🟡 勝大盤但下跌" if r.excess > 0 else ("🔴 大幅錯誤" if r.ret <= -0.03 else "❌"))
        L.append(f"| {int(r['rank'])} | {c} | {r['name']} | {r['industry']} | {r['prob']:.1%} | {_p(r.ret)} | "
                 f"{_p(r.excess)} | {_p(r.sector_excess)} | {_p(r.idio)} | {res} |")
    L += ["", "報酬拆解：實際報酬 = 加權指數 + 產業超額（同產業平均−大盤）+ 個股特有（個股−產業平均）。", ""]
    wr = [a for a in analyses if a["kind"] == "wrong_pick"]
    mw = [a for a in analyses if a["kind"] == "missed_winner"]
    L += ["## 判斷錯誤的個股：根因分析", ""]
    if not wr:
        L.append("Top 名單全部上漲且勝過大盤。")
    for a in wr:
        L += _stock_block(a, "看多主因（事後看是誤判的依據）")
    L += ["", "## 漏掉的大漲股（不在 Top 名單、超額報酬 > 3%）", ""]
    if not mw:
        L.append("沒有超額報酬 > 3% 的漏網之魚。")
    for a in mw:
        L += _stock_block(a, "讓模型不看好的因素")
    L += ["", "## 產業表現（目標日，依平均超額排序）", "", "| 產業 | 檔數 | 平均報酬 | 平均超額 | 模型平均機率 |", "|---|---|---|---|---|"]
    for ind, r in pd.concat([sec.head(5), sec.tail(5)]).drop_duplicates().iterrows():
        L.append(f"| {ind} | {int(r.n)} | {_p(r.ret)} | {_p(r.ex)} | {r.prob:.1%} |")
    cnt = Counter(c for a in analyses for c, _ in a["causes"])
    L += ["", "## 本次錯誤/遺漏原因統計與改進建議", ""]
    for c, n in cnt.most_common():
        L.append(f"- **{c}** ×{n}：{SUGGEST.get(c, '')}")
    L += ["", f"累積統計見 `{PATTERNS}`。"]
    return "\n".join(L) + "\n"


def _stock_block(a, driver_title):
    tag = "（🔴 跌幅≥3%）" if a.get("major") else ""
    L = [f"### {a['code']} {a['name']}（{a['industry']}）{tag}",
         f"- 預測排名 #{a['rank']}，機率 {a['prob']:.1%}；實際 {_p(a['ret'])}，超額 {_p(a['excess'])}，"
         f"產業超額 {_p(a['sector_excess'])}，個股特有 {_p(a['idio'])}",
         f"- {driver_title}：" + ("；".join(f"{d['label']}={d['value_str']}（{d['contrib']:+.3f}）" for d in a["drivers"][:4]) or "無")]
    if a.get("inst_target"):
        i = a["inst_target"]
        L.append(f"- 目標日法人：外資 {(i.get('foreign_net') or 0) / 1000:+,.0f} 張、投信 {(i.get('trust_net') or 0) / 1000:+,.0f} 張、"
                 f"自營 {(i.get('dealer_net') or 0) / 1000:+,.0f} 張")
    L.append("- 判定原因：" + "；".join(f"**{c}**（{d}）" for c, d in a["causes"]))
    if a["news"]:
        L.append("- 相關新聞（Google News）：")
        for n in a["news"][:4]:
            L.append(f"  - {n['time']} {n['title']} {('[' + ','.join(n['tags']) + ']') if n['tags'] else ''}")
    L.append("")
    return L


def _append_logs(perf, analyses, pred_date, target):
    pf = pd.DataFrame([perf])
    if PERF.exists():
        old = pd.read_csv(PERF, dtype={"pred_date": str})
        pf = pd.concat([old[old.pred_date != pred_date], pf])
    pf.to_csv(PERF, index=False)
    rows = [{"pred_date": pred_date, "target_date": target, "kind": a["kind"], "code": a["code"], "name": a["name"],
             "industry": a["industry"], "rank": a["rank"], "prob": round(a["prob"], 4), "ret": round(a["ret"], 4),
             "excess": round(a["excess"], 4), "sector_excess": round(a["sector_excess"], 4),
             "causes": ";".join(c for c, _ in a["causes"]),
             "driver_features": ";".join(d["feature"] for d in a["drivers"][:5]),
             "driver_categories": ";".join(F.category(d["feature"]) for d in a["drivers"][:5]),
             "news_tags": ";".join(sorted({t for n in a["news"] for t in n["tags"]}))} for a in analyses]
    lg = pd.DataFrame(rows)
    if LOG.exists():
        old = pd.read_csv(LOG, dtype={"pred_date": str, "code": str})
        lg = pd.concat([old[old.pred_date != pred_date], lg])
    lg.to_csv(LOG, index=False)


def write_patterns():
    if not LOG.exists():
        return
    lg = pd.read_csv(LOG, dtype={"code": str})
    pf = pd.read_csv(PERF) if PERF.exists() else pd.DataFrame()
    L = ["# 累積錯誤模式統計", "", DISCLAIMER, "",
         f"- 已檢討預測日數：{len(pf)}；記錄個股案例 {len(lg)} 筆（判斷錯誤 {int((lg.kind == 'wrong_pick').sum())}、漏掉大漲 {int((lg.kind == 'missed_winner').sum())}）"]
    if len(pf):
        L += [f"- 累積 Top-N 上漲比例 {pf.top_hit_abs.mean():.1%}、勝過大盤比例 {pf.top_hit_excess.mean():.1%}、"
              f"平均超額 {pf.top_avg_excess.mean() * 100:+.2f}%、平均 Rank IC {pf.rank_ic.mean():+.3f}",
              f"- 最近 20 次：Top-N 勝過大盤比例 {pf.tail(20).top_hit_excess.mean():.1%}"]
    L += ["", "## 原因類別出現次數", "", "| 原因 | 判斷錯誤 | 漏掉大漲 | 合計 | 建議新增/調整 |", "|---|---|---|---|---|"]
    ex = lg.assign(cause=lg.causes.fillna("").str.split(";")).explode("cause")
    tab = ex.groupby(["cause", "kind"]).size().unstack(fill_value=0)
    for k in ("wrong_pick", "missed_winner"):
        if k not in tab:
            tab[k] = 0
    tab["total"] = tab.sum(axis=1)
    for c, r in tab.sort_values("total", ascending=False).iterrows():
        L.append(f"| {c} | {r.wrong_pick} | {r.missed_winner} | {r.total} | {SUGGEST.get(c, '')} |")
    L += ["", "## 誤判時最常見的看多依據（特徵類別）", ""]
    dc = lg[lg.kind == "wrong_pick"].driver_categories.fillna("").str.split(";").explode()
    for c, n in dc[dc != ""].value_counts().head(10).items():
        L.append(f"- {c}：{n}")
    L += ["", "## 漏掉大漲股最常見的扣分特徵", ""]
    dm = lg[lg.kind == "missed_winner"].driver_features.fillna("").str.split(";").explode()
    for c, n in dm[dm != ""].value_counts().head(10).items():
        L.append(f"- {c}（{F.label(c)}）：{n}")
    L += ["", "## 新聞標籤分布", ""]
    nt = lg.news_tags.fillna("").str.split(";").explode()
    for c, n in nt[nt != ""].value_counts().items():
        L.append(f"- {c}：{n}")
    PATTERNS.write_text("\n".join(L) + "\n")
