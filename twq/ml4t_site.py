"""把第三批參考的評估結果（models/ml4t_eval.json）整理成網站用的 docs/data/ml4t.json。"""
import json

from . import ml4t_eval as E

NAMES = {
    "+ta": "技術指標庫 ta_*（25，Stock_feature_engineering/ML4T）", "+signal": "訊號處理 sg_*（10：Hurst、頻譜、排列熵、Kalman…）",
    "+qr": "QuantResearch 風格 qr_*（12：配對/殘差/市場狀態）", "all_ml4t": "三組全加（47 個）",
    "preopen_ta": "技術指標庫 ta_*", "preopen_signal": "訊號處理 sg_*", "preopen_qr": "QuantResearch 風格 qr_*", "preopen_ml4t": "三組全加",
    "tail30": "尾端標籤（只用前/後 30% 訓練）", "tb05": "波動門檻標籤（三重障礙 1 日版，±0.5σ）",
    "rank_reg": "排名回歸（高斯化排名）", "vn_reg": "波動標準化報酬回歸",
    "icir60": "折內 ICIR 篩選（保留 60 個特徵）", "noise": "雜訊特徵篩選（Boruta-lite）",
    "embargo5": "隔離訓練窗尾端 5 日（洩漏檢查）", "shuffled": "洗牌標籤對照（placebo，應該完全失效）",
    "secfix": "產業平均只算已上市股票（截斷稽核發現的小修正）",
}
GROUPS = {"feat": ["+ta", "+signal", "+qr", "all_ml4t", "preopen_ta", "preopen_signal", "preopen_qr", "preopen_ml4t"],
          "label": ["tail30", "tb05", "rank_reg", "vn_reg"], "select": ["icir60", "noise"], "audit": ["embargo5", "shuffled", "secfix"]}


def _row(model, name, d):
    r = d.get(f"{model}|{name}")
    if not r:
        return None
    s, lg = r["short"], r["long"]
    return {"model": model, "name": name, "label": NAMES.get(name, name), "n_features": r["n_features"],
            "net_oc_bp": s["net_oc_bp"], "excess_bp": s["excess_bp"], "auc_oc": s["auc_oc"], "h1": s["net_h1_bp"], "h2": s["net_h2_bp"],
            "long_auc": lg["auc"], "long_ic": lg["rank_ic"], "long_top_bp": lg["top_avg_oc_bp"],
            "short_msg": r.get("verdict_msg"), "long_msg": r.get("long_msg"), "enable": bool(r.get("enable"))}


def payload():
    d = E._done() if E.PATH.exists() else {}
    if not d:
        return None
    out = {"champions": {m: _row(m, "champion", d) for m in ("close", "preopen")}, "groups": {}}
    for g, names in GROUPS.items():
        out["groups"][g] = [x for m in ("close", "preopen") for n in names if (x := _row(m, n, d))]
    for k in ("truncation", "calibration", "ic_decay_close", "ic_table"):
        if f"extra|{k}" in d:
            out[k] = d[f"extra|{k}"]
    out["meta"] = {m: d[f"extra|meta|{m}"] for m in ("close", "preopen") if f"extra|meta|{m}" in d}
    if "extra|ic_table" in d:   # 只留 |t| 最大的新特徵，網站才不會太長
        new = set(__import__("twq.features_ta", fromlist=["ALL_F"]).ALL_F)
        top = {}
        for tgt, nm in (("cc", "收盤→隔收（close model 目標）"), ("oc", "開→收（盤前 model 目標）")):
            rows = [{"feature": k, **v} for k, v in d["extra|ic_table"][tgt].items() if k in new]
            rows.sort(key=lambda x: -abs(x["t"]))
            top[tgt] = {"name": nm, "rows": rows[:8]}
        out["ic_top"] = top
        out.pop("ic_table", None)
    out["enabled"] = [f"{m}|{n}" for m in ("close", "preopen") for n in sum(GROUPS.values(), []) if d.get(f"{m}|{n}", {}).get("enable")]
    return out
