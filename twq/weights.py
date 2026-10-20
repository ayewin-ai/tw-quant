"""Tunable weights (config/weights.json, committed) and the post-model score adjustments that use them."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PATH = ROOT / "config" / "weights.json"
HIST = ROOT / "config" / "history"

DEFAULT = {
    "version": 1,
    "updated": "2026-10-20",
    "model": {"learning_rate": 0.03, "num_leaves": 31, "min_child_samples": 300, "lambda_l2": 10.0,
              "feature_fraction": 0.7, "num_rounds": 300},
    "recency_half_life_days": None,          # None = all training days weighted equally
    "feature_set": "base",                   # close-time model feature set (see features.FEATURE_SETS)
    "preopen_feature_set": "preopen",
    "feature_group_multipliers": {},         # reserved: {"籌碼": 1.0, ...} multiplies contributions (not used yet)
    "score": {
        "limitup_penalty": 0.0,              # 前日漲停股扣分（機率單位）
        "gap_risk_penalty": 0.0,             # 前日大漲（r1>0 部分）× 係數 扣分：預防開高走低
        "max_per_industry": 10,              # Top-10 中同一產業最多幾檔（產業集中度上限）
        "overnight_strength": 0.0,           # 盤前：預期跳空(gap_expect) × 係數 扣分（跳空越大越不追）
    },
    "liquidity_min_avg_value": 50_000_000,   # Top 名單的 20 日平均成交值下限（元）
}

# allowed ranges and max per-day step for each tunable (guard rails)
BOUNDS = {
    "score.limitup_penalty": (0.0, 0.10, 0.02),
    "score.gap_risk_penalty": (0.0, 1.0, 0.2),
    "score.max_per_industry": (2, 10, 2),
    "score.overnight_strength": (0.0, 2.0, 0.5),
    "liquidity_min_avg_value": (50_000_000, 500_000_000, 100_000_000),
    "recency_half_life_days": (60, 720, None),
    "model.num_leaves": (15, 63, 16),
    "model.min_child_samples": (100, 1000, 200),
}


def load():
    if PATH.exists():
        w = json.loads(PATH.read_text())
        out = copy.deepcopy(DEFAULT)
        for k, v in w.items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k].update(v)
            else:
                out[k] = v
        return out
    return copy.deepcopy(DEFAULT)


def save(w):
    PATH.parent.mkdir(parents=True, exist_ok=True)
    PATH.write_text(json.dumps(w, ensure_ascii=False, indent=2) + "\n")
    HIST.mkdir(parents=True, exist_ok=True)
    (HIST / f"weights_v{w['version']}.json").write_text(json.dumps(w, ensure_ascii=False, indent=2) + "\n")


def get(w, key):
    v = w
    for k in key.split("."):
        v = v.get(k) if isinstance(v, dict) else None
    return v


def set_(w, key, val):
    w = copy.deepcopy(w)
    d = w
    ks = key.split(".")
    for k in ks[:-1]:
        d = d.setdefault(k, {})
    lo, hi, _ = BOUNDS.get(key, (None, None, None))
    if val is not None and lo is not None:
        val = type(lo)(min(max(val, lo), hi))
    d[ks[-1]] = val
    return w


def model_hash(w):
    m = {"model": w["model"], "hl": w.get("recency_half_life_days"), "fs": w.get("feature_set")}
    return hashlib.md5(json.dumps(m, sort_keys=True).encode()).hexdigest()[:10]


def adjust(df: pd.DataFrame, w, top_n=10, preopen=False):
    """df: one day's cross-section with prob, r1, industry, log_value20 (+ gap_expect for pre-open).
    Returns df with 'score' and 'rank' (industry cap + liquidity applied to the Top-N selection)."""
    s = w["score"]
    df = df.copy()
    sc = df["prob"].astype(float).copy()
    r1 = df["r1"].fillna(0)
    sc -= s.get("limitup_penalty", 0) * (r1 >= 0.095)
    sc -= s.get("gap_risk_penalty", 0) * r1.clip(lower=0)
    if preopen and "gap_expect" in df and s.get("overnight_strength", 0):
        sc -= s["overnight_strength"] * df["gap_expect"].fillna(0).abs() * 0.1
    liq = w.get("liquidity_min_avg_value") or 0
    if liq and "log_value20" in df:
        sc -= 1.0 * (np.exp(df["log_value20"].fillna(0)) < liq)
    df["score"] = sc
    df = df.sort_values("score", ascending=False)
    cap = int(s.get("max_per_industry", 10) or 10)
    if cap < top_n:
        cnt, top, rest = {}, [], []
        for i, ind in zip(df.index, df["industry"].fillna("")):
            if len(top) < top_n and cnt.get(ind, 0) < cap:
                top.append(i)
                cnt[ind] = cnt.get(ind, 0) + 1
            else:
                rest.append(i)
        df = df.loc[top + rest]
    df["rank"] = np.arange(1, len(df) + 1)
    return df
