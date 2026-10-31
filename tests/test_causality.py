"""因果性（無未來資訊）單元測試：截斷到某日重算，該日以前的特徵必須與完整資料算出來完全一致。
可直接 `python tests/test_causality.py` 執行（不需 pytest）；pytest 也能收集。
實務概念參考 jpmorganchase/python-training 的測試習慣；整條管線版本見 `python -m twq.ml4t_extra trunc`。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from twq import features_ta as T  # noqa: E402


def _panel(n=400, k=12, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.RangeIndex(n)
    cols = [f"c{i}" for i in range(k)]
    r = pd.DataFrame(rng.normal(0, 0.02, (n, k)), index=idx, columns=cols)
    close = 50 * (1 + r).cumprod()
    opn = close.shift(1) * (1 + rng.normal(0, 0.005, (n, k)))
    high = np.maximum(close, opn) * (1 + rng.uniform(0, 0.01, (n, k)))
    low = np.minimum(close, opn) * (1 - rng.uniform(0, 0.01, (n, k)))
    vol = pd.DataFrame(rng.integers(1000, 9000, (n, k)).astype(float), index=idx, columns=cols)
    traded = pd.DataFrame(True, index=idx, columns=cols)
    adj = (1 + r).cumprod()
    return dict(adj=adj, close=close, opn=opn, high=high, low=low, vol=vol, val=vol * close, r=r, traded=traded)


def _cut(d, t):
    return {k: v.iloc[:t] for k, v in d.items()}


def _same(a, b, name):
    a, b = a.iloc[: len(b)], b
    bad = ~(((a - b).abs() < 1e-9) | (a.isna() & b.isna()))
    assert not bad.values.any(), f"{name}: 截斷後數值改變（有未來資訊）"


def test_ta_causal():
    d = _panel()
    full = T.ta_feats(**d)
    part = T.ta_feats(**_cut(d, 300))
    for k in part:
        _same(full[k], part[k], k)


def test_sig_causal():
    d = _panel()
    full = T.sig_feats(d["adj"], d["r"], d["traded"])
    part = T.sig_feats(d["adj"].iloc[:300], d["r"].iloc[:300], d["traded"].iloc[:300])
    for k in part:
        _same(full[k], part[k], k)


if __name__ == "__main__":
    test_ta_causal()
    test_sig_causal()
    print("ok: ta/sig features are causal")


def test_151_causal():
    from twq import features_151 as K
    d = _panel()
    ind = pd.Series(["a", "b", "c"] * 4, index=d["r"].columns)
    mkt = d["r"].mean(axis=1)
    args = lambda t: (d["adj"].iloc[:t], d["close"].iloc[:t], d["high"].iloc[:t], d["low"].iloc[:t],  # noqa: E731
                      d["r"].iloc[:t], mkt.iloc[:t], ind)
    full, part = K.feats_151(*args(400)), K.feats_151(*args(330))
    for k in part:
        _same(full[k], part[k], k)
    print("ok: 151 features are causal")


if __name__ == "__main__":
    test_151_causal()
