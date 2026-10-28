"""第三批外部參考（LinkedIn 分享清單）衍生的特徵組：技術指標庫、訊號處理、統計套利／因子殘差。

全部依公式自行以 numpy/pandas/scipy 重寫（沒有複製任何外部程式碼），只用 t 日（含）以前的資料（因果濾波，不用 filtfilt）。
概念出處（詳見網站「第三批參考」區與 reports/CHANGELOG_model.md）：
  TA   ← stefan-jansen/machine-learning-for-trading (MIT) 第 8 章 alpha 因子庫、hjeffreywang/Stock_feature_engineering
          (無授權，僅借概念：MACD/布林/ADX/MFI/CCI/ATR…)、Alpha101 風格量價因子、Amihud 流動性；
          VisualHFT 需要逐筆/委託簿，這裡只做日線可算的「資金流向代理」（CMF、OBV、上漲量占比、隔夜/盤中報酬拆解）。
  SIG  ← SparkAbhi/SignalProcessingWithPython（低通濾波、FFT 頻譜）、ML4T 第 9 章（Kalman、頻域特徵）、
          Hurst 指數／變異數比、排列熵。
  QR   ← letianzj/QuantResearch (MIT)：Kalman 配對交易、共整合/均值回歸（OU）、Fama-French 型因子殘差（特質動能）、
          波動 regime。
"""
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter, lfilter_zi

INFO = {
    # --- 技術指標庫 (TA)
    "ta_macd": ("TA-指標", "MACD 柱狀(12,26,9)/價"),
    "ta_bbp": ("TA-指標", "布林 %b(20,2)"),
    "ta_bbw": ("TA-指標", "布林帶寬(20,2)"),
    "ta_stoch_k": ("TA-指標", "KD 的 K(14)"),
    "ta_stoch_d": ("TA-指標", "KD 的 D(3日平均 K)"),
    "ta_cci": ("TA-指標", "CCI(20)"),
    "ta_atr": ("TA-指標", "ATR(14)/價"),
    "ta_adx": ("TA-指標", "ADX(14)"),
    "ta_di_diff": ("TA-指標", "+DI − −DI (14)"),
    "ta_mfi": ("TA-指標", "資金流量指標 MFI(14)"),
    "ta_cmf": ("TA-資金流", "Chaikin 資金流 CMF(20)"),
    "ta_obv10": ("TA-資金流", "OBV 10日斜率(量加權方向)"),
    "ta_upvol20": ("TA-資金流", "20日上漲日成交量占比"),
    "ta_clv5": ("TA-資金流", "收盤位置 5日平均"),
    "ta_trix": ("TA-指標", "TRIX(15)"),
    "ta_amihud": ("TA-流動性", "Amihud 非流動性(20日,log)"),
    "ta_park20": ("TA-波動", "Parkinson 波動(20日)"),
    "ta_gk20": ("TA-波動", "Garman-Klass 波動(20日)"),
    "ta_ac1": ("TA-統計", "報酬 1 階自相關(20日)"),
    "ta_cpv": ("TA-統計", "報酬與成交量相關(10日, Alpha6 型)"),
    "ta_a101": ("TA-Alpha", "(收−開)/(高−低)（Alpha101 #101）"),
    "ta_a12": ("TA-Alpha", "−sign(Δ量)×當日報酬（Alpha101 #12 型）"),
    "ta_gap10": ("TA-隔夜/盤中", "10日累積隔夜報酬(收→開)"),
    "ta_intra10": ("TA-隔夜/盤中", "10日累積盤中報酬(開→收)"),
    "ta_gap_share": ("TA-隔夜/盤中", "20日隔夜占總波動比"),
    # --- 訊號處理 (SIG)
    "sg_hurst": ("訊號-Hurst", "Hurst 指數(變異數比, 60日)"),
    "sg_vr5": ("訊號-Hurst", "5日變異數比(60日)"),
    "sg_spec_ent": ("訊號-頻譜", "32日頻譜熵(FFT)"),
    "sg_lf_ratio": ("訊號-頻譜", "低頻(週期≥8日)能量占比"),
    "sg_dom_period": ("訊號-頻譜", "主頻週期(日)"),
    "sg_perm_ent": ("訊號-熵", "排列熵(階3,20日)"),
    "sg_kal_slope": ("訊號-Kalman", "Kalman 趨勢斜率/σ"),
    "sg_kal_dev": ("訊號-Kalman", "Kalman 新息(價−預測)/σ"),
    "sg_lp_dev": ("訊號-低通", "價−因果低通(巴特沃斯2階,週期10日)/σ"),
    "sg_lp_slope": ("訊號-低通", "低通濾波 3日斜率/σ"),
    # --- 量化研究 / 統計套利 (QR)
    "qr_resid_z": ("QR-殘差", "20日特質報酬 z（扣市場beta與產業）"),
    "qr_resid_mom": ("QR-殘差", "特質動能(60日扣近5日)"),
    "qr_resid_vol": ("QR-殘差", "特質波動(60日)"),
    "qr_ou": ("QR-均值回歸", "OU 回歸係數(60日,越負越強)"),
    "qr_pair_z": ("QR-配對", "與同產業最相關夥伴價差 z(60日)"),
    "qr_pair_gap1": ("QR-配對", "夥伴前一日報酬−自己"),
    "qr_pair_gap5": ("QR-配對", "夥伴5日報酬−自己"),
    "qr_pair_corr": ("QR-配對", "與夥伴相關係數(120日)"),
    "qr_mkt_vol_pct": ("QR-regime", "大盤波動分位(擴張視窗)"),
    "qr_mkt_dd": ("QR-regime", "大盤距120日高點"),
    "qr_mkt_hurst": ("QR-regime", "大盤 Hurst(變異數比)"),
}
TA_F = [k for k in INFO if k.startswith("ta_")]
SIG_F = [k for k in INFO if k.startswith("sg_")]
QR_F = [k for k in INFO if k.startswith("qr_")]
QR_STOCK_F = [k for k in QR_F if not k.startswith("qr_mkt")]
QR_MKT_F = [k for k in QR_F if k.startswith("qr_mkt")]
ALL_F = TA_F + SIG_F + QR_F


def _roll(df, n, minp=None):
    return df.rolling(n, min_periods=minp or n)


def _wilder(df, n):
    return df.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def ta_feats(adj, close, opn, high, low, vol, val, r, traded):
    """adj: 還原指數；close/opn/high/low: 原始價 (days x codes)；vol/val: 量/值；r: 日報酬(未交易=0)。"""
    s = (adj / close).ffill()           # 還原係數：把 OHLC 轉成同一尺度，避免除權息造成假跳空
    c = adj
    o, h, l = (opn * s).where(traded), (high * s).where(traded), (low * s).where(traded)
    h = np.maximum(h, c)
    l = np.minimum(l, c)
    f = {}
    # MACD / Bollinger
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    f["ta_macd"] = ((macd - macd.ewm(span=9, adjust=False).mean()) / c).clip(-0.2, 0.2)
    ma20, sd20 = _roll(c, 20).mean(), _roll(c, 20).std()
    f["ta_bbp"] = ((c - (ma20 - 2 * sd20)) / (4 * sd20)).clip(-1, 2)
    f["ta_bbw"] = (4 * sd20 / ma20).clip(0, 1)
    # Stochastic (KD)
    ll14, hh14 = _roll(l, 14, 10).min(), _roll(h, 14, 10).max()
    k = ((c - ll14) / (hh14 - ll14)).where(hh14 > ll14, 0.5)
    f["ta_stoch_k"] = k
    f["ta_stoch_d"] = _roll(k, 3, 2).mean()
    # CCI(20)
    tp = (h + l + c) / 3
    mad = _roll(tp, 20).apply(lambda a: np.mean(np.abs(a - a.mean())), raw=True)
    f["ta_cci"] = ((tp - _roll(tp, 20).mean()) / (0.015 * mad)).clip(-300, 300)
    # ATR / ADX (Wilder)
    pc = c.shift(1)
    tr = np.maximum(h - l, np.maximum((h - pc).abs(), (l - pc).abs()))
    atr = _wilder(tr, 14)
    f["ta_atr"] = (atr / c).clip(0, 0.5)
    up, dn = h - h.shift(1), l.shift(1) - l
    pdm = up.where((up > dn) & (up > 0), 0.0)
    mdm = dn.where((dn > up) & (dn > 0), 0.0)
    pdi, mdi = 100 * _wilder(pdm, 14) / atr, 100 * _wilder(mdm, 14) / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    f["ta_adx"] = _wilder(dx, 14)
    f["ta_di_diff"] = (pdi - mdi)
    # MFI(14)
    mf = tp * val
    pos = mf.where(tp > tp.shift(1), 0.0)
    neg = mf.where(tp < tp.shift(1), 0.0)
    f["ta_mfi"] = 100 * _roll(pos, 14).sum() / (_roll(pos, 14).sum() + _roll(neg, 14).sum()).replace(0, np.nan)
    # CMF / OBV / up-volume share  — 日線版「資金流向代理」（VisualHFT 需逐筆資料，這裡只能用 OHLCV 近似）
    clv = ((2 * c - l - h) / (h - l)).where(h > l, 0.0)
    f["ta_cmf"] = _roll(clv * val, 20).sum() / _roll(val, 20).sum().replace(0, np.nan)
    sv = np.sign(r) * vol.fillna(0)
    f["ta_obv10"] = _roll(sv, 10).sum() / _roll(vol.fillna(0), 10).sum().replace(0, np.nan)
    upv = vol.fillna(0).where(r > 0, 0.0)
    f["ta_upvol20"] = _roll(upv, 20).sum() / _roll(vol.fillna(0), 20).sum().replace(0, np.nan)
    f["ta_clv5"] = _roll(clv, 5).mean()
    # TRIX
    e3 = np.log(c).ewm(span=15, adjust=False).mean().ewm(span=15, adjust=False).mean().ewm(span=15, adjust=False).mean()
    f["ta_trix"] = (e3.diff() * 100).clip(-5, 5)
    # Amihud illiquidity
    ami = (r.abs() / val.replace(0, np.nan)).where(traded)
    f["ta_amihud"] = np.log1p(1e9 * _roll(ami, 20, 10).mean())
    # range-based volatility
    lhl = np.log(h / l)
    f["ta_park20"] = np.sqrt(_roll(lhl ** 2, 20).mean() / (4 * np.log(2)))
    gk = 0.5 * lhl ** 2 - (2 * np.log(2) - 1) * np.log(c / o) ** 2
    f["ta_gk20"] = np.sqrt(_roll(gk, 20).mean().clip(lower=0))
    # statistics
    f["ta_ac1"] = _roll(r, 20).corr(r.shift(1)).clip(-1, 1)
    f["ta_cpv"] = _roll(r, 10).corr(vol.fillna(0)).clip(-1, 1)
    f["ta_a101"] = ((c - o) / (h - l)).where(h > l, 0.0).clip(-1, 1)
    f["ta_a12"] = -np.sign(vol.diff()) * r
    # overnight vs intraday decomposition (收→開 / 開→收)
    lg = np.log(o / c.shift(1)).where(traded)          # 隔夜（還原尺度）
    li = np.log(c / o).where(traded)                   # 盤中
    lg = lg.clip(-0.2, 0.2)
    li = li.clip(-0.2, 0.2)
    f["ta_gap10"] = _roll(lg, 10, 6).sum()
    f["ta_intra10"] = _roll(li, 10, 6).sum()
    gv, iv = _roll(lg ** 2, 20, 12).mean(), _roll(li ** 2, 20, 12).mean()
    f["ta_gap_share"] = gv / (gv + iv).replace(0, np.nan)
    return f


def _hurst_vr(r, win=60):
    """以 τ=1,2,4,8 的 τ 日報酬標準差對 τ 做 log-log 斜率估 Hurst；回傳 (hurst, vr5)。"""
    lags = (1, 2, 4, 8)
    ls = []
    for t in lags:
        st = r.rolling(t).sum().rolling(win, min_periods=int(win * 0.8)).std()
        ls.append(np.log(st.replace(0, np.nan)))
    x = np.log(np.array(lags, float))
    xm = x.mean()
    num = sum((x[i] - xm) * ls[i] for i in range(len(lags)))
    den = ((x - xm) ** 2).sum()
    hurst = (num / den).clip(0, 1)
    v1 = r.rolling(win, min_periods=int(win * 0.8)).var()
    s5 = r.rolling(5).sum().rolling(win, min_periods=int(win * 0.8)).var()
    vr5 = (s5 / (5 * v1.replace(0, np.nan))).clip(0, 4)
    return hurst, vr5


def _spectral(r, n=32):
    """滾動 n 日 FFT（Hann 窗）：頻譜熵、低頻占比、主頻週期。逐欄、因果（窗內只含 t 以前）。"""
    T, N = r.shape
    a = r.values.astype(np.float32)
    ent = np.full((T, N), np.nan, np.float32)
    lf = np.full((T, N), np.nan, np.float32)
    dom = np.full((T, N), np.nan, np.float32)
    win = np.hanning(n).astype(np.float32)
    nb = n // 2 + 1
    chunk = 120
    for t0 in range(n - 1, T, chunk):
        t1 = min(T, t0 + chunk)
        idx = np.arange(t0 - n + 1, t1)
        sub = a[idx]                                   # (len, N)
        w = np.lib.stride_tricks.sliding_window_view(sub, n, axis=0)   # (len-n+1, N, n)
        w = w - w.mean(axis=2, keepdims=True)
        p = np.abs(np.fft.rfft(w * win, axis=2)) ** 2   # (m, N, nb)
        p = p[:, :, 1:]                                # drop DC
        tot = p.sum(axis=2) + 1e-20
        q = p / tot[:, :, None]
        e = -(q * np.log(q + 1e-12)).sum(axis=2) / np.log(q.shape[2])
        low = p[:, :, :4].sum(axis=2) / tot            # bins 1..4 ↔ 週期 32,16,10.7,8
        d = n / (p.argmax(axis=2) + 1)
        ent[t0:t1], lf[t0:t1], dom[t0:t1] = e, low, d
    mk = lambda z: pd.DataFrame(z, index=r.index, columns=r.columns)
    return mk(ent), mk(lf), mk(dom)


def _perm_entropy(c, win=20):
    """排列熵：階 3、延遲 1；p_k 為 win 日內各序型出現比率。"""
    a, b, d = c.values[:-2], c.values[1:-1], c.values[2:]
    pat = (a < b).astype(int) * 4 + (b < d).astype(int) * 2 + (a < d).astype(int)  # 0..7 (6 種可能)
    pat = np.vstack([np.full((2, pat.shape[1]), -1), pat])
    P = pd.DataFrame(pat, index=c.index, columns=c.columns)
    ent = 0
    for k in range(8):
        pk = (P == k).astype(float).rolling(win, min_periods=win).mean()
        ent = ent - pk * np.log(pk.where(pk > 0, 1.0))
    return (ent / np.log(6)).clip(0, 1)


def _kalman(logc, sig, q_level=0.05, q_slope=0.002, r_obs=1.0):
    """局部線性趨勢 Kalman 濾波（狀態：水準、斜率），噪音尺度＝個股日波動 σ。向量化跨股票，逐日遞迴（因果）。"""
    T, N = logc.shape
    y = logc.values
    s2 = (sig.values ** 2)
    lvl = np.where(np.isnan(y[0]), 0.0, y[0])
    slp = np.zeros(N)
    P00 = np.ones(N) * 1e-4
    P01 = np.zeros(N)
    P11 = np.ones(N) * 1e-6
    slope_o = np.full((T, N), np.nan)
    dev_o = np.full((T, N), np.nan)
    for t in range(1, T):
        sg = np.where(np.isnan(s2[t - 1]), 1e-4, s2[t - 1])   # 只用 t-1 的波動估計（避免 t 日資訊進噪音尺度）
        # predict
        lp = lvl + slp
        A00 = P00 + 2 * P01 + P11 + q_level * sg
        A01 = P01 + P11
        A11 = P11 + q_slope * sg
        yt = y[t]
        ok = ~np.isnan(yt)
        innov = np.where(ok, yt - lp, 0.0)
        S = A00 + r_obs * sg
        K0, K1 = A00 / S, A01 / S
        lvl_n = lp + K0 * innov
        slp_n = slp + K1 * innov
        P00_n = (1 - K0) * A00
        P01_n = (1 - K0) * A01
        P11_n = A11 - K1 * A01
        lvl = np.where(ok, lvl_n, lp)
        slp = np.where(ok, slp_n, slp)
        P00 = np.where(ok, P00_n, A00)
        P01 = np.where(ok, P01_n, A01)
        P11 = np.where(ok, P11_n, A11)
        sd = np.sqrt(sg)
        slope_o[t] = slp / sd
        dev_o[t] = np.where(ok, innov / np.sqrt(S), np.nan)
    mk = lambda z: pd.DataFrame(z, index=logc.index, columns=logc.columns)
    return mk(slope_o), mk(dev_o)


def sig_feats(adj, r, traded):
    f = {}
    r = r.where(traded, 0.0)
    f["sg_hurst"], f["sg_vr5"] = _hurst_vr(r)
    f["sg_spec_ent"], f["sg_lf_ratio"], f["sg_dom_period"] = _spectral(r)
    f["sg_perm_ent"] = _perm_entropy(adj)
    sig = r.rolling(20, min_periods=10).std().clip(lower=1e-3)
    lc = np.log(adj)
    sl, dev = _kalman(lc, sig)
    f["sg_kal_slope"] = sl.clip(-3, 3)
    f["sg_kal_dev"] = dev.clip(-6, 6)
    # 因果低通（Butterworth 2 階；週期 ~10 日；lfilter，不是 filtfilt，所以不看未來）
    b, a = butter(2, 2.0 / 10.0)
    arr = lc.values
    arr0 = np.where(np.isnan(arr), 0.0, arr)
    zi = lfilter_zi(b, a)[:, None] * arr0[0][None, :]
    lp, _ = lfilter(b, a, arr0, axis=0, zi=zi)
    lp = pd.DataFrame(lp, index=lc.index, columns=lc.columns)
    lp.iloc[:40] = np.nan
    f["sg_lp_dev"] = ((lc - lp) / sig).clip(-8, 8)
    f["sg_lp_slope"] = ((lp - lp.shift(3)) / sig).clip(-8, 8)
    return f


def qr_feats(adj, r, traded, beta, ind, av20, mkt_r1, min_val=5e7):
    f = {}
    m = mkt_r1.reindex(r.index)
    e = r.sub(beta.mul(m, axis=0))
    sec_e = e.T.groupby(ind).transform("mean").T
    e2 = (e - sec_e).where(traded)
    ee = e2.fillna(0)
    sd60 = ee.rolling(60, min_periods=30).std()
    f["qr_resid_z"] = (ee.rolling(20, min_periods=15).sum() / (sd60 * np.sqrt(20)).replace(0, np.nan)).clip(-5, 5)
    f["qr_resid_mom"] = (ee.rolling(60, min_periods=40).sum() - ee.rolling(5).sum()).clip(-2, 2)
    f["qr_resid_vol"] = sd60
    # OU 回歸：Δx_t = b x_{t-1} + ε；x = log 價 − 60日均；b 越負＝均值回歸越強
    lc = np.log(adj)
    x = lc - lc.rolling(60, min_periods=40).mean()
    dx = lc.diff()
    xl = x.shift(1)
    cov = (dx * xl).rolling(60, min_periods=40).mean() - dx.rolling(60, min_periods=40).mean() * xl.rolling(60, min_periods=40).mean()
    var = xl.rolling(60, min_periods=40).var(ddof=0)
    f["qr_ou"] = (cov / var.replace(0, np.nan)).clip(-1, 0.3)
    # 配對：每 21 日用過去 120 日報酬，在同產業流動股中找最相關夥伴（只用重選日以前資料）
    T, N = r.shape
    cols = np.array(r.columns)
    indv = ind.reindex(r.columns).fillna("未知").to_numpy(dtype=object)
    R = r.where(traded).values
    L = lc.values
    pz = np.full((T, N), np.nan)
    pg1 = np.full((T, N), np.nan)
    pg5 = np.full((T, N), np.nan)
    pc = np.full((T, N), np.nan)
    r1 = r.values
    r5 = (adj / adj.shift(5) - 1).values
    same_ind = (indv[:, None] == indv[None, :]) & (indv[:, None] != "未知")
    np.fill_diagonal(same_ind, False)
    for i in range(120, T, 21):
        win = R[i - 120 + 1:i + 1]                      # 含 i 日（i 日收盤後才選，之後的日子才使用）
        cnt = np.isfinite(win).sum(axis=0)
        okc = (cnt >= 100) & (av20.iloc[i] >= min_val).values
        W = np.where(np.isfinite(win), win, 0.0)
        W = W - W.mean(axis=0)
        nrm = np.sqrt((W ** 2).sum(axis=0))
        nrm[nrm == 0] = np.nan
        C = (W.T @ W) / (nrm[:, None] * nrm[None, :])
        C = np.where(same_ind & okc[:, None] & okc[None, :], C, -9)
        C[~np.isfinite(C)] = -9
        part = C.argmax(axis=1)
        best = C[np.arange(N), part]
        has = (best > -9) & okc
        j1 = min(T, i + 22)
        lo = max(0, i - 59)
        spread = L[lo:j1] - L[lo:j1][:, part]           # (rows, N)
        s = pd.DataFrame(spread)
        mu, sd = s.rolling(60, min_periods=40).mean().values, s.rolling(60, min_periods=40).std().values
        z = (spread - mu) / np.where(sd > 0, sd, np.nan)
        # 只寫入 [i+1, j1)，也就是重選之後的日子（i 日本身為重選日，用上一輪夥伴；此處嚴格取 i+1 起，避免用到 i 日收盤選出的夥伴於 i 日）
        a0 = i + 1
        off = a0 - lo
        sl = slice(a0, j1)
        zz = z[off:]
        pz[sl] = np.where(has[None, :], zz, np.nan)
        pg1[sl] = np.where(has[None, :], r1[sl][:, part] - r1[sl], np.nan)
        pg5[sl] = np.where(has[None, :], r5[sl][:, part] - r5[sl], np.nan)
        pc[sl] = np.where(has[None, :], best[None, :], np.nan)
    mk = lambda z: pd.DataFrame(z, index=r.index, columns=r.columns)
    f["qr_pair_z"] = mk(pz).clip(-5, 5)
    f["qr_pair_gap1"] = mk(pg1).clip(-0.2, 0.2)
    f["qr_pair_gap5"] = mk(pg5).clip(-0.5, 0.5)
    f["qr_pair_corr"] = mk(pc)
    return f



def mkt_regime(mkt_r1, tclose):
    """大盤層級 regime 特徵（日期層級，擴張視窗，無未來資訊）。"""
    out = pd.DataFrame(index=mkt_r1.index)
    v20 = mkt_r1.rolling(20).std()
    out["qr_mkt_vol_pct"] = v20.expanding(min_periods=60).apply(lambda a: (a[:-1] < a[-1]).mean(), raw=True)
    out["qr_mkt_dd"] = tclose / tclose.rolling(120, min_periods=40).max() - 1
    h, _ = _hurst_vr(mkt_r1.to_frame("m"))
    out["qr_mkt_hurst"] = h["m"]
    return out
