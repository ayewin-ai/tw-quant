"""gs-quant 風格的風險／統計／技術分析指標（離線、只用台股公開價量資料，不依賴 gs-quant 套件與 GS 帳號）。

概念出處：Goldman Sachs gs-quant (Apache-2.0) 的 gs_quant/timeseries —
  econometrics.volatility / exponential_std(EWMA 波動) / max_drawdown / beta / correlation，
  technicals.bollinger_bands / macd / relative_strength_index，statistics.zscores / winsorize(±2.5σ)，
  以及 gs_quant/models 的「因子風險模型：總風險＝因子(市場、產業)風險＋特定風險」概念。
本檔依公式自行以 pandas 重寫（沒有複製原始碼），全部只用 t 日（含）以前的資料。
"""
import numpy as np
import pandas as pd

INFO = {
    "ewvol94": ("風險-波動", "EWMA 波動(λ=0.94)"),
    "ewvol_ratio": ("風險-波動", "EWMA 波動/60日波動"),
    "var95_60": ("風險-尾端", "60日歷史 VaR95"),
    "cvar95_60": ("風險-尾端", "60日 CVaR95(預期短缺)"),
    "mdd60": ("風險-回撤", "60日最大回撤"),
    "downdev60": ("風險-尾端", "60日下行標準差"),
    "beta120": ("風險-因子", "120日市場 beta"),
    "r2_mkt60": ("風險-因子", "60日市場模型 R²（系統性占比）"),
    "sec_beta60": ("風險-因子", "60日產業 beta"),
    "bb_pct20": ("技術-布林", "布林通道 %b(20,2)"),
    "macd_hist": ("技術-MACD", "MACD 柱狀(12,26,9)/價"),
    "skew60": ("風險-尾端", "60日報酬偏態"),
    "z20w": ("統計-zscore", "20日報酬 z 分數(±2.5 截尾)"),
}
RISK_F = list(INFO)


def risk_feats(adj, r, mk, ind):
    """adj: 還原股價指數 (days x codes); r: 日報酬; mk: 大盤日報酬 (Series by day); ind: 產業 (Series by code)."""
    f = {}
    # EWMA volatility (RiskMetrics λ=0.94), gs-quant 'exponential_std' idea
    f["ewvol94"] = np.sqrt((r ** 2).ewm(alpha=0.06, adjust=False, min_periods=20).mean())
    f["ewvol_ratio"] = f["ewvol94"] / r.rolling(60, min_periods=30).std()
    # historical VaR / CVaR (5% tail) over 60 days
    q = r.rolling(60, min_periods=40).quantile(0.05)
    f["var95_60"] = q
    # CVaR ≈ mean of the worst 3 of 60 (5%)
    f["cvar95_60"] = r.rolling(60, min_periods=40).apply(lambda a: np.sort(a)[:max(1, int(len(a) * 0.05))].mean(), raw=True)
    # max drawdown over 60d (gs-quant max_drawdown: min over window of price/rolling max - 1)
    peak = adj.rolling(60, min_periods=30).max()
    f["mdd60"] = (adj / peak - 1).rolling(60, min_periods=30).min()
    f["downdev60"] = np.sqrt((r.clip(upper=0) ** 2).rolling(60, min_periods=30).mean())
    # factor exposures: market beta (120d), R² of market model (60d), sector beta
    m = mk.reindex(r.index)

    def beta_r2(win, mm):
        mm_ = mm if isinstance(mm, pd.DataFrame) else None
        if mm_ is None:
            cov = r.mul(mm, axis=0).rolling(win, min_periods=win // 2).mean() - \
                  r.rolling(win, min_periods=win // 2).mean().mul(mm.rolling(win, min_periods=win // 2).mean(), axis=0)
            var = mm.rolling(win, min_periods=win // 2).var(ddof=0)
            b = cov.div(var, axis=0)
            corr = cov.div(np.sqrt(var), axis=0).div(r.rolling(win, min_periods=win // 2).std(ddof=0))
        else:
            cov = (r * mm_).rolling(win, min_periods=win // 2).mean() - \
                  r.rolling(win, min_periods=win // 2).mean() * mm_.rolling(win, min_periods=win // 2).mean()
            var = mm_.rolling(win, min_periods=win // 2).var(ddof=0)
            b = cov / var
            corr = cov / np.sqrt(var) / r.rolling(win, min_periods=win // 2).std(ddof=0)
        return b, corr ** 2
    b120, _ = beta_r2(120, m)
    f["beta120"] = b120.clip(-1, 4)
    _, r2 = beta_r2(60, m)
    f["r2_mkt60"] = r2.clip(0, 1)
    sec = r.T.groupby(ind).transform("mean").T
    sb, _ = beta_r2(60, sec)
    f["sec_beta60"] = sb.clip(-1, 4)
    # Bollinger %b, MACD histogram
    ma20, sd20 = adj.rolling(20).mean(), adj.rolling(20).std()
    f["bb_pct20"] = ((adj - (ma20 - 2 * sd20)) / (4 * sd20)).clip(-1, 2)
    macd = adj.ewm(span=12, adjust=False).mean() - adj.ewm(span=26, adjust=False).mean()
    f["macd_hist"] = ((macd - macd.ewm(span=9, adjust=False).mean()) / adj).clip(-0.2, 0.2)
    f["skew60"] = r.rolling(60, min_periods=40).skew().clip(-5, 5)
    # winsorised z-score of the 20d return (gs-quant zscores + winsorize limit 2.5)
    r20 = adj / adj.shift(20) - 1
    z = (r20 - r20.rolling(120, min_periods=60).mean()) / r20.rolling(120, min_periods=60).std()
    f["z20w"] = z.clip(-2.5, 2.5)
    return f
