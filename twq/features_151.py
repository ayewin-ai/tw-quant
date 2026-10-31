"""Kakushadze & Serur,《151 Trading Strategies》(2018; SSRN 3247865; 西文版 arXiv:1912.04492) 第 3 章（股票）
與第 4 章（ETF/產業輪動）中，只需台股日資料即可實作的策略，依論文公式自行改寫為 point-in-time 特徵（無未來資訊）。

群組：
  mom  3.1 價格動能（累積/平均/風險調整報酬，跳過最近 1 個月）、3.7 殘差動能（扣市場 beta 後的殘差累積報酬）
  rev  3.9/3.10 均值回歸：產業去均值的 1 日 / 5 日報酬（群內反轉），以及以 20 日波動標準化的版本
  ma   3.11-3.13 均線：價/均線、雙均線差、三均線排列訊號
  sr   3.14 支撐壓力（前一日 pivot C=(H+L+C)/3、R=2C-L、S=2C-H）與 3.15 通道（20 日 Donchian 位置）
  rot  4.1 產業輪動：產業 6-1 月動能的橫斷面排名
  lowvol 3.4 低波動（6 個月日報酬標準差）
  mf   3.6 多因子：動能、低波動、反轉、通道之橫斷面排名平均
跳過：3.2/3.3 盈餘動能與價值已在 features_guru（eps_yoy/ep/bp）；3.5 隱含波動、3.19 造市需選擇權或逐筆資料；
  3.8 配對交易以 3.9 產業群組版取代；3.17 KNN 與 3.20 Alpha combo 由 LightGBM 本身擔任；3.16 併購事件無資料。
"""
import numpy as np
import pandas as pd

INFO = {
    "k_mom_cum": ("151-動能", "3.1 累積報酬(12-1月)"), "k_mom_ra": ("151-動能", "3.1 風險調整動能"),
    "k_resmom": ("151-動能", "3.7 殘差動能(12-1月)"), "k_resmom_ra": ("151-動能", "3.7 殘差動能/殘差波動"),
    "k_rev1_ind": ("151-均值回歸", "3.9 產業去均值1日報酬"), "k_rev5_ind": ("151-均值回歸", "3.9 產業去均值5日報酬"),
    "k_rev5_z": ("151-均值回歸", "3.10 標準化產業去均值5日報酬"),
    "k_ma20": ("151-均線", "3.11 價/20日均線"), "k_ma_x": ("151-均線", "3.12 10/50日均線差"),
    "k_ma3": ("151-均線", "3.13 三均線排列(5/20/60)"),
    "k_piv": ("151-支撐壓力", "3.14 收盤相對 pivot"), "k_piv_rs": ("151-支撐壓力", "3.14 R/S 位置"),
    "k_chan": ("151-通道", "3.15 20日通道位置"), "k_rot": ("151-產業輪動", "4.1 產業動能排名"),
    "k_lowvol": ("151-低波動", "3.4 6月日波動"), "k_mf": ("151-多因子", "3.6 多因子排名平均"),
}
GROUPS = {
    "mom": ["k_mom_cum", "k_mom_ra", "k_resmom", "k_resmom_ra"],
    "rev": ["k_rev1_ind", "k_rev5_ind", "k_rev5_z"],
    "ma": ["k_ma20", "k_ma_x", "k_ma3"],
    "sr": ["k_piv", "k_piv_rs", "k_chan"],
    "rot": ["k_rot", "k_lowvol", "k_mf"],
}
ALL_F = [c for g in GROUPS.values() for c in g]


def _ind_demean(x, ind):
    return x - x.T.groupby(ind).transform("mean").T


def feats_151(adj, close, high, low, r, mkt, ind):
    f = {}
    S, T = 21, 231                      # 跳過最近 1 個月，回看 11 個月
    rs = r.shift(S)
    f["k_mom_cum"] = adj.shift(S) / adj.shift(S + T) - 1
    mu, sd = rs.rolling(T, min_periods=120).mean(), rs.rolling(T, min_periods=120).std()
    f["k_mom_ra"] = mu / sd.replace(0, np.nan)
    m = mkt.reindex(r.index)
    cov = (r.mul(m, axis=0).rolling(120, min_periods=60).mean()
           - r.rolling(120, min_periods=60).mean().mul(m.rolling(120, min_periods=60).mean(), axis=0))
    beta = cov.div(m.rolling(120, min_periods=60).var(ddof=0), axis=0)
    eps = (r - beta.shift(1).mul(m, axis=0)).shift(S)   # beta 只用前一日以前的資料
    f["k_resmom"] = eps.rolling(T, min_periods=120).sum()
    f["k_resmom_ra"] = eps.rolling(T, min_periods=120).mean() / eps.rolling(T, min_periods=120).std().replace(0, np.nan)
    r5 = adj / adj.shift(5) - 1
    f["k_rev1_ind"] = _ind_demean(r, ind)
    f["k_rev5_ind"] = _ind_demean(r5, ind)
    f["k_rev5_z"] = f["k_rev5_ind"] / (r.rolling(20, min_periods=10).std() * np.sqrt(5)).replace(0, np.nan)
    ma = {n: adj.rolling(n, min_periods=n).mean() for n in (5, 10, 20, 50, 60)}
    f["k_ma20"] = adj / ma[20] - 1
    f["k_ma_x"] = ma[10] / ma[50] - 1
    f["k_ma3"] = ((ma[5] > ma[20]) & (ma[20] > ma[60])).astype(float) - ((ma[5] < ma[20]) & (ma[20] < ma[60])).astype(float)
    f["k_ma3"] = f["k_ma3"].where(ma[60].notna())
    C = ((high + low + close) / 3).shift(1)
    R, Sp = 2 * C - low.shift(1), 2 * C - high.shift(1)
    f["k_piv"] = close / C - 1
    f["k_piv_rs"] = (close - Sp) / (R - Sp).replace(0, np.nan)
    hh, ll = high.rolling(20, min_periods=20).max(), low.rolling(20, min_periods=20).min()
    f["k_chan"] = (close - ll) / (hh - ll).replace(0, np.nan)
    im = (adj.shift(S) / adj.shift(S + 105) - 1).T.groupby(ind).transform("mean").T
    f["k_rot"] = im.rank(axis=1, pct=True)
    f["k_lowvol"] = r.rolling(126, min_periods=60).std()
    rk = lambda x: x.rank(axis=1, pct=True)  # noqa: E731
    f["k_mf"] = (rk(f["k_mom_cum"]) + rk(-f["k_lowvol"]) + rk(-f["k_rev5_ind"]) + rk(f["k_chan"])) / 4
    return f
