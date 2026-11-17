"""Strategy sleeves. Every function returns TARGET WEIGHTS decided at the close.

Sleeves are taken from published, widely replicated research. Parameters are
the "textbook" ones and were fixed BEFORE looking at results:

A. Multi-asset trend following (time-series momentum)
   Moskowitz, Ooi & Pedersen (2012) "Time Series Momentum";
   Hurst, Ooi & Pedersen (2017) "A Century of Evidence on Trend-Following";
   Faber (2007) "A Quantitative Approach to Tactical Asset Allocation".
   Signal = average of sign(return) over 1, 3, 6, 12 months (long-only),
   inverse-volatility sizing, portfolio volatility target.

B. Short-term equity mean reversion (Internal Bar Strength)
   Connors & Alvarez (2009) "Short Term Trading Strategies That Work";
   Pagonidis (2014) "The IBS Effect: Mean Reversion in Equity ETFs".
   Buy when IBS < 0.2 and price > 200d SMA, exit when close > prior high.

C. Crypto trend following (BTC, ETH)
   Liu & Tsyvinski (2021) "Risks and Returns of Cryptocurrency" (time-series
   momentum in crypto). Same trend signal as A, volatility targeted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LOOKBACKS = (21, 63, 126, 252)  # 1, 3, 6, 12 months


def trend_score(close: pd.DataFrame, lookbacks=LOOKBACKS) -> pd.DataFrame:
    """Fraction of lookbacks with positive return -> score in [0, 1]."""
    s = sum((close / close.shift(lb) - 1 > 0).astype(float) for lb in lookbacks)
    s = s / len(lookbacks)
    return s.where(close.shift(max(lookbacks)).notna())


def golden_cross_score(close: pd.DataFrame, fast: int = 50, slow: int = 200) -> pd.DataFrame:
    """1 when the fast SMA is above the slow SMA (after a golden cross), else 0."""
    f, sl = close.rolling(fast).mean(), close.rolling(slow).mean()
    return (f > sl).astype(float).where(sl.notna())


def ewm_vol(rets: pd.DataFrame, halflife: int = 21, ann: int = 252) -> pd.DataFrame:
    return rets.ewm(halflife=halflife, min_periods=60).std() * np.sqrt(ann)


def rebalance_mask(index: pd.DatetimeIndex, freq: str = "W-FRI") -> pd.Series:
    """True on the last trading day of each period."""
    s = pd.Series(index, index=index)
    last = s.groupby(s.dt.to_period(freq)).transform("max")
    return pd.Series(index == last.values, index=index)


def vol_target_scale(weights: pd.DataFrame, rets: pd.DataFrame, target_vol: float,
                     cov_window: int = 126, max_gross: float = 1.0) -> pd.Series:
    """Scalar per date so that ex-ante portfolio vol ~= target_vol."""
    scale = pd.Series(np.nan, index=weights.index)
    r = rets[weights.columns]
    for t in weights.index:
        w = weights.loc[t].fillna(0.0)
        if w.abs().sum() == 0:
            scale.loc[t] = 0.0
            continue
        hist = r.loc[:t].iloc[-cov_window:]
        cols = w[w != 0].index
        h = hist[cols].dropna()
        if len(h) < 60:
            continue
        cov = h.cov().values * 252
        pv = float(np.sqrt(w[cols].values @ cov @ w[cols].values))
        k = target_vol / pv if pv > 0 else 0.0
        k = min(k, max_gross / w.abs().sum())
        scale.loc[t] = k
    return scale


# ---------------------------------------------------------------- Sleeve A
def multi_asset_trend(close: pd.DataFrame, assets: list[str], target_vol=0.10,
                      max_gross=1.0, freq="W-FRI", signal="tsmom",
                      fast=50, slow=200) -> pd.DataFrame:
    """signal="tsmom": 1/3/6/12m momentum ensemble (periodic rebalance).
    signal="golden_cross": SMA(fast) > SMA(slow); rebalanced at period end AND
    on any day a cross happens, so entries/exits are not delayed."""
    px = close[assets]
    rets = px.pct_change(fill_method=None)
    if signal == "golden_cross":
        score = golden_cross_score(px, fast, slow)
    else:
        score = trend_score(px)
    vol = ewm_vol(rets)
    raw = score / vol                               # inverse-vol risk budgeting
    raw = raw.div(raw.notna().sum(axis=1), axis=0)  # spread across available assets
    mask = rebalance_mask(px.index, freq)
    if signal == "golden_cross":
        flips = score.fillna(0).diff().abs().sum(axis=1) > 0
        mask = mask | flips
    raw = raw[mask].dropna(how="all")
    k = vol_target_scale(raw.fillna(0), rets, target_vol, max_gross=max_gross)
    w = raw.fillna(0).mul(k, axis=0).dropna(how="all")
    return w.reindex(px.index).ffill().fillna(0.0)


# ---------------------------------------------------------------- Sleeve B
def ibs_mean_reversion(o, h, l, c, assets: list[str], entry=0.2, sma=200,
                       max_hold=10) -> pd.DataFrame:
    """Long-only, equal capital per ETF. Returns 0/1 position per asset
    (to be scaled by caller). State machine evaluated at each close."""
    out = {}
    for a in assets:
        hi, lo, cl = h[a], l[a], c[a]
        rng = (hi - lo).replace(0, np.nan)
        ibs = (cl - lo) / rng
        trend_ok = cl > cl.rolling(sma).mean() if sma else pd.Series(True, index=cl.index)
        pos = np.zeros(len(cl))
        in_pos, held = False, 0
        prev_high = hi.shift(1).values
        ibs_v, tr_v, cl_v = ibs.values, trend_ok.values, cl.values
        for i in range(len(cl)):
            if np.isnan(cl_v[i]):
                continue
            if in_pos:
                held += 1
                if cl_v[i] > prev_high[i] or held >= max_hold:
                    in_pos = False
            else:
                if ibs_v[i] < entry and tr_v[i]:
                    in_pos, held = True, 0
            pos[i] = 1.0 if in_pos else 0.0
        out[a] = pd.Series(pos, index=cl.index)
    return pd.DataFrame(out) / len(assets)


# ---------------------------------------------------------------- Sleeve C
def crypto_trend(close: pd.DataFrame, assets: list[str], target_vol=0.30,
                 max_gross=1.0, freq="W-FRI", signal="tsmom", fast=50, slow=200) -> pd.DataFrame:
    return multi_asset_trend(close, assets, target_vol=target_vol, max_gross=max_gross,
                             freq=freq, signal=signal, fast=fast, slow=slow)
