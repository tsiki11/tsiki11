"""Gold (GLD) long/short bot: the same three strategy families, buy AND sell.

1. TSMOM  : signal = mean(sign(ret 1,3,6,12m)) in [-1,1], vol-targeted, monthly.
2. GOLDEN : +1 if SMA50 > SMA200 else -1 (full position, evaluated daily).
3. IBS    : long  if IBS<0.2 and close>SMA200 -> exit close>prior high
            short if IBS>0.8 and close<SMA200 -> exit close<prior low
            max hold 10 days, filled at next open.
4. COMBO  : inverse-vol mix of 1-3, portfolio vol target 10%, no leverage.

GOLD_CONFIG was fixed BEFORE running any test (textbook parameters, same as
the equity experiments). Do not change it after looking at 2023+ results.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import engine, strategies as S, portfolio as P

GOLD_CONFIG = {
    "asset": "GLD",
    "tsmom_target_vol": 0.10, "tsmom_freq": "M", "max_gross": 1.0,
    "gc_fast": 50, "gc_slow": 200,
    "ibs_long": 0.2, "ibs_short": 0.8, "ibs_sma": 200, "ibs_max_hold": 10,
    "cost_bps": 5, "short_fee_yr": 0.005, "short_rebate": False,
    "alloc_vol_window": 252, "alloc_freq": "M",
    "portfolio_target_vol": 0.10, "max_leverage": 1.0,
}


def tsmom_ls(close: pd.Series, target_vol=0.10, max_gross=1.0, freq="M") -> pd.Series:
    sig = sum(np.sign(close / close.shift(lb) - 1) for lb in S.LOOKBACKS) / len(S.LOOKBACKS)
    sig = sig.where(close.shift(max(S.LOOKBACKS)).notna())
    vol = close.pct_change(fill_method=None).ewm(halflife=21, min_periods=60).std() * np.sqrt(252)
    w = (sig * target_vol / vol).clip(-max_gross, max_gross)
    mask = S.rebalance_mask(close.index, freq)
    return w[mask].reindex(close.index).ffill().fillna(0.0)


def golden_cross_ls(close: pd.Series, fast=50, slow=200) -> pd.Series:
    f, s = close.rolling(fast).mean(), close.rolling(slow).mean()
    return pd.Series(np.where(f > s, 1.0, -1.0), index=close.index).where(s.notna()).fillna(0.0)


def ibs_ls(o, h, l, c, long_th=0.2, short_th=0.8, sma=200, max_hold=10) -> pd.Series:
    ibs = ((c - l) / (h - l).replace(0, np.nan)).values
    ma = c.rolling(sma).mean().values
    cv, ph, pl = c.values, h.shift(1).values, l.shift(1).values
    pos = np.zeros(len(c)); state, held = 0, 0
    for i in range(len(c)):
        if np.isnan(cv[i]) or np.isnan(ma[i]):
            continue
        if state == 1:
            held += 1
            if cv[i] > ph[i] or held >= max_hold: state = 0
        elif state == -1:
            held += 1
            if cv[i] < pl[i] or held >= max_hold: state = 0
        if state == 0:
            if ibs[i] < long_th and cv[i] > ma[i]: state, held = 1, 0
            elif ibs[i] > short_th and cv[i] < ma[i]: state, held = -1, 0
        pos[i] = state
    return pd.Series(pos, index=c.index)


def run_gold(d: dict, end: str | None = None, cfg=GOLD_CONFIG):
    o, h, l, c = P.prepare(d, end)
    a = cfg["asset"]
    rcc, roo = c.pct_change(fill_method=None), o.pct_change(fill_method=None)
    cash = rcc["BIL"].fillna(0.0)
    kw = dict(cost_bps=cfg["cost_bps"], short_fee_yr=cfg["short_fee_yr"], short_rebate=cfg["short_rebate"])
    W = {
        "TSMOM": (tsmom_ls(c[a], cfg["tsmom_target_vol"], cfg["max_gross"], cfg["tsmom_freq"]), rcc),
        "GoldenCross": (golden_cross_ls(c[a], cfg["gc_fast"], cfg["gc_slow"]), rcc),
        "IBS": (ibs_ls(o[a], h[a], l[a], c[a], cfg["ibs_long"], cfg["ibs_short"], cfg["ibs_sma"], cfg["ibs_max_hold"]), roo),
    }
    res, weights = {}, {}
    for k, (w, r) in W.items():
        res[k] = engine.run(w.to_frame(a), r[[a]], cash, lag=1, **kw)
        weights[k] = w
    sleeves = pd.DataFrame({k: v["ret"] for k, v in res.items()})
    acfg = dict(P.CONFIG, alloc_vol_window=cfg["alloc_vol_window"], alloc_freq=cfg["alloc_freq"],
                portfolio_target_vol=cfg["portfolio_target_vol"], max_leverage=cfg["max_leverage"])
    alloc = P.allocation(sleeves, cash, acfg)
    combo = engine.run(alloc, sleeves.fillna(0.0), cash, lag=1, cost_bps=cfg["cost_bps"])
    res["Combo"] = combo
    return res, weights, alloc, cash, rcc
