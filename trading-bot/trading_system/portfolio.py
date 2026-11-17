"""Build the sleeves and combine them into one portfolio.

LOCKED CONFIGURATION: chosen using 2013-2022 data only. Do not change after
looking at out-of-sample (2023+) results.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import engine, strategies as S

CONFIG = {
    "trend_assets": ["SPY", "QQQ", "IWM", "EFA", "EEM", "VNQ",
                     "TLT", "IEF", "LQD", "TIP", "GLD", "DBC", "UUP"],
    "trend_target_vol": 0.10, "trend_freq": "M",
    "mr_assets": ["SPY", "QQQ", "IWM"], "mr_entry": 0.2, "mr_sma": 200, "mr_max_hold": 10,
    "crypto_assets": ["BTC-USD", "ETH-USD"], "crypto_target_vol": 0.30, "crypto_freq": "W-FRI",
    "etf_cost_bps": 5, "crypto_cost_bps": 15,
    # sleeve allocation
    "alloc_vol_window": 252, "alloc_freq": "M",
    "portfolio_target_vol": 0.10, "max_leverage": 1.0,
}


def prepare(d: dict, end: str | None = None):
    days = d["Close"]["SPY"].dropna().index
    o, h, l, c = (d[k].loc[days] for k in ("Open", "High", "Low", "Close"))
    if end is not None:
        o, h, l, c = (x.loc[:end] for x in (o, h, l, c))
    return o, h, l, c


def sleeve_returns(d: dict, end: str | None = None, cfg=CONFIG) -> tuple[pd.DataFrame, dict]:
    o, h, l, c = prepare(d, end)
    rcc = c.pct_change(fill_method=None)
    roo = o.pct_change(fill_method=None)
    cash = rcc["BIL"].fillna(0.0)

    wA = S.multi_asset_trend(c, cfg["trend_assets"], cfg["trend_target_vol"], freq=cfg["trend_freq"])
    bA = engine.run(wA, rcc, cash, lag=1, cost_bps=cfg["etf_cost_bps"])

    wB = S.ibs_mean_reversion(o, h, l, c, cfg["mr_assets"], cfg["mr_entry"], cfg["mr_sma"], cfg["mr_max_hold"])
    # roo[t] = open(t-1)->open(t); with lag=1 the order decided at close t-1
    # is filled at open t (market-on-open order) -> no look-ahead
    bB = engine.run(wB, roo, cash, lag=1, cost_bps=cfg["etf_cost_bps"])

    cr = cfg["crypto_assets"]
    wC = S.crypto_trend(c[cr], cr, cfg["crypto_target_vol"], freq=cfg["crypto_freq"])
    bC = engine.run(wC, rcc[cr], cash, lag=1, cost_bps=cfg["crypto_cost_bps"])
    # before crypto history exists the sleeve is just cash -> mark as NaN
    first_c = wC.abs().sum(axis=1).gt(0).idxmax()
    rC = bC["ret"].where(bC.index >= first_c)

    sleeves = pd.DataFrame({"Trend": bA["ret"], "MeanRev": bB["ret"], "Crypto": rC})
    details = {"Trend": (wA, bA), "MeanRev": (wB, bB), "Crypto": (wC, bC), "cash": cash}
    return sleeves, details


def allocation(sleeves: pd.DataFrame, cash: pd.Series, cfg=CONFIG) -> pd.DataFrame:
    """Inverse-vol weights across sleeves, scaled to a portfolio vol target."""
    ex = sleeves.sub(cash, axis=0)
    vol = ex.rolling(cfg["alloc_vol_window"], min_periods=126).std() * np.sqrt(252)
    inv = 1.0 / vol
    w = inv.div(inv.sum(axis=1), axis=0)
    mask = S.rebalance_mask(sleeves.index, cfg["alloc_freq"])
    w = w[mask].dropna(how="all")
    # scale to target vol using trailing sleeve covariance
    k = pd.Series(index=w.index, dtype=float)
    for t in w.index:
        hist = ex.loc[:t].iloc[-cfg["alloc_vol_window"]:]
        wt = w.loc[t].fillna(0.0)
        cols = wt[wt > 0].index
        cov = hist[cols].dropna().cov().values * 252
        pv = float(np.sqrt(wt[cols].values @ cov @ wt[cols].values))
        k[t] = min(cfg["portfolio_target_vol"] / pv, cfg["max_leverage"]) if pv > 0 else 0
    w = w.fillna(0.0).mul(k, axis=0)
    return w.reindex(sleeves.index).ffill().fillna(0.0)


def run_portfolio(d: dict, end: str | None = None, cfg=CONFIG):
    sleeves, det = sleeve_returns(d, end, cfg)
    cash = det["cash"]
    alloc = allocation(sleeves, cash, cfg)
    # sleeve re-allocation trades cost ~ETF cost; executed the next day
    port = engine.run(alloc, sleeves.fillna(0.0), cash, lag=1, cost_bps=cfg["etf_cost_bps"])
    return port, sleeves, alloc, det
