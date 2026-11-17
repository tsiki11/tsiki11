"""Minimal, transparent vectorised backtest engine driven by target weights.

Conventions (no look-ahead):
  * `weights.loc[t]` is the target portfolio decided with information
    available at the CLOSE of day t.
  * The target in force at the end of bar t is weights[t - lag]; it then
    earns the returns of bar t+1 onwards (holdings drift between trades).
    With close-to-close returns and lag=1 the order fills at the close of
    t+1 (one full day of delay -> conservative).
  * Uninvested capital (1 - gross) earns the cash return (BIL ETF).
  * Costs: `cost_bps` per unit of one-way turnover (spread + slippage +
    commission), charged on the day the trade happens.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def run(weights: pd.DataFrame, returns: pd.DataFrame, cash_ret: pd.Series,
        lag: int = 1, cost_bps: float | pd.Series = 5.0,
        short_fee_yr: float = 0.0, short_rebate: bool = True) -> pd.DataFrame:
    """Event-driven simulation with weight drift between rebalances.

    A trade happens only when the (lagged) target changes; in between,
    holdings drift with prices (no free daily rebalancing).
    Shorts: pay `short_fee_yr` (borrow fee, e.g. 0.005) on short notional; if
    short_rebate=False the short sale proceeds earn no interest (conservative).
    """
    weights = weights.reindex(returns.index).ffill().fillna(0.0)
    tgt = weights.shift(lag).fillna(0.0)           # target in force at close of day t
    cols = list(tgt.columns)
    R = returns[cols].fillna(0.0).values
    T = tgt.values
    rc = cash_ret.reindex(returns.index).fillna(0.0).values
    if isinstance(cost_bps, pd.Series):
        cb = cost_bps.reindex(cols).fillna(5.0).values / 1e4
    else:
        cb = np.full(len(cols), cost_bps / 1e4)
    borrow_spread = 0.01 / 252                     # +1% p.a. on borrowed cash
    n = len(returns)
    h = np.zeros(len(cols))                        # current asset weights
    out_ret, out_gross, out_to, out_cost = (np.zeros(n) for _ in range(4))
    for t in range(n):
        cash_w = 1.0 - h.sum()
        cr = rc[t] if cash_w >= 0 else rc[t] + borrow_spread
        port = float(h @ R[t] + cash_w * cr)
        short_notional = float(-h[h < 0].sum())
        if short_notional > 0:
            port -= short_notional * short_fee_yr / 252
            if not short_rebate:
                port -= short_notional * rc[t]
        # drift
        if 1.0 + port != 0:
            h = h * (1.0 + R[t]) / (1.0 + port)
        # rebalance at today's close if the target changed
        cost = 0.0
        if t == 0 or np.any(np.abs(T[t] - T[t - 1]) > 1e-12):
            trade = np.abs(T[t] - h)
            cost = float(trade @ cb)
            out_to[t] = trade.sum()
            h = T[t].copy()
        out_ret[t] = (1.0 + port) * (1.0 - cost) - 1.0
        out_cost[t] = cost
        out_gross[t] = np.abs(h).sum()
    idx = returns.index
    return pd.DataFrame({"ret": out_ret, "gross": out_gross,
                         "turnover": out_to, "costs": out_cost}, index=idx)
