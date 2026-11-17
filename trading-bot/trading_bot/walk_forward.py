"""Walk-forward tools.

rolling_windows   : performance of a FIXED rule set over consecutive windows
                    (stability check; used for un-optimised baselines).
walk_forward_optimize : anchored/rolling re-optimisation (train -> test), used in Phase 8.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from . import performance as P
from .backtester import run_backtest


def rolling_windows(trades: pd.DataFrame, equity: pd.Series, freq: str = "6MS") -> pd.DataFrame:
    rows = []
    e = equity.copy()
    starts = pd.date_range(e.index[0].normalize(), e.index[-1], freq=freq, tz="UTC")
    if len(starts) == 0 or starts[0] > e.index[0]:
        starts = starts.insert(0, e.index[0].normalize())
    for a, b in zip(starts, list(starts[1:]) + [e.index[-1] + pd.Timedelta(seconds=1)]):
        seg = e[(e.index >= a) & (e.index < b)]
        if len(seg) < 50:
            continue
        tr = trades[(trades["entry_time"] >= a) & (trades["entry_time"] < b)] if len(trades) else trades
        ret = seg.iloc[-1] / seg.iloc[0] - 1
        rows.append({"window_start": a.date(), "window_end": (b - pd.Timedelta(seconds=1)).date(),
                     "return": ret, "trades": len(tr),
                     "win_rate": (tr["net_pnl"] > 0).mean() if len(tr) else np.nan,
                     "expectancy_R": tr["R"].mean() if len(tr) else np.nan,
                     "max_dd": float((seg / seg.cummax() - 1).min())})
    return pd.DataFrame(rows)


def walk_forward_optimize(strategy_cls, grid: dict, data, spec, risk, cost, start, end,
                          train_months=24, test_months=6, objective="Expectancy (R)", min_trades=30):
    """Re-optimise on each training window, apply the best params to the following test window.
    Returns the stitched out-of-sample test trades plus the chosen params per window."""
    keys = list(grid)
    combos = [dict(zip(keys, v)) for v in itertools.product(*grid.values())]
    cache = {i: strategy_cls(c).signals(data) for i, c in enumerate(combos)}
    t0 = pd.Timestamp(start, tz="UTC")
    end = pd.Timestamp(end, tz="UTC")
    rows, test_trades = [], []
    while True:
        tr_end = t0 + pd.DateOffset(months=train_months)
        te_end = tr_end + pd.DateOffset(months=test_months)
        if tr_end >= end:
            break
        best, best_val = None, -np.inf
        for i, c in enumerate(combos):
            res = run_backtest(data, cache[i], spec, risk, cost, t0, tr_end - pd.Timedelta(seconds=1))
            if len(res.trades) < min_trades:
                continue
            val = P.compute(res.equity, res.trades)[objective]
            if val > best_val:
                best, best_val = i, val
        if best is not None:
            res = run_backtest(data, cache[best], spec, risk, cost, tr_end, min(te_end, end) - pd.Timedelta(seconds=1))
            m = P.compute(res.equity, res.trades)
            rows.append({"train_start": t0.date(), "test_start": tr_end.date(), "test_end": min(te_end, end).date(),
                         "params": combos[best], "train_obj": best_val, "test_return": m["Total return"],
                         "test_trades": m["Total trades"]})
            test_trades.append(res.trades)
        t0 = t0 + pd.DateOffset(months=test_months)
    return pd.DataFrame(rows), (pd.concat(test_trades) if test_trades else pd.DataFrame())
