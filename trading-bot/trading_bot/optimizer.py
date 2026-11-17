"""Parameter sensitivity (NOT selection): measures how fragile a rule set is.

A robust edge should degrade smoothly around the chosen parameters. A result that
only exists at one point of the grid is treated as noise (multiple-testing risk).
Run on development data only (never on the final out-of-sample segment).
"""
from __future__ import annotations

import itertools

import pandas as pd

from . import performance as P
from .backtester import run_backtest


def sensitivity(strategy_cls, base: dict, grid: dict, data, spec, risk, cost, start, end, rf=None) -> pd.DataFrame:
    rows = []
    keys = list(grid)
    for vals in itertools.product(*grid.values()):
        params = dict(base, **dict(zip(keys, vals)))
        strat = strategy_cls(params)
        res = run_backtest(data, strat.signals(data), spec, risk, cost, start, end)
        m = P.compute(res.equity, res.trades, rf)
        rows.append({**dict(zip(keys, vals)), "trades": m.get("Total trades", 0),
                     "return": m["Total return"], "sharpe": m["Sharpe (excess)"],
                     "profit_factor": m.get("Profit factor"), "expectancy_R": m.get("Expectancy (R)"),
                     "max_dd": m["Max drawdown"]})
    return pd.DataFrame(rows)
