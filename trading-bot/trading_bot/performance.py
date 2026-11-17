"""Performance metrics. All results are NET of spread, commission, slippage and swap.

Definitions
  Daily equity     : last mark-to-market equity of each UTC day with market data.
  r_d              : daily simple return of equity.
  rf_d             : daily T-bill return (BIL ETF) -> opportunity cost of the margin capital.
  Sharpe           : mean(r_d - rf_d) / std(r_d - rf_d) * sqrt(N), N = trading days per year (observed)
  Sortino          : mean(r_d - rf_d) * N / (std of negative excess returns * sqrt(N))
  Max drawdown     : min over time of equity / running max - 1 (bar-level equity)
  CAGR             : (E_end / E_start)^(1/years) - 1
  Calmar           : CAGR / |MaxDD|
  VaR/CVaR 95%     : historical, on daily returns (loss reported as a positive number)
  Profit factor    : sum(net wins) / |sum(net losses)|
  Expectancy       : mean net P&L per trade ($) and mean R-multiple (R = net P&L / planned risk)
  Payoff ratio     : average win / |average loss|
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def daily_returns(equity: pd.Series) -> pd.Series:
    e = equity.copy()
    e.index = e.index.tz_convert(None)
    d = e.resample("1D").last().dropna()
    return d.pct_change().dropna()


def max_consecutive(x: pd.Series) -> int:
    best = cur = 0
    for v in x:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def compute(equity: pd.Series, trades: pd.DataFrame, rf_daily: pd.Series | None = None) -> dict:
    out = {}
    e0, e1 = float(equity.iloc[0]), float(equity.iloc[-1])
    years = (equity.index[-1] - equity.index[0]).total_seconds() / (365.25 * 86400)
    r = daily_returns(equity)
    n_per_year = len(r) / years if years > 0 else 252
    rf = (rf_daily.reindex(r.index).fillna(0.0) if rf_daily is not None else pd.Series(0.0, index=r.index))
    ex = r - rf
    dd = equity / equity.cummax() - 1
    out.update({
        "Start": str(equity.index[0].date()), "End": str(equity.index[-1].date()), "Years": round(years, 2),
        "Initial equity": e0, "Final equity": e1, "Net profit": e1 - e0,
        "Total return": e1 / e0 - 1,
        "CAGR": (e1 / e0) ** (1 / years) - 1 if years > 0 and e1 > 0 else np.nan,
        "Ann. volatility": r.std() * np.sqrt(n_per_year),
        "Sharpe (excess)": ex.mean() / ex.std() * np.sqrt(n_per_year) if ex.std() > 0 else np.nan,
        "Sharpe (rf=0)": r.mean() / r.std() * np.sqrt(n_per_year) if r.std() > 0 else np.nan,
        "Sortino": (ex.mean() * n_per_year) / (ex[ex < 0].std() * np.sqrt(n_per_year)) if (ex < 0).sum() > 1 else np.nan,
        "Max drawdown": float(dd.min()),
        "VaR 95% (daily)": float(-np.percentile(r, 5)) if len(r) else np.nan,
        "CVaR 95% (daily)": float(-r[r <= np.percentile(r, 5)].mean()) if len(r) else np.nan,
    })
    out["Calmar"] = out["CAGR"] / abs(out["Max drawdown"]) if out["Max drawdown"] < 0 else np.nan
    if trades is None or len(trades) == 0:
        out["Total trades"] = 0
        return out
    pnl = trades["net_pnl"]
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    out.update({
        "Total trades": int(len(trades)),
        "Trades / year": len(trades) / years if years > 0 else np.nan,
        "Long trades": int((trades["direction"] == "LONG").sum()),
        "Short trades": int((trades["direction"] == "SHORT").sum()),
        "Win rate": len(wins) / len(trades),
        "Profit factor": wins.sum() / abs(losses.sum()) if losses.sum() != 0 else np.inf,
        "Expectancy ($/trade)": pnl.mean(),
        "Expectancy (R)": trades["R"].mean(),
        "Avg trade return (% equity)": (pnl / (trades["equity_after"] - pnl)).mean(),
        "Average win ($)": wins.mean() if len(wins) else 0.0,
        "Average loss ($)": losses.mean() if len(losses) else 0.0,
        "Payoff ratio": wins.mean() / abs(losses.mean()) if len(wins) and len(losses) and losses.mean() != 0 else np.nan,
        "Max consecutive losses": max_consecutive(pnl <= 0),
        "Avg holding (hours)": trades["holding_hours"].mean(),
        "Worst trade (R)": trades["R"].min(),
        "Trades losing > 1.1R (gaps/slippage)": int((trades["R"] < -1.1).sum()),
        "Total costs ($)": float(trades["commission"].sum() + trades["swap"].sum()),
    })
    return out


def acceptance_check(m: dict, crit, wf_positive_share: float | None = None) -> dict:
    checks = {
        f"Sharpe > {crit.min_sharpe}": m.get("Sharpe (excess)", np.nan) > crit.min_sharpe,
        f"Profit factor > {crit.min_profit_factor}": m.get("Profit factor", 0) > crit.min_profit_factor,
        f"Max DD < {crit.max_drawdown:.0%}": abs(m.get("Max drawdown", -1)) < crit.max_drawdown,
        "Net return > 0": m.get("Total return", -1) > crit.min_net_return,
        f"Trades >= {crit.min_trades_per_segment}": m.get("Total trades", 0) >= crit.min_trades_per_segment,
        "Expectancy (R) > 0": m.get("Expectancy (R)", -1) > crit.min_expectancy_r,
    }
    if wf_positive_share is not None:
        checks[f"Positive WF windows >= {crit.min_positive_wf_windows:.0%}"] = wf_positive_share >= crit.min_positive_wf_windows
    return checks
