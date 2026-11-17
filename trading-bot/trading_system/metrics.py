"""Performance statistics. Sharpe is computed on EXCESS returns over T-bills."""
from __future__ import annotations

import numpy as np
import pandas as pd

ANN = 252


def stats(ret: pd.Series, rf: pd.Series | None = None, turnover: pd.Series | None = None) -> dict:
    ret = ret.dropna()
    if rf is None:
        rf = pd.Series(0.0, index=ret.index)
    rf = rf.reindex(ret.index).fillna(0.0)
    ex = ret - rf
    eq = (1 + ret).cumprod()
    years = len(ret) / ANN
    cagr = eq.iloc[-1] ** (1 / years) - 1
    vol = ret.std() * np.sqrt(ANN)
    sharpe = ex.mean() / ex.std() * np.sqrt(ANN) if ex.std() > 0 else np.nan
    downside = ex[ex < 0].std() * np.sqrt(ANN)
    sortino = ex.mean() * ANN / downside if downside > 0 else np.nan
    dd = eq / eq.cummax() - 1
    mdd = dd.min()
    out = {
        "Total return": eq.iloc[-1] - 1,
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe (excess)": sharpe,
        "Sharpe (rf=0)": ret.mean() / ret.std() * np.sqrt(ANN) if ret.std() > 0 else np.nan,
        "Sortino": sortino,
        "Max drawdown": mdd,
        "Calmar": cagr / abs(mdd) if mdd < 0 else np.nan,
        "Best year": ret.groupby(ret.index.year).apply(lambda r: (1 + r).prod() - 1).max(),
        "Worst year": ret.groupby(ret.index.year).apply(lambda r: (1 + r).prod() - 1).min(),
        "% positive months": (ret.resample("ME").apply(lambda r: (1 + r).prod() - 1) > 0).mean(),
    }
    if turnover is not None:
        out["Turnover / yr"] = turnover.reindex(ret.index).sum() / years
    return out


def yearly(ret: pd.Series) -> pd.Series:
    return ret.groupby(ret.index.year).apply(lambda r: (1 + r).prod() - 1)


def table(d: dict) -> pd.DataFrame:
    df = pd.DataFrame(d)
    fmt = df.copy().astype(object)
    for idx in df.index:
        for c in df.columns:
            v = df.loc[idx, c]
            if idx in ("Sharpe (excess)", "Sharpe (rf=0)", "Sortino", "Calmar", "Turnover / yr"):
                fmt.loc[idx, c] = f"{v:.2f}"
            else:
                fmt.loc[idx, c] = f"{v:.1%}"
    return fmt
