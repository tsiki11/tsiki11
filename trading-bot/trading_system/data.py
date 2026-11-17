"""Download and cache free daily price data from Yahoo Finance (yfinance).

All prices are dividend/split adjusted (auto_adjust=True), so total return
is captured. Data is cached to data/prices.csv so the backtest is reproducible
and does not hit Yahoo every run.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FIELDS = ["Open", "High", "Low", "Close"]

# Diversified, liquid ETFs (all with history from <=2007) + crypto + cash proxy
ETFS = ["SPY", "QQQ", "IWM", "EFA", "EEM", "VNQ",      # equities / REITs
        "TLT", "IEF", "LQD", "TIP",                   # bonds
        "GLD", "DBC", "UUP",                          # gold, commodities, USD
        "BIL"]                                        # cash proxy (1-3m T-bills)
CRYPTO = ["BTC-USD", "ETH-USD"]
ALL = ETFS + CRYPTO


def download(tickers=ALL, start="2005-01-01", end=None, retries=4) -> dict:
    frames = {f: {} for f in FIELDS}
    for t in tickers:
        for i in range(retries):
            try:
                d = yf.download(t, start=start, end=end, progress=False,
                                auto_adjust=True, threads=False)
                if len(d):
                    for f in FIELDS:
                        frames[f][t] = d[f].squeeze().rename(t)
                    break
            except Exception as e:  # network hiccup / rate limit
                print(f"{t}: {e}")
            time.sleep(2 * (i + 1))
        else:
            print(f"WARNING: no data for {t}")
    out = {}
    for f in FIELDS:
        px = pd.concat(frames[f].values(), axis=1, sort=True)
        px.index = pd.to_datetime(px.index).tz_localize(None)
        out[f] = px.sort_index()
    return out


def load(refresh: bool = False, drop_today: bool = True) -> dict:
    """Return {"Open","High","Low","Close"} -> DataFrame (dates x tickers).

    drop_today removes a possibly incomplete intraday bar for today's date.
    """
    paths = {f: DATA_DIR / f"prices_{f.lower()}.csv" for f in FIELDS}
    if refresh or not all(p.exists() for p in paths.values()):
        DATA_DIR.mkdir(exist_ok=True)
        for f, df in download().items():
            df.to_csv(paths[f])
    out = {f: pd.read_csv(p, index_col=0, parse_dates=True) for f, p in paths.items()}
    if drop_today:
        today = pd.Timestamp.today().normalize()
        out = {f: df[df.index < today] for f, df in out.items()}
    return out


if __name__ == "__main__":
    p = load(refresh=True)["Close"]
    print(p.apply(lambda s: s.first_valid_index()).to_string())
    print(p.tail(3).T)
