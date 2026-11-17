"""Free data loaders.

Dukascopy (primary, intraday FX/metals):
  https://datafeed.dukascopy.com/datafeed/{SYMBOL}/{YYYY}/{MM-1:02d}/{BID|ASK}_candles_hour_1.bi5
  One LZMA-compressed file per month; each record = 24 bytes big-endian:
  int32 seconds-from-month-start, int32 open, close, low, high (in points), float32 volume.
  Timestamps are UTC, candle labelled by its OPEN time. Volume is Dukascopy's own
  traded/tick volume -- NOT total market volume (FX is decentralised).
  Non-trading hours are filled with flat zero-volume candles (removed by the validator).

Yahoo (secondary, cross-check only): EURUSD=X daily (long history) and 1h (last ~730 days).
"""
from __future__ import annotations

import io
import logging
import lzma
import struct
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_DIR, INSTRUMENTS

log = logging.getLogger(__name__)
DUKA_URL = "https://datafeed.dukascopy.com/datafeed/{sym}/{y}/{m:02d}/{side}_candles_hour_1.bi5"


def _fetch(url: str, retries: int = 8) -> bytes:
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "research-client/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return b""
            log.warning("HTTP %s for %s (try %d)", e.code, url, i + 1)
        except Exception as e:  # network hiccup
            log.warning("%s for %s (try %d)", e, url, i + 1)
        time.sleep(min(60, 5 * 2 ** i))   # Dukascopy throttles bursts -> back off
    raise RuntimeError(f"failed to download {url}")


def _decode_month(raw: bytes, year: int, month: int, point: float) -> pd.DataFrame:
    if not raw:
        return pd.DataFrame()
    data = lzma.decompress(raw)
    n = len(data) // 24
    arr = np.frombuffer(data[: n * 24], dtype=np.dtype([("t", ">i4"), ("o", ">i4"), ("c", ">i4"),
                                                         ("l", ">i4"), ("h", ">i4"), ("v", ">f4")]))
    base = pd.Timestamp(year=year, month=month, day=1, tz="UTC")
    idx = base + pd.to_timedelta(arr["t"].astype(np.int64), unit="s")
    return pd.DataFrame({"open": arr["o"] * point, "high": arr["h"] * point,
                         "low": arr["l"] * point, "close": arr["c"] * point,
                         "volume": arr["v"].astype(float)}, index=idx)


def load_dukascopy_h1(symbol: str, start: str, end: str, refresh: bool = False) -> pd.DataFrame:
    """Hourly BID+ASK OHLC (UTC). Cached per month under data/fx/{symbol}/."""
    spec = INSTRUMENTS[symbol]
    cache_dir = DATA_DIR / symbol
    cache_dir.mkdir(parents=True, exist_ok=True)
    months = pd.period_range(pd.Timestamp(start), pd.Timestamp(end), freq="M")
    frames = []
    for p in months:
        f = cache_dir / f"{p.year}-{p.month:02d}.csv.gz"
        if f.exists() and not refresh:
            frames.append(pd.read_csv(f, index_col=0, parse_dates=True))
            continue
        sides = {}
        for side in ("BID", "ASK"):
            url = DUKA_URL.format(sym=symbol, y=p.year, m=p.month - 1, side=side)
            sides[side] = _decode_month(_fetch(url), p.year, p.month, spec.point)
            time.sleep(1.0)  # be polite to the free service (bursts get throttled)
        if sides["BID"].empty or sides["ASK"].empty:
            log.warning("no data for %s %s", symbol, p)
            continue
        df = sides["BID"].add_prefix("bid_").join(sides["ASK"].add_prefix("ask_"), how="inner")
        is_current = p == pd.Timestamp.now(tz="UTC").tz_localize(None).to_period("M")
        if not is_current:            # never cache an incomplete month
            df.to_csv(f, compression="gzip")
        frames.append(df)
    out = pd.concat(frames).sort_index()
    out.index = pd.to_datetime(out.index, utc=True)
    out.index.name = "time_utc"
    return out.loc[pd.Timestamp(start, tz="UTC"): pd.Timestamp(end, tz="UTC")]


def load_yahoo(ticker: str, interval: str = "1d", start: str | None = None,
               period: str | None = None) -> pd.DataFrame:
    import yfinance as yf
    d = yf.download(ticker, start=start, period=period, interval=interval,
                    progress=False, auto_adjust=False, threads=False)
    if d.empty:
        return d
    d.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in d.columns]
    d.index = pd.to_datetime(d.index, utc=True)
    return d


def load_daily_riskfree(start: str) -> pd.Series:
    """Daily T-bill return proxy (BIL ETF total return) for Sharpe calculations."""
    d = load_yahoo("BIL", "1d", start=start)
    px = d["adj close"] if "adj close" in d else d["close"]
    r = px.pct_change().dropna()
    r.index = r.index.tz_convert(None).normalize()
    return r
