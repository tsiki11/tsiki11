"""Indicators (Wilder smoothing where the original definition uses it).

All functions are causal: value at bar t uses data up to and including bar t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def wilder(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up, dn = d.clip(lower=0), (-d).clip(lower=0)
    rs = wilder(up, n) / wilder(dn, n)
    return 100 - 100 / (1 + rs)


def true_range(h: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
    pc = c.shift(1)
    return pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)


def atr(h, l, c, n: int = 14) -> pd.Series:
    return wilder(true_range(h, l, c), n)


def adx(h: pd.Series, l: pd.Series, c: pd.Series, n: int = 14) -> pd.DataFrame:
    up, dn = h.diff(), -l.diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=h.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=h.index)
    tr = wilder(true_range(h, l, c), n)
    pdi = 100 * wilder(plus_dm, n) / tr
    mdi = 100 * wilder(minus_dm, n) / tr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi)
    return pd.DataFrame({"plus_di": pdi, "minus_di": mdi, "adx": wilder(dx, n)})


def mid_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({k: (df[f"bid_{k}"] + df[f"ask_{k}"]) / 2 for k in ("open", "high", "low", "close")},
                        index=df.index)


def resample_ohlc(ohlc: pd.DataFrame, rule: str = "4h", offset: str | None = None,
                  min_bars: int = 0) -> pd.DataFrame:
    """Bars labelled by OPEN time (UTC, anchored at 00:00 + offset), e.g. 00-04, 04-08...
    Daily FX bars: rule="24h", offset="21h" -> days roll at 21:00 UTC (New York close)."""
    # origin="epoch": bins never depend on where the sample starts (look-ahead-test safe).
    # Use "24h" (not "1D") for daily bars: pandas ignores offset for calendar frequencies.
    r = ohlc.resample(rule, label="left", closed="left", origin="epoch", offset=offset)
    agg = r.agg({"open": "first", "high": "max", "low": "min", "close": "last"})
    if min_bars:   # drop stub bars (e.g. the 1-hour Friday/Sunday slivers around the weekend)
        agg = agg[r["close"].count() >= min_bars]
    return agg.dropna()


def align_htf(htf: pd.DataFrame, htf_period: str, ltf_index: pd.DatetimeIndex,
              ltf_period: str = "1h") -> pd.DataFrame:
    """Map higher-timeframe values onto lower-timeframe bars WITHOUT look-ahead.

    A HTF bar labelled L (open time) is only known at L + htf_period.
    A LTF bar labelled t is evaluated at its close t + ltf_period.
    => use the latest HTF bar with L + htf_period <= t + ltf_period.
    """
    avail = htf.copy()
    avail.index = avail.index + pd.Timedelta(htf_period)
    left = pd.DataFrame(index=ltf_index + pd.Timedelta(ltf_period))
    out = pd.merge_asof(left, avail, left_index=True, right_index=True, direction="backward")
    out.index = ltf_index
    return out


def htf_on_ltf(htf_values: pd.DataFrame, htf_period: str, ltf_index: pd.DatetimeIndex) -> pd.DataFrame:
    """align_htf + a `fresh` flag that is True only on the LTF bar at whose close a NEW
    HTF bar has just completed -> HTF events (crosses, breakouts) fire exactly once."""
    v = htf_values.copy()
    v["_bar"] = np.arange(len(v), dtype=float)
    out = align_htf(v, htf_period, ltf_index)
    out["fresh"] = out["_bar"].notna() & (out["_bar"] != out["_bar"].shift(1))
    return out.drop(columns="_bar")
