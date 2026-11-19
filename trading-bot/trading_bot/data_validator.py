"""Data quality checks run before every backtest. Nothing is silently filled."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def _ohlc_invalid(df: pd.DataFrame, p: str) -> pd.Series:
    o, h, l, c = (df[f"{p}_{k}"] for k in ("open", "high", "low", "close"))
    return (h < np.maximum(o, c) - 1e-12) | (l > np.minimum(o, c) + 1e-12) | (h < l) | (df[[f"{p}_{k}" for k in ("open", "high", "low", "close")]] <= 0).any(axis=1)


def is_market_open_hour(ts: pd.DatetimeIndex) -> np.ndarray:
    """FX retail session: Sun ~21:00/22:00 UTC -> Fri ~21:00/22:00 UTC.
    We treat Sat entirely closed, Fri >= 22:00 closed, Sun < 21:00 closed."""
    dow, hr = ts.dayofweek, ts.hour
    closed = (dow == 5) | ((dow == 4) & (hr >= 22)) | ((dow == 6) & (hr < 21))
    return ~np.asarray(closed)


def validate_fx(df: pd.DataFrame, source: str, asset: str, timeframe: str = "1H",
                spread_alert_pips: float = 5.0, pip: float = 1e-4) -> tuple[pd.DataFrame, dict]:
    """Return (clean_df, report). Removes only rows that are provably not tradable
    (flat zero-volume filler candles, duplicates); every removal is counted."""
    rep = {"source": source, "asset": asset, "timeframe": timeframe,
           "raw_rows": int(len(df)), "warnings": []}
    # chronology & duplicates
    rep["sorted_chronologically"] = bool(df.index.is_monotonic_increasing)
    if not rep["sorted_chronologically"]:
        df = df.sort_index(); rep["warnings"].append("index was not sorted -> sorted")
    dup = df.index.duplicated(keep="first")
    rep["duplicate_timestamps"] = int(dup.sum())
    df = df[~dup]
    rep["timezone"] = str(df.index.tz)
    if str(df.index.tz) != "UTC":
        rep["warnings"].append("timestamps not UTC")
    # missing values
    px_cols = [c for c in df.columns if not c.endswith("volume")]
    rep["missing_values"] = int(df[px_cols].isna().sum().sum())
    if df[[c for c in df.columns if c.endswith("volume")]].isna().all().all():
        rep["warnings"].append("source has no volume data")
    df = df.dropna(subset=px_cols)
    # flat filler candles (Dukascopy fills closed hours with O=H=L=C and volume 0)
    flat = ((df["bid_high"] == df["bid_low"]) & (df["bid_volume"] == 0) & (df["ask_volume"] == 0))
    rep["flat_zero_volume_rows_removed"] = int(flat.sum())
    in_session = is_market_open_hour(df.index)
    rep["flat_rows_inside_session"] = int((flat & in_session).sum())   # holidays / outages
    df = df[~flat]
    # candles stamped while the market is closed (e.g. stray Saturday prints) are not tradable
    off = ~is_market_open_hour(df.index)
    rep["out_of_session_rows_removed"] = int(off.sum())
    df = df[~off]
    # invalid OHLC
    bad = _ohlc_invalid(df, "bid") | _ohlc_invalid(df, "ask")
    rep["invalid_ohlc_rows"] = int(bad.sum())
    if bad.any():
        rep["warnings"].append(f"{int(bad.sum())} invalid OHLC rows removed")
        df = df[~bad]
    # bid/ask sanity
    spread = (df["ask_close"] - df["bid_close"]) / pip
    rep["negative_spread_rows"] = int((spread < 0).sum())
    rep["spread_pips_median"] = float(spread.median())
    rep["spread_pips_p95"] = float(spread.quantile(0.95))
    rep["spread_pips_p99"] = float(spread.quantile(0.99))
    rep["wide_spread_rows(>%.0fpip)" % spread_alert_pips] = int((spread > spread_alert_pips).sum())
    # abnormal moves: |hourly log return| > 10 rolling std (500h) or > 1.5%
    mid = (df["bid_close"] + df["ask_close"]) / 2
    lr = np.log(mid).diff()
    z = lr / lr.rolling(500, min_periods=100).std()
    ab = (z.abs() > 10) | (lr.abs() > 0.015)
    rep["abnormal_moves"] = int(ab.sum())
    rep["abnormal_move_times"] = [str(t) for t in df.index[ab.fillna(False).values][:20]]
    # gaps inside trading session (expected: weekends)
    gaps = df.index.to_series().diff()
    long_gaps = gaps[gaps > pd.Timedelta(hours=1)]
    weekend_like = [(g <= pd.Timedelta(hours=60)) and (t.dayofweek in (6, 0)) for t, g in long_gaps.items()]
    unexpected = long_gaps[~np.array(weekend_like, dtype=bool)] if len(long_gaps) else long_gaps
    rep["weekend_gaps"] = int(sum(weekend_like))
    rep["unexpected_gaps(>1h)"] = int(len(unexpected))
    rep["largest_unexpected_gaps"] = [f"{t} ({g})" for t, g in unexpected.sort_values(ascending=False).head(10).items()]
    # incomplete last candle
    now = pd.Timestamp.now(tz="UTC")
    if len(df) and df.index[-1] + pd.Timedelta(hours=1) > now:
        df = df.iloc[:-1]; rep["warnings"].append("dropped incomplete last candle")
    rep.update({"start": str(df.index[0]), "end": str(df.index[-1]), "clean_rows": int(len(df))})
    expected = int(is_market_open_hour(pd.date_range(df.index[0], df.index[-1], freq="h")).sum())
    rep["expected_session_hours"] = expected
    rep["missing_session_hours"] = max(expected - int(len(df)), 0)
    rep["missing_session_pct"] = round(100 * rep["missing_session_hours"] / expected, 3)
    if rep["missing_session_pct"] > 2:
        rep["warnings"].append("more than 2% of session hours missing")
    return df, rep


def cross_check_daily(h1: pd.DataFrame, yahoo_daily: pd.DataFrame) -> dict:
    """Compare the hourly feed's mid closes resampled to UTC days vs Yahoo EURUSD=X closes.
    Yahoo labels each FX daily bar one calendar day later than the UTC day it mostly covers
    (measured: corr 0.96 at lag -1 vs 0.03 at lag 0), so the lag-aligned figure is the one to read."""
    mid = ((h1["bid_close"] + h1["ask_close"]) / 2)
    d = mid.resample("1D").last().dropna()
    d.index = d.index.tz_convert(None)
    y = yahoo_daily["close"].copy(); y.index = y.index.tz_convert(None).normalize()
    j = pd.concat([d.rename("duka"), y.rename("yahoo")], axis=1).dropna()
    rd, ry = j["duka"].pct_change(), j["yahoo"].pct_change()
    lvl = (j["duka"] / j["yahoo"].shift(-1) - 1).abs()   # level diff on the aligned day
    return {"overlap_days": int(len(j)),
            "daily_return_corr_by_lag": {f"{k:+d}d": float(rd.corr(ry.shift(k))) for k in (-1, 0, 1)},
            "daily_return_corr_aligned": float(max(rd.corr(ry.shift(k)) for k in (-1, 0, 1))),
            "weekly_return_corr": float(j["duka"].resample("W").last().pct_change().corr(j["yahoo"].resample("W").last().pct_change())),
            "median_abs_level_diff_pct": float(lvl.median() * 100),
            "p99_abs_level_diff_pct": float(lvl.quantile(0.99) * 100)}
