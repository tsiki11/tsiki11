"""Unit tests for the execution engine on synthetic data (python -m pytest tests -q)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trading_bot.backtester import run_backtest
from trading_bot.config import INSTRUMENTS, CostModel, RiskConfig
from trading_bot.risk_management import RiskManager
from trading_bot import indicators as I

SPEC = INSTRUMENTS["EURUSD"]
NOCOST = CostModel(commission_per_lot_side=0, slippage_pips=0, swap_per_lot_night=0)


def bars(rows, spread=0.0, start="2024-01-02 00:00"):
    """rows: list of (open, high, low, close) mid prices."""
    idx = pd.date_range(start, periods=len(rows), freq="h", tz="UTC")
    m = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)
    df = pd.DataFrame(index=idx)
    for k in m:
        df[f"bid_{k}"] = m[k] - spread / 2
        df[f"ask_{k}"] = m[k] + spread / 2
    df["bid_volume"] = df["ask_volume"] = 1.0
    return df


def sig(n, at, d, sl, tp, exits=()):
    s = pd.DataFrame({"entry": 0, "sl_dist": np.nan, "tp_dist": np.nan, "exit_long": False, "exit_short": False},
                     index=range(n))
    s.loc[at, ["entry", "sl_dist", "tp_dist"]] = [d, sl, tp]
    for e in exits:
        s.loc[e, "exit_long" if d > 0 else "exit_short"] = True
    return s


def run(df, s, cost=NOCOST):
    s.index = df.index
    return run_backtest(df, s, SPEC, RiskConfig(), cost)


def test_entry_next_open_and_tp():
    df = bars([(1.1, 1.1, 1.1, 1.1), (1.1000, 1.1005, 1.0995, 1.1000), (1.1000, 1.1030, 1.0999, 1.1020)])
    r = run(df, sig(3, 0, 1, 0.0010, 0.0020))
    t = r.trades.iloc[0]
    assert t["entry_time"] == df.index[1] and abs(t["entry"] - 1.1000) < 1e-12   # next bar open, not signal bar
    assert t["exit_reason"] == "TP" and abs(t["R"] - 2.0) < 0.02


def test_both_hit_is_conservative_stop():
    df = bars([(1.1, 1.1, 1.1, 1.1), (1.1000, 1.1030, 1.0985, 1.1000)])
    r = run(df, sig(2, 0, 1, 0.0010, 0.0020))
    assert r.trades.iloc[0]["exit_reason"] == "SL_both_hit" and r.trades.iloc[0]["R"] < -0.95


def test_gap_through_stop_loses_more_than_1R():
    df = bars([(1.1, 1.1, 1.1, 1.1), (1.1000, 1.1002, 1.0998, 1.1000), (1.0970, 1.0975, 1.0960, 1.0965)])
    r = run(df, sig(3, 0, 1, 0.0010, 0.0020))
    t = r.trades.iloc[0]
    assert t["exit_reason"] == "SL_gap_open" and t["R"] < -2.5


def test_short_uses_ask_and_spread_costs():
    df = bars([(1.1, 1.1, 1.1, 1.1), (1.1000, 1.1001, 1.0999, 1.1000), (1.1000, 1.1001, 1.0999, 1.1000)], spread=0.0002)
    r = run(df, sig(3, 0, -1, 0.0010, 0.0020, exits=[1]))
    t = r.trades.iloc[0]
    assert abs(t["entry"] - 1.0999) < 1e-12 and abs(t["exit"] - 1.1001) < 1e-12 and t["net_pnl"] < 0


def test_position_sizing():
    rm = RiskManager(RiskConfig(), SPEC)
    s = rm.size(10_000, 1.1000, 1.0980)        # 20 pip stop, 0.5% = $50 -> 25,000 units
    assert s.units == 25_000 and abs(s.planned_risk - 50) < 1e-6


def test_htf_alignment_has_no_lookahead():
    idx = pd.date_range("2024-01-01 00:00", periods=12, freq="h", tz="UTC")
    htf = pd.DataFrame({"v": [1.0, 2.0, 3.0]}, index=pd.date_range("2024-01-01", periods=3, freq="4h", tz="UTC"))
    a = I.align_htf(htf, "4h", idx)
    # 1H bar 02:00 closes 03:00 -> 4H bar 00:00 (complete at 04:00) not yet known
    assert np.isnan(a.loc[idx[2], "v"])
    # 1H bar 03:00 closes 04:00 -> 4H bar 00-04 complete -> value 1
    assert a.loc[idx[3], "v"] == 1.0 and a.loc[idx[7], "v"] == 2.0 and a.loc[idx[6], "v"] == 1.0
