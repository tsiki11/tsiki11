"""S003 - 4H Donchian channel breakout with DAILY trend filter (Turtle-style trend following).

Hypothesis: when EURUSD closes a 4H bar beyond its 20-bar (~3.5 trading days) range in the
direction of the daily trend (D1 MA20 vs MA50), the breakout continues often and far enough
that a 3R target / trailing channel exit pays for the frequent small losses net of costs.
Channel breakouts are the canonical trend-following entry (Donchian; Turtle rules, Faith 2007;
Clenow 2013) and capture the multi-day FX momentum documented by Menkhoff et al. (2012).

Behaviour exploited : continuation after range expansion in the trend direction.
Expected to work    : trending regimes; low win rate (~30-40%) with large winners.
Expected to fail    : choppy ranges with false breakouts.
Data                : 1H bid/ask (execution); 4H and daily bars resampled from 1H mid.
                      Daily bars roll at 21:00 UTC; stub bars (<6 hours) dropped.
Main risks          : false breakouts, low trade count, long losing streaks.

Exact rules (evaluated when a 4H bar has just COMPLETED, mid prices):
  LONG  : 4H close > highest 4H high of the previous entry_n bars
          AND last COMPLETED daily bar: D1 MA20 > D1 MA50
  SHORT : 4H close < lowest 4H low of the previous entry_n bars AND D1 MA20 < D1 MA50
  Entry : next 1H bar open (= next 4H open), ask for longs / bid for shorts + slippage
  Stop  : sl_atr x ATR14(4H);  Take profit: rr x stop distance
  Exit  : 4H close below the lowest low of the previous exit_n bars (long) / above the
          highest high (short) -> close at next bar open (if SL/TP not hit first)
  Max 1 position; no pyramiding.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import indicators as I
from ..strategy_base import Strategy


class Donchian4H(Strategy):
    def __init__(self, params: dict | None = None):
        super().__init__(params=params or {}, strategy_id="S003", name="4H Donchian breakout + D1 trend",
                         version="1.0", hypothesis=__doc__.split("\n\n")[1].strip(), timeframe="4H entry, D1 confirm")

    @classmethod
    def default_params(cls) -> dict:
        return {"entry_n": 20, "exit_n": 10, "fast": 20, "slow": 50, "atr_n": 14,
                "sl_atr": 2.0, "rr": 3.0}

    @classmethod
    def sensitivity_grid(cls) -> dict:
        return {"entry_n": [10, 20, 40], "sl_atr": [1.5, 2.0, 3.0], "rr": [2.0, 3.0, 5.0]}

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        m = I.mid_ohlc(data)
        h4 = I.resample_ohlc(m, "4h")
        hi_in = h4.high.rolling(p["entry_n"]).max().shift(1)
        lo_in = h4.low.rolling(p["entry_n"]).min().shift(1)
        hi_out = h4.high.rolling(p["exit_n"]).max().shift(1)
        lo_out = h4.low.rolling(p["exit_n"]).min().shift(1)
        v4 = pd.DataFrame({"bo_up": h4.close > hi_in, "bo_dn": h4.close < lo_in,
                           "x_long": h4.close < lo_out, "x_short": h4.close > hi_out,
                           "atr": I.atr(h4.high, h4.low, h4.close, p["atr_n"])}, index=h4.index)
        d1 = I.resample_ohlc(m, "24h", offset="21h", min_bars=6)
        vd = pd.DataFrame({"f": I.sma(d1.close, p["fast"]), "s": I.sma(d1.close, p["slow"])}, index=d1.index)
        h = I.htf_on_ltf(v4, "4h", data.index)
        d = I.align_htf(vd, "24h", data.index)
        fresh = h["fresh"]
        long_ = h["bo_up"].eq(True) & fresh & (d["f"] > d["s"])
        short_ = h["bo_dn"].eq(True) & fresh & (d["f"] < d["s"])
        entry = pd.Series(np.where(long_, 1, np.where(short_, -1, 0)), index=data.index)
        sl = p["sl_atr"] * h["atr"]
        out = pd.DataFrame({"entry": entry, "sl_dist": sl.where(entry != 0),
                            "tp_dist": (p["rr"] * sl).where(entry != 0),
                            "exit_long": h["x_long"].eq(True) & fresh,
                            "exit_short": h["x_short"].eq(True) & fresh,
                            "quality": (d["f"] / d["s"] - 1).abs()}, index=data.index)
        out.loc[h["atr"].isna() | d["s"].isna(), "entry"] = 0
        return out
