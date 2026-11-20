"""S002 - S001 lifted one timeframe: 4H MA20/MA50 crossover with DAILY trend confirmation.

Hypothesis: the S001 idea failed on 1H, where EURUSD noise dominates; at a 4x longer
horizon, MA crosses aligned with the daily trend (D1 MA20 vs MA50, ADX>20) catch
moves large enough to reach a 2R target more often than 1-in-3 times net of costs.
FX trend-following is documented mainly at multi-day horizons (Menkhoff et al. 2012,
Moskowitz-Ooi-Pedersen 2012), so moving up the timeframe is the natural first test.

Behaviour exploited : multi-day trend persistence.
Expected to work    : sustained USD trends (2018, 2021-22, 2024-25).
Expected to fail    : long ranges (2016-17, 2019-20) -> whipsaws.
Data                : 1H bid/ask (execution); 4H and daily bars resampled from 1H mid.
                      Daily bars roll at 21:00 UTC (New York close); stub bars (<6 hours) dropped.
Main risks          : very few signals (~5-10/yr) -> statistically weak and likely to fail the
                      pre-registered 50-trades-per-segment criterion BY CONSTRUCTION.

Exact rules (evaluated when a 4H bar has just COMPLETED, mid prices):
  LONG  : 4H MA20 crosses above 4H MA50 AND 4H RSI14 < 70
          AND last COMPLETED daily bar: D1 MA20 > D1 MA50 AND D1 ADX14 > adx_min
  SHORT : mirror (cross below, RSI14 > 30, D1 MA20 < MA50, ADX14 > adx_min)
  Entry : next 1H bar open (= next 4H open), ask for longs / bid for shorts + slippage
  Stop  : sl_atr x ATR14(4H);  Take profit: rr x stop distance
  Exit  : opposite 4H MA cross -> close at next bar open (if SL/TP not hit first)
  Max 1 position; no pyramiding.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import indicators as I
from ..strategy_base import Strategy


class MACross4H(Strategy):
    def __init__(self, params: dict | None = None):
        super().__init__(params=params or {}, strategy_id="S002", name="4H MA20/50 cross + D1 trend",
                         version="1.0", hypothesis=__doc__.split("\n\n")[1].strip(), timeframe="4H entry, D1 confirm")

    @classmethod
    def default_params(cls) -> dict:
        return {"fast": 20, "slow": 50, "rsi_n": 14, "rsi_long_max": 70, "rsi_short_min": 30,
                "adx_n": 14, "adx_min": 20, "atr_n": 14, "sl_atr": 1.5, "rr": 2.0}

    @classmethod
    def sensitivity_grid(cls) -> dict:
        return {"sl_atr": [1.0, 1.5, 2.0], "rr": [1.5, 2.0, 3.0], "adx_min": [15, 20, 25]}

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        m = I.mid_ohlc(data)
        h4 = I.resample_ohlc(m, "4h")
        f, s = I.sma(h4.close, p["fast"]), I.sma(h4.close, p["slow"])
        v4 = pd.DataFrame({
            "up": (f > s) & (f.shift(1) <= s.shift(1)),
            "dn": (f < s) & (f.shift(1) >= s.shift(1)),
            "rsi": I.rsi(h4.close, p["rsi_n"]),
            "atr": I.atr(h4.high, h4.low, h4.close, p["atr_n"]),
        }, index=h4.index)
        d1 = I.resample_ohlc(m, "24h", offset="21h", min_bars=6)
        vd = pd.DataFrame({"f": I.sma(d1.close, p["fast"]), "s": I.sma(d1.close, p["slow"]),
                           "adx": I.adx(d1.high, d1.low, d1.close, p["adx_n"])["adx"]}, index=d1.index)
        h = I.htf_on_ltf(v4, "4h", data.index)
        d = I.align_htf(vd, "24h", data.index)
        fresh = h["fresh"]
        up = h["up"].eq(True) & fresh
        dn = h["dn"].eq(True) & fresh
        upD = (d["f"] > d["s"]) & (d["adx"] > p["adx_min"])
        dnD = (d["f"] < d["s"]) & (d["adx"] > p["adx_min"])
        long_ = up & (h["rsi"] < p["rsi_long_max"]) & upD
        short_ = dn & (h["rsi"] > p["rsi_short_min"]) & dnD
        entry = pd.Series(np.where(long_, 1, np.where(short_, -1, 0)), index=data.index)
        sl = p["sl_atr"] * h["atr"]
        out = pd.DataFrame({"entry": entry, "sl_dist": sl.where(entry != 0),
                            "tp_dist": (p["rr"] * sl).where(entry != 0),
                            "exit_long": dn, "exit_short": up, "quality": d["adx"]}, index=data.index)
        out.loc[h["atr"].isna() | d["adx"].isna(), "entry"] = 0
        return out
