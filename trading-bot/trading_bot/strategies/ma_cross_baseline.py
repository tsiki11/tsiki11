"""S001 - Baseline: 1H MA20/MA50 crossover with 4H trend confirmation.

Hypothesis: when a short-term trend (1H MA20 crossing MA50) starts in the same
direction as an established higher-timeframe trend (4H MA20 vs MA50, ADX>threshold),
price continues far enough in that direction to pay a 2R target more often than
1-in-3 times (break-even win rate for 1:2 R:R before costs is 33.3%).

Behaviour exploited : trend persistence / momentum across timeframes.
Expected to work    : sustained directional regimes (e.g. 2022 USD rally).
Expected to fail    : range-bound, mean-reverting markets (EURUSD spends much
                      of its time ranging) -> repeated whipsaws and stop-outs.
Data                : 1H bid/ask OHLC (UTC); 4H built by resampling 1H mid prices.
Main risks          : MA crosses lag (late entries), costs on many small trades,
                      regime dependence.
Accept / reject     : pre-registered acceptance criteria in config.AcceptanceCriteria.

Exact rules (evaluated on the CLOSE of each 1H bar t, mid prices):
  LONG  : MA20 crosses above MA50 on 1H (MA20[t] > MA50[t] and MA20[t-1] <= MA50[t-1])
          AND RSI14[t] < 70
          AND last COMPLETED 4H bar: MA20_4H > MA50_4H AND ADX14_4H > adx_min
  SHORT : mirror (cross below, RSI14 > 30, MA20_4H < MA50_4H, ADX14_4H > adx_min)
  Entry : next 1H bar open (ask for longs, bid for shorts) + slippage
  Stop  : sl_atr x ATR14(1H) from entry;  Take profit: rr x stop distance
  Exit  : opposite 1H MA cross -> close at next bar open (if SL/TP not hit first)
  Max 1 position; no pyramiding.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import indicators as I
from ..strategy_base import Strategy


class MACrossBaseline(Strategy):
    def __init__(self, params: dict | None = None):
        super().__init__(params=params or {}, strategy_id="S001", name="MA20/50 cross + 4H trend",
                         version="1.0", hypothesis=__doc__.split("\n\n")[1].strip())

    @classmethod
    def default_params(cls) -> dict:
        return {"fast": 20, "slow": 50, "rsi_n": 14, "rsi_long_max": 70, "rsi_short_min": 30,
                "adx_n": 14, "adx_min": 20, "atr_n": 14, "sl_atr": 1.5, "rr": 2.0,
                "htf": "4h", "exit_on_opposite_cross": True}

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        m = I.mid_ohlc(data)
        f, s = I.sma(m.close, p["fast"]), I.sma(m.close, p["slow"])
        r = I.rsi(m.close, p["rsi_n"])
        a = I.atr(m.high, m.low, m.close, p["atr_n"])
        cross_up = (f > s) & (f.shift(1) <= s.shift(1))
        cross_dn = (f < s) & (f.shift(1) >= s.shift(1))

        h4 = I.resample_ohlc(m, p["htf"])
        h4i = pd.DataFrame({"f": I.sma(h4.close, p["fast"]), "s": I.sma(h4.close, p["slow"]),
                            "adx": I.adx(h4.high, h4.low, h4.close, p["adx_n"])["adx"]}, index=h4.index)
        h = I.align_htf(h4i, p["htf"], data.index)
        up4 = (h["f"] > h["s"]) & (h["adx"] > p["adx_min"])
        dn4 = (h["f"] < h["s"]) & (h["adx"] > p["adx_min"])

        long_ = cross_up & (r < p["rsi_long_max"]) & up4
        short_ = cross_dn & (r > p["rsi_short_min"]) & dn4
        entry = pd.Series(np.where(long_, 1, np.where(short_, -1, 0)), index=data.index)
        sl = p["sl_atr"] * a
        out = pd.DataFrame({
            "entry": entry,
            "sl_dist": sl.where(entry != 0),
            "tp_dist": (p["rr"] * sl).where(entry != 0),
            "exit_long": cross_dn if p["exit_on_opposite_cross"] else False,
            "exit_short": cross_up if p["exit_on_opposite_cross"] else False,
            "quality": h["adx"],
        }, index=data.index)
        out.loc[a.isna() | h["adx"].isna(), "entry"] = 0
        return out
