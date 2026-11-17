"""Bar-by-bar execution simulator for one instrument (bid/ask OHLC).

Execution model
  * Signals are computed on the CLOSE of bar t and executed at the OPEN of bar t+1
    (no look-ahead). Long entries fill at ASK open + slippage, shorts at BID open - slippage.
  * Longs are closed on BID prices, shorts on ASK prices (stops/targets trigger on the
    side you would actually trade).
  * Order at each bar open: (1) gap through SL/TP of an open position,
    (2) signal exits, (3) new entries (allows a reversal on the same open).
  * Intrabar: if both SL and TP lie inside the same bar's range the order is unknown
    -> CONSERVATIVE: assume the stop was hit first.
  * Gaps: if a bar opens beyond the stop, the fill is at the open (loss can exceed -1R).
    A gap beyond the TP is filled at the TP (conservative, ignores price improvement).
  * Costs: real bid/ask spread from the data (x spread_multiplier, + extra pips),
    commission per lot per side, slippage on market/stop fills, swap per night.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import CostModel, InstrumentSpec, RiskConfig
from .risk_management import RiskManager

log = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity: pd.Series                 # mark-to-market equity at each bar close
    params: dict = field(default_factory=dict)


def apply_spread_model(df: pd.DataFrame, cost: CostModel, pip: float, spread_multiplier: float = 1.0) -> pd.DataFrame:
    if spread_multiplier == 1.0 and cost.extra_spread_pips == 0:
        return df
    out = df.copy()
    for k in ("open", "high", "low", "close"):
        mid = (df[f"bid_{k}"] + df[f"ask_{k}"]) / 2
        half = (df[f"ask_{k}"] - df[f"bid_{k}"]) / 2 * spread_multiplier + cost.extra_spread_pips * pip / 2
        out[f"bid_{k}"], out[f"ask_{k}"] = mid - half, mid + half
    return out


def run_backtest(data: pd.DataFrame, signals: pd.DataFrame, spec: InstrumentSpec,
                 risk: RiskConfig, cost: CostModel, start=None, end=None,
                 spread_multiplier: float = 1.0) -> BacktestResult:
    """`data`/`signals` may include warm-up history; trading only between start and end."""
    df = apply_spread_model(data, cost, spec.pip, spread_multiplier)
    sig = signals.reindex(df.index)
    if start is not None or end is not None:
        mask = np.ones(len(df), dtype=bool)
        if start is not None: mask &= df.index >= pd.Timestamp(start, tz="UTC")
        if end is not None: mask &= df.index <= pd.Timestamp(end, tz="UTC")
        df, sig = df[mask], sig[mask]
    rm = RiskManager(risk, spec)
    slip = cost.slippage_pips * spec.pip
    t = df.index
    bo, bh, bl, bc = (df[f"bid_{k}"].values for k in ("open", "high", "low", "close"))
    ao, ah, al, ac = (df[f"ask_{k}"].values for k in ("open", "high", "low", "close"))
    entry_sig = sig["entry"].fillna(0).astype(int).values
    sl_d, tp_d = sig["sl_dist"].values, sig["tp_dist"].values
    ex_l = sig["exit_long"].fillna(False).astype(bool).values
    ex_s = sig["exit_short"].fillna(False).astype(bool).values

    equity = risk.initial_equity
    eq_curve = np.empty(len(df))
    trades = []
    pos = None              # dict for the open position
    pend_entry = None       # (direction, sl_dist, tp_dist, signal_time)
    pend_exit = False

    def commission(units):
        return units / spec.contract_size * cost.commission_per_lot_side

    def close(i, price, reason):
        nonlocal equity, pos
        d = pos["dir"]
        gross = (price - pos["entry"]) * pos["units"] * d
        com = commission(pos["units"])
        net = gross - pos["com_in"] - com - pos["swap"]
        equity += gross - com - pos["swap"]          # entry commission already deducted
        trades.append({**{k: pos[k] for k in ("signal_time", "entry_time", "entry", "sl", "tp", "units", "planned_risk")},
                       "direction": "LONG" if d > 0 else "SHORT", "exit_time": t[i], "exit": price,
                       "exit_reason": reason, "gross_pnl": gross, "commission": pos["com_in"] + com,
                       "swap": pos["swap"], "net_pnl": net, "R": net / pos["planned_risk"],
                       "bars_held": i - pos["i"] + (0 if reason.endswith("open") else 1),
                       "equity_after": equity})
        pos = None

    for i in range(len(df)):
        # ---------- at the OPEN of bar i
        if pos is not None:
            d = pos["dir"]
            px_open = bo[i] if d > 0 else ao[i]
            if (d > 0 and px_open <= pos["sl"]) or (d < 0 and px_open >= pos["sl"]):
                close(i, px_open - d * slip, "SL_gap_open")
            elif (d > 0 and px_open >= pos["tp"]) or (d < 0 and px_open <= pos["tp"]):
                close(i, pos["tp"], "TP_gap_open")
        if pos is not None and pend_exit:
            d = pos["dir"]
            close(i, (bo[i] - slip) if d > 0 else (ao[i] + slip), "signal_exit_open")
        pend_exit = False
        if pend_entry is not None and pos is None:
            d, sld, tpd, st = pend_entry
            fill = ao[i] + slip if d > 0 else bo[i] - slip
            sl = fill - d * sld
            tp = fill + d * tpd if not np.isnan(tpd) else (np.inf if d > 0 else -np.inf)
            sz = rm.size(equity, fill, sl)
            if sz.units > 0:
                com_in = commission(sz.units)
                equity -= com_in
                pos = {"dir": d, "entry": fill, "sl": sl, "tp": tp, "units": sz.units,
                       "planned_risk": sz.planned_risk, "com_in": com_in, "swap": 0.0,
                       "i": i, "signal_time": st, "entry_time": t[i]}
        pend_entry = None
        # ---------- swap: position open across the rollover hour
        if pos is not None and t[i].hour == cost.rollover_hour_utc and pos["i"] < i:
            nights = 3 if t[i].dayofweek == 2 else 1
            sw = pos["units"] / spec.contract_size * cost.swap_per_lot_night * nights
            pos["swap"] += sw
            equity -= sw
        # ---------- intrabar SL / TP
        if pos is not None:
            d = pos["dir"]
            if d > 0:
                sl_hit, tp_hit = bl[i] <= pos["sl"], bh[i] >= pos["tp"]
            else:
                sl_hit, tp_hit = ah[i] >= pos["sl"], al[i] <= pos["tp"]
            if sl_hit:                                  # conservative when both are hit
                close(i, pos["sl"] - d * slip, "SL" if not tp_hit else "SL_both_hit")
            elif tp_hit:
                close(i, pos["tp"], "TP")
        # ---------- at the CLOSE of bar i: mark to market, read signals
        if pos is not None:
            d = pos["dir"]
            mtm = (bc[i] if d > 0 else ac[i]) - pos["entry"]
            eq_curve[i] = equity + mtm * pos["units"] * d
            if (d > 0 and ex_l[i]) or (d < 0 and ex_s[i]):
                pend_exit = True
        else:
            eq_curve[i] = equity
        e = entry_sig[i]
        if e != 0 and i + 1 < len(df):
            flat_next = pos is None or pend_exit
            if flat_next or (pos is not None and pos["dir"] != e and pend_exit):
                pend_entry = (e, sl_d[i], tp_d[i], t[i])
    if pos is not None:                                  # close at the end of the test window
        i = len(df) - 1
        close(i, bc[i] if pos["dir"] > 0 else ac[i], "end_of_test")
        eq_curve[i] = equity
    tr = pd.DataFrame(trades)
    if len(tr):
        tr["holding_hours"] = (tr["exit_time"] - tr["entry_time"]).dt.total_seconds() / 3600
        tr["lots"] = tr["units"] / spec.contract_size
    return BacktestResult(trades=tr, equity=pd.Series(eq_curve, index=t, name="equity"))
