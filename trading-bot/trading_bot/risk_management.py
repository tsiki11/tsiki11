"""Position sizing & exposure limits (independent of any strategy).

Risk Amount   = Equity x risk%                (risk% capped at max_risk_per_trade)
Units         = Risk Amount / (Stop Distance x V)
  V = account-currency value of a 1.0 price move per unit.
  For a USD account and a XXX/USD pair (EURUSD, XAUUSD) V = 1 USD.
  For other pairs V needs a conversion rate -> must be supplied explicitly.
Pip value (EURUSD, USD account) = 0.0001 x 100,000 = 10 USD per standard lot.
Units are rounded DOWN to the instrument's min step (micro lot), so the realised
planned risk is <= the target. Notional is capped by max_leverage.
NOTE: planned risk != realised loss. Gaps/slippage can make losses exceed -1R;
the trade log records both (R multiple).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .config import InstrumentSpec, RiskConfig


@dataclass
class SizeResult:
    units: float
    planned_risk: float
    notional: float
    capped_by_leverage: bool
    reason: str = ""


class RiskManager:
    def __init__(self, cfg: RiskConfig, spec: InstrumentSpec, account_ccy: str = "USD"):
        self.cfg, self.spec, self.account_ccy = cfg, spec, account_ccy

    def value_per_unit(self, quote_to_account: float | None = None) -> float:
        if self.spec.quote_ccy == self.account_ccy:
            return 1.0
        if quote_to_account is None:
            raise NotImplementedError(f"{self.spec.symbol}: quote {self.spec.quote_ccy} != account "
                                      f"{self.account_ccy}; a conversion rate is required")
        return quote_to_account

    def size(self, equity: float, entry: float, stop: float, risk_pct: float | None = None,
             quote_to_account: float | None = None) -> SizeResult:
        risk_pct = self.cfg.risk_per_trade if risk_pct is None else risk_pct
        risk_pct = min(risk_pct, self.cfg.max_risk_per_trade)
        dist = abs(entry - stop)
        if dist <= 0 or equity <= 0:
            return SizeResult(0, 0, 0, False, "invalid stop or equity")
        v = self.value_per_unit(quote_to_account)
        units = equity * risk_pct / (dist * v)
        capped = False
        max_units = equity * self.cfg.max_leverage / (entry * v)
        if units > max_units:
            units, capped = max_units, True
        step = self.spec.min_units
        units = math.floor(units / step + 1e-9) * step   # tolerance for float error
        if units <= 0:
            return SizeResult(0, 0, 0, capped, "below minimum lot")
        return SizeResult(units, units * dist * v, units * entry * v, capped)
