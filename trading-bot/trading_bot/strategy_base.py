"""Strategy interface. The backtester only consumes the signal frame, so
strategies can be swapped without touching the engine.

Signal frame (index = bar open time; values decided at that bar's CLOSE):
    entry      : +1 open long, -1 open short, 0 nothing   (executed next bar open)
    sl_dist    : stop-loss distance in price units (>0)
    tp_dist    : take-profit distance in price units (NaN = no TP)
    exit_long  : True -> close an open long at next bar open
    exit_short : True -> close an open short at next bar open
    quality    : optional numeric signal-quality score (diagnostics)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pandas as pd

REQUIRED = ["entry", "sl_dist", "tp_dist", "exit_long", "exit_short"]


@dataclass
class Strategy(ABC):
    params: dict = field(default_factory=dict)
    strategy_id: str = "BASE"
    name: str = "base"
    version: str = "0.0"
    hypothesis: str = ""
    timeframe: str = "1H"          # signal / confirmation timeframes (label for the tracker)

    def __post_init__(self):
        merged = dict(self.default_params())
        merged.update(self.params or {})
        self.params = merged

    @classmethod
    def default_params(cls) -> dict:
        return {}

    @classmethod
    def sensitivity_grid(cls) -> dict:
        """Parameter grid for the IS sensitivity diagnostic (never used to pick parameters)."""
        return {}

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        ...

    def signals(self, data: pd.DataFrame) -> pd.DataFrame:
        sig = self.generate_signals(data)
        missing = [c for c in REQUIRED if c not in sig]
        if missing:
            raise ValueError(f"{self.name}: missing signal columns {missing}")
        bad = sig.loc[sig["entry"] != 0, "sl_dist"]
        if (bad.isna() | (bad <= 0)).any():
            raise ValueError(f"{self.name}: entries without a valid stop distance")
        return sig
