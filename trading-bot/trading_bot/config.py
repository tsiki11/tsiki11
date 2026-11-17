"""Central configuration. Every assumption lives here and is logged per experiment.

Nothing in this file is a fact about a specific broker: contract sizes, costs and
leverage are ASSUMPTIONS for simulation and must be replaced with real broker
specifications before paper/live trading.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "fx"
RESULTS_DIR = ROOT / "results" / "fx"


@dataclass(frozen=True)
class InstrumentSpec:
    symbol: str                 # Dukascopy symbol
    point: float                # price increment used by the data feed (1e-5 for EURUSD)
    pip: float                  # 1 pip in price units
    contract_size: float        # units per 1 standard lot
    quote_ccy: str              # P&L currency of 1 unit price move
    min_units: float            # smallest tradable step (micro lot = 1,000)


INSTRUMENTS = {
    "EURUSD": InstrumentSpec("EURUSD", point=1e-5, pip=1e-4, contract_size=100_000,
                             quote_ccy="USD", min_units=1_000),
}


@dataclass
class CostModel:
    """Transaction-cost ASSUMPTIONS (ECN-style account).

    * spread: taken from the data (real Dukascopy bid/ask at each bar).
    * extra_spread_pips: added on top of the data spread (0 = Dukascopy ECN spread).
    * commission_per_lot_side: USD per 100k units per side (typical ECN $3.5).
    * slippage_pips: adverse slippage on market & stop orders (not on TP limits).
    * swap_per_lot_night: USD charged per standard lot per night held, BOTH
      directions (conservative: ignores positive carry); x3 on Wednesday rollover.
    """
    extra_spread_pips: float = 0.0
    commission_per_lot_side: float = 3.5
    slippage_pips: float = 0.1
    swap_per_lot_night: float = 6.0
    rollover_hour_utc: int = 21


@dataclass
class RiskConfig:
    initial_equity: float = 10_000.0
    risk_per_trade: float = 0.005          # 0.5% default
    max_risk_per_trade: float = 0.01       # hard cap 1%
    max_leverage: float = 30.0             # ASSUMPTION (ESMA retail cap for majors)
    max_open_positions: int = 1


@dataclass
class SplitConfig:
    """Chronological 60/20/20 split on complete months 2016-01 .. 2026-10 (130 months)."""
    warmup_start: str = "2015-07-01"
    is_start: str = "2016-01-01"
    val_start: str = "2022-07-01"
    oos_start: str = "2024-09-01"
    end: str = "2026-10-31 23:59"


@dataclass
class AcceptanceCriteria:
    """Fixed BEFORE any test. Must not be changed after seeing results."""
    min_sharpe: float = 1.0
    min_profit_factor: float = 1.20
    max_drawdown: float = 0.20
    min_net_return: float = 0.0
    min_trades_per_segment: int = 50
    min_expectancy_r: float = 0.0
    min_positive_wf_windows: float = 0.60   # share of walk-forward windows with profit > 0

    # Development gate (pre-registered): the untouched OOS segment is only opened for a
    # strategy that passes ALL criteria on the Validation segment and has positive
    # expectancy in-sample. Otherwise it is REJECTED/NEEDS IMPROVEMENT without burning OOS data.
    gate_rule: str = "validation passes all criteria AND in-sample expectancy > 0"

    def as_dict(self):
        return asdict(self)
