"""Μετατροπή αποδόσεων σε πιθανότητες, εύρεση value και staking.

Παράδειγμα: αποδόσεις 1X2 = 2.00 / 3.40 / 4.00
  implied = 1/2.00 + 1/3.40 + 1/4.00 = 0.500 + 0.294 + 0.250 = 1.044
  -> margin (overround) 4.4%. Αφαιρώντας το αναλογικά: 0.479 / 0.282 / 0.239

Value (expected value ανά 1 μονάδα): EV = p_model * odds - 1
  π.χ. p_model = 0.55 σε απόδοση 2.00 -> EV = +10%
"""
from __future__ import annotations

import numpy as np


def implied(odds) -> np.ndarray:
    return 1.0 / np.asarray(odds, dtype=float)


def overround(odds) -> float:
    return float(implied(odds).sum() - 1)


def devig(odds) -> np.ndarray:
    """Αναλογική αφαίρεση margin. Δέχεται 1D (ένας αγώνας) ή 2D (γραμμές = αγώνες)."""
    p = implied(odds)
    return p / p.sum(axis=-1, keepdims=True)


def expected_value(p, odds):
    return np.asarray(p) * np.asarray(odds) - 1


def fair_odds(p):
    p = np.asarray(p, dtype=float)
    with np.errstate(divide="ignore"):
        return np.where(p > 0, 1 / p, np.inf)


def kelly_fraction(p, odds, fraction: float = 1.0, cap: float | None = None):
    """Kelly: f* = (p*o - 1) / (o - 1). Αρνητικό -> δεν ποντάρουμε.
    Το full Kelly είναι πολύ επιθετικό όταν το p είναι εκτίμηση, γι' αυτό fractional + cap."""
    p, odds = np.asarray(p, float), np.asarray(odds, float)
    f = (p * odds - 1) / (odds - 1)
    f = np.clip(f * fraction, 0, None)
    if cap is not None:
        f = np.minimum(f, cap)
    return f
