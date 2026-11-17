"""Monte Carlo on the trade sequence.

Resamples trade R-multiples with replacement (bootstrap) and replays them with
fixed-fractional risk. Answers: how bad can drawdowns get with the SAME edge but
a different ordering/sample of trades, and how likely is a loss over N trades?
Limitation: assumes trades are independent (ignores regime clustering).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def bootstrap_trades(R: pd.Series, risk_pct: float, n_sims: int = 5000, n_trades: int | None = None,
                     seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    R = np.asarray(R.dropna(), dtype=float)
    n = n_trades or len(R)
    if len(R) == 0:
        return {}
    samples = rng.choice(R, size=(n_sims, n), replace=True)
    growth = 1 + risk_pct * samples
    eq = np.cumprod(growth, axis=1)
    final = eq[:, -1] - 1
    peak = np.maximum.accumulate(np.concatenate([np.ones((n_sims, 1)), eq], axis=1), axis=1)[:, 1:]
    mdd = (eq / peak - 1).min(axis=1)
    return {"n_sims": n_sims, "n_trades": n, "risk_pct": risk_pct,
            "final_return_p5": float(np.percentile(final, 5)),
            "final_return_p50": float(np.percentile(final, 50)),
            "final_return_p95": float(np.percentile(final, 95)),
            "prob_loss": float((final < 0).mean()),
            "max_dd_p50": float(np.percentile(mdd, 50)),
            "max_dd_p95_worst": float(np.percentile(mdd, 5)),
            "mean_R_ci95": [float(np.percentile(samples.mean(axis=1), 2.5)),
                            float(np.percentile(samples.mean(axis=1), 97.5))]}
