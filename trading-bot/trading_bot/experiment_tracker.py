"""Append-only experiment log (JSONL) + comparison table (CSV). Nothing is overwritten."""
from __future__ import annotations

import json
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd

from .config import RESULTS_DIR

LOG = RESULTS_DIR / "experiments.jsonl"
TABLE = RESULTS_DIR / "strategy_comparison.csv"


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp, datetime, date)):
        return str(o)
    return o


def log_experiment(record: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    record = {"logged_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), **record}
    with open(LOG, "a") as f:
        f.write(json.dumps(_clean(record)) + "\n")
    rebuild_table()


def rebuild_table() -> pd.DataFrame:
    rows = []
    for line in open(LOG):
        r = json.loads(line)
        row = {k: r.get(k) for k in ("logged_at_utc", "experiment_id", "strategy_id", "strategy_name", "version",
                                     "asset", "timeframe", "final_decision")}
        for seg in ("in_sample", "validation", "out_of_sample"):
            m = (r.get("results") or {}).get(seg) or {}
            for k in ("Total return", "Sharpe (excess)", "Profit factor", "Max drawdown", "Total trades", "Expectancy (R)"):
                row[f"{seg}: {k}"] = m.get(k)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(TABLE, index=False)
    return df
