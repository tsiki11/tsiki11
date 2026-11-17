"""Run a complete, logged experiment.

    python -m trading_bot.main --strategy S001                # IS + Validation (OOS stays locked)
    python -m trading_bot.main --strategy S001 --run-oos      # only allowed if the dev gate is passed

Outputs -> results/fx/<experiment_id>/ : Excel workbook, CSV trade logs, equity CSV,
charts, data-quality report; appended to results/fx/experiments.jsonl.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from dataclasses import asdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import data_loader as DL, data_validator as DV, experiment_tracker as ET
from . import monte_carlo as MC, optimizer as OPT, performance as P, visualization as V, walk_forward as WF
from .backtester import run_backtest
from .config import (INSTRUMENTS, RESULTS_DIR, ROOT, AcceptanceCriteria, CostModel, RiskConfig, SplitConfig)
from .strategies import REGISTRY

warnings.filterwarnings("ignore")
log = logging.getLogger("trading_bot")


def load_rf() -> pd.Series:
    cache = ROOT / "data" / "prices_close.csv"
    if cache.exists():
        px = pd.read_csv(cache, index_col=0, parse_dates=True)["BIL"].dropna()
        r = px.pct_change().dropna()
    else:
        r = DL.load_daily_riskfree("2015-01-01")
    r.index = pd.to_datetime(r.index).normalize()
    return r


def benchmarks(data: pd.DataFrame, rf: pd.Series, a: str, b: str, eq0: float) -> dict:
    mid = ((data["bid_close"] + data["ask_close"]) / 2)
    m = mid[(mid.index >= pd.Timestamp(a, tz="UTC")) & (mid.index <= pd.Timestamp(b, tz="UTC"))]
    long_eq = eq0 * m / m.iloc[0]                      # 1x notional, long EUR vs USD
    short_eq = eq0 * (2 - m / m.iloc[0])               # 1x notional, short (no carry, no costs)
    days = (m.index[-1] - m.index[0]).days
    rfs = rf[(rf.index >= pd.Timestamp(a)) & (rf.index <= pd.Timestamp(b))]
    return {"EURUSD long 1x (no carry/costs)": P.compute(long_eq, None, rf),
            "EURUSD short 1x (no carry/costs)": P.compute(short_eq, None, rf),
            "Cash (T-bills)": {"Total return": float((1 + rfs).prod() - 1), "Years": round(days / 365.25, 2)}}


def lookahead_test(strategy, data: pd.DataFrame, cutoff: str) -> float:
    """Signals computed on truncated history must equal signals computed on full history."""
    full = strategy.signals(data)
    trunc = strategy.signals(data[data.index < pd.Timestamp(cutoff, tz="UTC")])
    a = full.loc[trunc.index, ["entry"]].astype(float)
    return float((a["entry"] - trunc["entry"].astype(float)).abs().sum())


def run_experiment(strategy_id: str, symbol: str = "EURUSD", params: dict | None = None, run_oos: bool = False):
    split, risk, cost, crit = SplitConfig(), RiskConfig(), CostModel(), AcceptanceCriteria()
    spec = INSTRUMENTS[symbol]
    strat = REGISTRY[strategy_id](params)
    exp_id = f"{strategy_id}_v{strat.version}_{symbol}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}"
    out = RESULTS_DIR / exp_id
    out.mkdir(parents=True, exist_ok=True)
    log.info("experiment %s", exp_id)

    # ---------------- data + validation
    raw = DL.load_dukascopy_h1(symbol, split.warmup_start, split.end)
    data, dq = DV.validate_fx(raw, "Dukascopy (datafeed.dukascopy.com)", symbol, "1H", pip=spec.pip)
    try:
        yd = DL.load_yahoo("EURUSD=X", "1d", start=split.warmup_start)
        dq["cross_check_vs_yahoo_daily"] = DV.cross_check_daily(data, yd) if len(yd) else "yahoo unavailable"
    except Exception as e:
        dq["cross_check_vs_yahoo_daily"] = f"failed: {e}"
    (out / "data_quality.json").write_text(json.dumps(ET._clean(dq), indent=1))
    log.info("data: %s rows %s -> %s, missing session %.2f%%", dq["clean_rows"], dq["start"], dq["end"], dq["missing_session_pct"])

    rf = load_rf()
    sig = strat.signals(data)
    la = lookahead_test(strat, data, split.val_start)
    log.info("look-ahead test diff = %s", la)

    segs = {"in_sample": (split.is_start, pd.Timestamp(split.val_start) - pd.Timedelta(seconds=1)),
            "validation": (split.val_start, pd.Timestamp(split.oos_start) - pd.Timedelta(seconds=1))}
    results, runs = {}, {}
    for name, (a, b) in segs.items():
        r = run_backtest(data, sig, spec, risk, cost, a, b)
        runs[name] = r
        results[name] = P.compute(r.equity, r.trades, rf)
        r.trades.to_csv(out / f"trades_{name}.csv", index=False)
        r.equity.to_csv(out / f"equity_{name}.csv")

    # ---------------- robustness on DEVELOPMENT data only
    dev_a, dev_b = split.is_start, segs["validation"][1]
    dev = run_backtest(data, sig, spec, risk, cost, dev_a, dev_b)
    wf = WF.rolling_windows(dev.trades, dev.equity, "6MS")
    wf_pos = float((wf["return"] > 0).mean()) if len(wf) else np.nan
    stress_rows = []
    scenarios = {"base (data spread, $3.5/lot/side, 0.1 pip slip, $6 swap)": (cost, 1.0),
                 "zero costs (gross edge)": (CostModel(commission_per_lot_side=0, slippage_pips=0, swap_per_lot_night=0), 1.0),
                 "spread x2": (cost, 2.0),
                 "spread x2 + slippage 0.5 pip": (CostModel(slippage_pips=0.5), 2.0),
                 "spread x3 + slip 1 pip + $5 comm": (CostModel(slippage_pips=1.0, commission_per_lot_side=5.0), 3.0),
                 "retail spread: +0.8 pip extra, no commission": (CostModel(extra_spread_pips=0.8, commission_per_lot_side=0), 1.0)}
    for nm, (c, mult) in scenarios.items():
        rr = run_backtest(data, sig, spec, risk, c, dev_a, dev_b, spread_multiplier=mult)
        m = P.compute(rr.equity, rr.trades, rf)
        stress_rows.append({"scenario": nm, "return": m["Total return"], "sharpe": m["Sharpe (excess)"],
                            "profit_factor": m.get("Profit factor"), "expectancy_R": m.get("Expectancy (R)"),
                            "max_dd": m["Max drawdown"]})
    stress = pd.DataFrame(stress_rows)
    sens = OPT.sensitivity(type(strat), strat.params, {"sl_atr": [1.0, 1.5, 2.0], "rr": [1.5, 2.0, 3.0],
                                                       "adx_min": [15, 20, 25]},
                           data, spec, risk, cost, split.is_start, segs["in_sample"][1], rf)
    mc = MC.bootstrap_trades(dev.trades["R"], risk.risk_per_trade) if len(dev.trades) else {}

    # ---------------- decision
    checks_val = P.acceptance_check(results["validation"], crit, wf_pos)
    gate = all(checks_val.values()) and results["in_sample"].get("Expectancy (R)", -1) > 0
    if run_oos and gate:
        a, b = split.oos_start, split.end
        r = run_backtest(data, sig, spec, risk, cost, a, b)
        runs["out_of_sample"] = r
        results["out_of_sample"] = P.compute(r.equity, r.trades, rf)
        r.trades.to_csv(out / "trades_out_of_sample.csv", index=False)
        checks_oos = P.acceptance_check(results["out_of_sample"], crit, wf_pos)
        decision = "PROMISING" if all(checks_oos.values()) else "NEEDS IMPROVEMENT"
    elif run_oos and not gate:
        decision = "REJECTED (failed development gate; OOS kept locked)"
    else:
        e_is, e_val = results["in_sample"].get("Expectancy (R)", -1), results["validation"].get("Expectancy (R)", -1)
        if gate:
            decision = "PASSED DEV GATE - run with --run-oos"
        elif e_is > 0 and e_val > 0:
            decision = "NEEDS IMPROVEMENT"
        else:
            decision = "REJECTED"
    bench = {n: benchmarks(data, rf, a, b, risk.initial_equity) for n, (a, b) in segs.items()}

    # ---------------- exports
    summary = pd.DataFrame({k: pd.Series(v) for k, v in results.items()})
    with pd.ExcelWriter(out / "report.xlsx") as xw:
        summary.to_excel(xw, sheet_name="summary")
        pd.DataFrame({"check": list(checks_val), "validation_pass": list(checks_val.values())}).to_excel(xw, sheet_name="acceptance", index=False)
        for n, r in runs.items():
            r.trades.assign(**{c: r.trades[c].dt.tz_convert(None) for c in ("signal_time", "entry_time", "exit_time")}).to_excel(xw, sheet_name=f"trades_{n}"[:31], index=False)
        wf.to_excel(xw, sheet_name="walk_forward_6m", index=False)
        stress.to_excel(xw, sheet_name="cost_stress", index=False)
        sens.to_excel(xw, sheet_name="sensitivity_IS", index=False)
        pd.DataFrame([ET._clean(dq)]).T.to_excel(xw, sheet_name="data_quality")
        pd.DataFrame({k: pd.Series(v) for k, v in bench["validation"].items()}).to_excel(xw, sheet_name="benchmarks_val")
    V.equity_chart({("In-sample" if n == "in_sample" else "Validation" if n == "validation" else "Out-of-sample"): r.equity
                    for n, r in runs.items()}, out / "equity.png",
                   f"{strat.strategy_id} {strat.name} — {symbol} 1H, net of costs")
    if len(dev.trades):
        V.r_histogram(dev.trades, out / "r_distribution.png", f"{strat.strategy_id}: trade outcomes, IS + validation")

    record = {"experiment_id": exp_id, "strategy_id": strat.strategy_id, "strategy_name": strat.name,
              "version": strat.version, "hypothesis": strat.hypothesis, "asset": symbol, "timeframe": "1H (4H confirm)",
              "data_source": dq["source"], "historical_period": {"warmup": split.warmup_start, **asdict(split)},
              "params": strat.params, "risk_settings": asdict(risk), "transaction_costs": asdict(cost),
              "acceptance_criteria": crit.as_dict(), "lookahead_test_diff": la,
              "results": results, "benchmarks": bench, "walk_forward_positive_share": wf_pos,
              "walk_forward_windows": wf.to_dict("records"), "cost_stress": stress.to_dict("records"),
              "monte_carlo": mc, "acceptance_validation": checks_val, "dev_gate_passed": gate,
              "final_decision": decision, "data_quality": dq}
    ET.log_experiment(record)
    (out / "experiment.json").write_text(json.dumps(ET._clean(record), indent=1, default=str))
    return record, summary, stress, sens, wf, mc, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="S001")
    ap.add_argument("--symbol", default="EURUSD")
    ap.add_argument("--run-oos", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    rec, summary, stress, sens, wf, mc, out = run_experiment(a.strategy, a.symbol, run_oos=a.run_oos)
    pd.set_option("display.width", 200)
    print("\n=== SUMMARY (net of costs) ===\n", summary.to_string())
    print("\n=== ACCEPTANCE (validation) ===\n", json.dumps(ET._clean(rec["acceptance_validation"]), indent=1))
    print("\n=== COST STRESS (IS+validation) ===\n", stress.round(3).to_string(index=False))
    print("\n=== SENSITIVITY (in-sample only) ===\n", sens.round(3).to_string(index=False))
    print("\n=== 6-MONTH WINDOWS (IS+validation) ===\n", wf.round(3).to_string(index=False))
    print("\n=== MONTE CARLO ===\n", json.dumps(mc, indent=1))
    print("\nDECISION:", rec["final_decision"], "\nOutputs:", out)


if __name__ == "__main__":
    main()
