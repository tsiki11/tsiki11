"""Daily runner: compute today's target portfolio and (optionally) send orders.

Run once per trading day AFTER the US close (e.g. 22:30 Athens time):
    python live_trader.py --capital 10000            # print targets/orders only
    python live_trader.py --capital 10000 --alpaca   # send to Alpaca PAPER account

Alpaca paper trading is free: create keys at https://app.alpaca.markets (Paper)
and export ALPACA_API_KEY / ALPACA_SECRET_KEY. Orders are placed as
market-on-open (ETFs, time_in_force="opg") and GTC market (crypto).

Safety rails: paper endpoint by default, max order notional, kill-switch file
(create a file named KILL to stop all trading), dry run unless --alpaca.
STATUS: the Alpaca part has NOT been tested against a live account yet.
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.request
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
from trading_system import data, portfolio as P

PAPER_URL = "https://paper-api.alpaca.markets"
ALPACA_SYMBOL = {"BTC-USD": "BTC/USD", "ETH-USD": "ETH/USD"}


def target_weights(d: dict, cfg=P.CONFIG) -> tuple[pd.Series, pd.Series, pd.Timestamp]:
    """Combined per-instrument target weight as of the latest close."""
    sleeves, det = P.sleeve_returns(d, None, cfg)
    alloc = P.allocation(sleeves, det["cash"], cfg).iloc[-1]
    t = sleeves.index[-1]
    combined = pd.Series(dtype=float)
    for name in ("Trend", "MeanRev", "Crypto"):
        w = det[name][0].iloc[-1]
        combined = combined.add(alloc[name] * w, fill_value=0.0)
    return combined[combined.abs() > 1e-4].sort_values(ascending=False), alloc, t


def alpaca(method: str, path: str, body: dict | None = None):
    req = urllib.request.Request(
        PAPER_URL + path, method=method,
        data=json.dumps(body).encode() if body else None,
        headers={"APCA-API-KEY-ID": os.environ["ALPACA_API_KEY"],
                 "APCA-API-SECRET-KEY": os.environ["ALPACA_SECRET_KEY"],
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read() or b"null")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capital", type=float, default=None, help="override equity (dry run)")
    ap.add_argument("--alpaca", action="store_true")
    ap.add_argument("--max-order", type=float, default=0.35, help="max order as fraction of equity")
    ap.add_argument("--min-trade", type=float, default=0.01, help="ignore drifts below this weight")
    a = ap.parse_args()
    if Path("KILL").exists():
        print("KILL file present -> no trading."); return

    d = data.load(refresh=True)
    w, alloc, t = target_weights(d)
    px = d["Close"].ffill().iloc[-1]
    print(f"Signals as of close {t:%Y-%m-%d}")
    print("Sleeve allocation:", alloc.round(3).to_dict())
    print("Target weights:\n", (w * 100).round(2).astype(str).add(" %").to_string())

    equity, current = a.capital or 10000.0, {}
    if a.alpaca:
        acct = alpaca("GET", "/v2/account")
        equity = float(acct["equity"])
        for p in alpaca("GET", "/v2/positions"):
            current[p["symbol"]] = float(p["market_value"])
    orders = []
    for sym in sorted(set(w.index) | {k for k in current}):
        broker_sym = ALPACA_SYMBOL.get(sym, sym)
        tgt_val = float(w.get(sym, 0.0)) * equity
        cur_val = current.get(broker_sym.replace("/", ""), current.get(broker_sym, 0.0))
        delta = tgt_val - cur_val
        if abs(delta) < a.min_trade * equity:
            continue
        delta = max(-a.max_order * equity, min(a.max_order * equity, delta))
        orders.append((sym, broker_sym, round(delta, 2), px.get(sym, float("nan"))))
    print(f"\nEquity ${equity:,.0f} -> orders:")
    for sym, bsym, notional, p in orders:
        print(f"  {'BUY ' if notional > 0 else 'SELL'} {bsym:8s} ${abs(notional):>10,.2f}  (~{abs(notional)/p:.4f} units @ {p:,.2f})")
    if not a.alpaca:
        print("\nDry run (use --alpaca to send to the paper account)."); return
    for sym, bsym, notional, p in orders:
        crypto = sym in ALPACA_SYMBOL
        body = {"symbol": bsym, "side": "buy" if notional > 0 else "sell", "type": "market",
                "time_in_force": "gtc" if crypto else "opg"}
        if crypto:
            body["notional"] = str(abs(notional))
        else:  # market-on-open (opg) orders accept whole shares only
            qty = int(abs(notional) // p)
            if qty == 0:
                continue
            body["qty"] = str(qty)
        print("sending", body)
        print(" ->", alpaca("POST", "/v2/orders", body).get("status"))


if __name__ == "__main__":
    main()
