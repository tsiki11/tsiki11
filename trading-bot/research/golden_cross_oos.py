"""Golden cross - LOCKED 50/200 configs, run once on IS / OOS / FULL.
Writes results/golden_cross_summary.md and results/golden_cross_equity.png"""
import sys; sys.path.insert(0, ".")
import warnings; warnings.filterwarnings("ignore")
import logging; logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import numpy as np, pandas as pd
from trading_system import data, engine, strategies as S, portfolio as P, metrics as M
from run_backtest import block_bootstrap_sharpe

d = data.load()
o, h, l, c = P.prepare(d)
rcc = c.pct_change(fill_method=None); cash = rcc["BIL"].fillna(0.0)
A, CR = P.CONFIG["trend_assets"], P.CONFIG["crypto_assets"]
last = c.index[-1].strftime("%Y-%m-%d")
PER = {"IN-SAMPLE 2013-2022": ("2013-01-01", "2022-12-31"),
       f"OUT-OF-SAMPLE 2023-{last}": ("2023-01-01", last),
       f"FULL 2013-{last}": ("2013-01-01", last)}

s = {}
s["GC1 SPY 50/200"] = engine.run(S.golden_cross_score(c[["SPY"]]).fillna(0), rcc, cash, lag=1, cost_bps=5)
s["GC2 13-ETF 50/200"] = engine.run(S.multi_asset_trend(c, A, 0.10, freq="M", signal="golden_cross"), rcc, cash, lag=1, cost_bps=5)
w3 = S.crypto_trend(c[CR], CR, 0.30, signal="golden_cross")
b3 = engine.run(w3, rcc[CR], cash, lag=1, cost_bps=15)
s["GC3 Crypto 50/200"] = b3.assign(ret=b3.ret.where(b3.index >= w3.abs().sum(axis=1).gt(0).idxmax()))
gcp, gsl, galloc, _ = P.run_portfolio(d, None, P.GC_CONFIG)
s["GC-Portfolio"] = gcp
s["Original Portfolio"] = P.run_portfolio(d, None, P.CONFIG)[0]
s["SPY buy&hold"] = pd.DataFrame({"ret": rcc["SPY"], "turnover": 0.0})
s["60/40 SPY/IEF"] = pd.DataFrame({"ret": 0.6 * rcc["SPY"] + 0.4 * rcc["IEF"], "turnover": 0.0})

lines = [f"# Golden cross (50/200) — locked before OOS. Data to {last}\n"]
for name, (a, b) in PER.items():
    tbl = {k: M.stats(v["ret"].loc[a:b].dropna(), cash, v["turnover"]) for k, v in s.items()}
    ci = {k: block_bootstrap_sharpe((s[k]["ret"] - cash).loc[a:b].dropna()) for k in ("GC-Portfolio", "GC3 Crypto 50/200")}
    t = M.table(tbl)
    print(f"\n===== {name} =====\n{t.to_string()}")
    cit = "  |  ".join(f"{k} Sharpe 95% CI [{v[0]:.2f}, {v[1]:.2f}]" for k, v in ci.items())
    print(cit)
    lines += [f"\n## {name}\n", t.to_markdown(), f"\n{cit}\n"]
yr = pd.DataFrame({k: M.yearly(v["ret"].loc["2013":].dropna()) for k, v in s.items()})
print("\n", (yr * 100).round(1).to_string())
lines += ["\n## Calendar-year returns (%)\n", (yr * 100).round(1).to_markdown()]
oos = PER[f"OUT-OF-SAMPLE 2023-{last}"]
cor = gsl.loc[oos[0]:].corr().round(2)
lines += ["\n## GC-Portfolio sleeve correlations (OOS)\n", cor.to_markdown()]
print("\nOOS sleeve corr:\n", cor.to_string())
open("results/golden_cross_summary.md", "w").write("\n".join(lines))

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
col = {"GC-Portfolio": "#2a78d6", "Original Portfolio": "#6250d6", "GC1 SPY 50/200": "#1baf7a", "SPY buy&hold": "#eb6834"}
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1.2]}, facecolor="#fcfcfb")
for k, cl in col.items():
    eq = (1 + s[k]["ret"].loc["2013":].fillna(0)).cumprod()
    ax1.plot(eq.index, eq.values, color=cl, lw=2 if k == "GC-Portfolio" else 1.3, label=k)
    ax1.annotate(f"{k}  {eq.iloc[-1]:.2f}x", (eq.index[-1], eq.iloc[-1]), xytext=(6, 0), textcoords="offset points", fontsize=9, va="center", color="#0b0b0b")
    ax2.plot(eq.index, (eq / eq.cummax() - 1) * 100, color=cl, lw=1.3 if k == "GC-Portfolio" else 0.9)
for ax in (ax1, ax2):
    ax.axvspan(pd.Timestamp("2023-01-01"), c.index[-1], color="#e8e7e2", zorder=0)
    ax.set_facecolor("#fcfcfb"); ax.grid(color="#e0dfda", lw=0.6)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
ax1.set_yscale("log"); ax1.yaxis.set_major_locator(FixedLocator([1, 1.5, 2, 3, 4, 6])); ax1.yaxis.set_minor_locator(NullLocator())
ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}x"))
ax1.set_ylabel("Growth of $1 (log)", color="#52514e"); ax2.set_ylabel("Drawdown %", color="#52514e")
ax1.set_title("Golden cross 50/200 vs original system, net of costs — shaded = out-of-sample (2023+)", loc="left", fontsize=12)
ax1.legend(frameon=False, loc="upper left"); ax1.set_xlim(right=c.index[-1] + pd.Timedelta(days=750))
fig.tight_layout(); fig.savefig("results/golden_cross_equity.png", dpi=150)
print("saved results/golden_cross_summary.md, results/golden_cross_equity.png")
