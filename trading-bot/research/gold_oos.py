"""Gold long/short - LOCKED config, all periods. Writes results/gold_summary.md + gold_equity.png"""
import sys; sys.path.insert(0, ".")
import warnings; warnings.filterwarnings("ignore")
import logging; logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import numpy as np, pandas as pd
from trading_system import data, gold as G, metrics as M
from run_backtest import block_bootstrap_sharpe

d = data.load()
res, W, alloc, cash, rcc = G.run_gold(d, None)
last = rcc.index[-1].strftime("%Y-%m-%d")
PER = {"PRE-IS 2006-2012 (gold bull, extra pre-2023 check)": ("2006-01-01", "2012-12-31"),
       "IN-SAMPLE 2013-2022": ("2013-01-01", "2022-12-31"),
       f"OUT-OF-SAMPLE 2023-{last}": ("2023-01-01", last)}
series = {k: v["ret"] for k, v in res.items()}; series["GLD buy&hold"] = rcc["GLD"]
turn = {k: v["turnover"] for k, v in res.items()}
lines = [f"# Gold (GLD) long/short — locked config. Data to {last}\n",
         "Costs: 5 bps/trade, short borrow 0.5%/yr, no interest on short proceeds.\n"]
for name, (a, b) in PER.items():
    tbl = {k: M.stats(v.loc[a:b].dropna(), cash, turn.get(k)) for k, v in series.items()}
    t = M.table(tbl); ci = block_bootstrap_sharpe((series["Combo"] - cash).loc[a:b].dropna())
    print(f"\n===== {name} =====\n{t.to_string()}\nCombo Sharpe 95% CI [{ci[0]:.2f}, {ci[1]:.2f}]")
    lines += [f"\n## {name}\n", t.to_markdown(), f"\nCombo Sharpe 95% CI [{ci[0]:.2f}, {ci[1]:.2f}]\n"]
yr = pd.DataFrame({k: M.yearly(v.loc["2006":].dropna()) for k, v in series.items()})
print("\n", (yr * 100).round(1).to_string())
lines += ["\n## Calendar-year returns (%)\n", (yr * 100).round(1).to_markdown()]
oos = PER[f"OUT-OF-SAMPLE 2023-{last}"]
pos = pd.DataFrame({k: [np.mean(w.loc[oos[0]:] > 0), np.mean(w.loc[oos[0]:] < 0)] for k, w in W.items()}, index=["% long", "% short"])
print("\nOOS positioning:\n", (pos * 100).round(0).to_string())
lines += ["\n## OOS positioning (% of days)\n", (pos * 100).round(0).to_markdown()]
open("results/gold_summary.md", "w").write("\n".join(lines))

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
col = {"Combo": "#2a78d6", "TSMOM": "#6250d6", "GoldenCross": "#1baf7a", "IBS": "#e87ba4", "GLD buy&hold": "#eda100"}
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1.2]}, facecolor="#fcfcfb")
for k, cl in col.items():
    eq = (1 + series[k].loc["2013":].fillna(0)).cumprod()
    ax1.plot(eq.index, eq.values, color=cl, lw=2 if k == "Combo" else 1.3, label=k)
    ax1.annotate(f"{k}  {eq.iloc[-1]:.2f}x", (eq.index[-1], eq.iloc[-1]), xytext=(6, 0), textcoords="offset points", fontsize=9, va="center", color="#0b0b0b")
    ax2.plot(eq.index, (eq / eq.cummax() - 1) * 100, color=cl, lw=1.3 if k == "Combo" else 0.9)
for ax in (ax1, ax2):
    ax.axvspan(pd.Timestamp("2023-01-01"), rcc.index[-1], color="#e8e7e2", zorder=0)
    ax.set_facecolor("#fcfcfb"); ax.grid(color="#e0dfda", lw=0.6)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
from matplotlib.ticker import FixedLocator, NullLocator
ax1.set_yscale("log"); ax1.yaxis.set_major_locator(FixedLocator([0.5, 0.6, 0.8, 1, 1.25, 1.5, 2, 3]))
ax1.yaxis.set_minor_locator(NullLocator()); ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}x"))
ax1.set_ylabel("Growth of $1 (log)", color="#52514e"); ax2.set_ylabel("Drawdown %", color="#52514e")
ax1.set_title("Gold (GLD) long/short strategies, net of costs — shaded = out-of-sample (2023+)", loc="left", fontsize=12)
ax1.legend(frameon=False, loc="upper left"); ax1.set_xlim(right=rcc.index[-1] + pd.Timedelta(days=750))
fig.tight_layout(); fig.savefig("results/gold_equity.png", dpi=150)
print("saved results/gold_summary.md, results/gold_equity.png")
