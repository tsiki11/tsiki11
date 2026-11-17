"""Run the LOCKED portfolio: in-sample 2013-2022, out-of-sample 2023 -> latest close.

Usage:  python run_backtest.py [--refresh]
Outputs: results/summary.md, results/equity.png, results/tearsheet_*.html
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import logging; logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
from trading_system import data, metrics as M, portfolio as P

OUT = Path("results"); OUT.mkdir(exist_ok=True)
IS = ("2013-01-01", "2022-12-31")


def block_bootstrap_sharpe(ex: pd.Series, n=2000, block=20, seed=0):
    rng = np.random.default_rng(seed)
    x = ex.dropna().values; T = len(x); res = []
    for _ in range(n):
        starts = rng.integers(0, T - block, size=T // block + 1)
        s = np.concatenate([x[i:i + block] for i in starts])[:T]
        res.append(s.mean() / s.std() * np.sqrt(252))
    return np.percentile(res, [2.5, 97.5])


def main():
    d = data.load(refresh="--refresh" in sys.argv)
    port, sleeves, alloc, det = P.run_portfolio(d)
    cash = det["cash"]
    c = d["Close"].loc[port.index]
    r = c.pct_change(fill_method=None)
    last = port.index[-1].strftime("%Y-%m-%d")
    OOS = ("2023-01-01", last)
    series = {
        "Portfolio": port["ret"],
        "SPY buy&hold": r["SPY"],
        "60/40 SPY/IEF": 0.6 * r["SPY"] + 0.4 * r["IEF"],
        "Trend sleeve": sleeves["Trend"],
        "MeanRev sleeve": sleeves["MeanRev"],
        "Crypto sleeve": sleeves["Crypto"],
    }
    lines = [f"# Backtest results (generated {pd.Timestamp.today():%Y-%m-%d}, data to {last})\n"]
    for name, (a, b) in {"IN-SAMPLE 2013-2022": IS, f"OUT-OF-SAMPLE 2023-{last}": OOS,
                         f"FULL 2013-{last}": (IS[0], last)}.items():
        tbl = {k: M.stats(v.loc[a:b].dropna(), cash,
                          port["turnover"] if k == "Portfolio" else None) for k, v in series.items()}
        ex = (port["ret"] - cash).loc[a:b]
        lo, hi = block_bootstrap_sharpe(ex)
        t_stat = ex.mean() / ex.std() * np.sqrt(len(ex))
        txt = M.table(tbl).to_markdown() if hasattr(pd.DataFrame, "to_markdown") else M.table(tbl).to_string()
        lines += [f"\n## {name}\n", txt,
                  f"\nPortfolio Sharpe 95% CI (block bootstrap): [{lo:.2f}, {hi:.2f}]  |  t-stat of mean excess return: {t_stat:.2f}\n"]
        print(f"\n===== {name} =====\n{M.table(tbl).to_string()}\nSharpe 95% CI [{lo:.2f}, {hi:.2f}]  t={t_stat:.2f}")
    yr = pd.DataFrame({k: M.yearly(v.loc[IS[0]:last].dropna()) for k, v in series.items()})
    yr.index = [str(i) + ("*" if i == pd.Timestamp(last).year else "") for i in yr.index]
    lines += ["\n## Calendar-year returns (%)  (* = year to date)\n", (yr * 100).round(1).to_markdown()]
    print("\n", (yr * 100).round(1).to_string())
    corr = pd.DataFrame({k: series[k] for k in ("Trend sleeve", "MeanRev sleeve", "Crypto sleeve", "SPY buy&hold")})
    lines += ["\n## Correlation of daily returns (OOS)\n", corr.loc[OOS[0]:].corr().round(2).to_markdown()]
    lines += ["\n## Average sleeve allocation\n",
              pd.DataFrame({"IS": alloc.loc[IS[0]:IS[1]].mean(), "OOS": alloc.loc[OOS[0]:].mean()}).round(2).to_markdown()]
    (OUT / "summary.md").write_text("\n".join(lines))
    pd.DataFrame(series).loc[IS[0]:].to_csv(OUT / "daily_returns.csv")

    # ---- chart: equity (log) + drawdown, OOS shaded
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"Portfolio": "#2a78d6", "SPY buy&hold": "#eb6834", "60/40 SPY/IEF": "#1baf7a"}
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1.2]}, facecolor="#fcfcfb")
    for k, cl in col.items():
        eq = (1 + series[k].loc[IS[0]:]).cumprod()
        ax1.plot(eq.index, eq.values, color=cl, lw=2 if k == "Portfolio" else 1.4, label=k)
        ax1.annotate(f"{k}  {eq.iloc[-1]:.2f}x", (eq.index[-1], eq.iloc[-1]), xytext=(6, 0),
                     textcoords="offset points", color="#0b0b0b", fontsize=9, va="center")
        dd = eq / eq.cummax() - 1
        ax2.plot(dd.index, dd.values * 100, color=cl, lw=1.4 if k == "Portfolio" else 1)
    for ax in (ax1, ax2):
        ax.axvspan(pd.Timestamp(OOS[0]), port.index[-1], color="#e8e7e2", zorder=0)
        ax.set_facecolor("#fcfcfb"); ax.grid(color="#e0dfda", lw=0.6)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax1.set_yscale("log")
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    ax1.yaxis.set_major_locator(FixedLocator([1, 1.5, 2, 3, 4, 6]))
    ax1.yaxis.set_minor_locator(NullLocator())
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}x"))
    ax1.set_ylabel("Growth of $1 (log)", color="#52514e")
    ax1.set_title("Locked portfolio vs benchmarks, net of costs — shaded = out-of-sample (2023+)",
                  loc="left", color="#0b0b0b", fontsize=12)
    ax1.legend(frameon=False, loc="upper left"); ax2.set_ylabel("Drawdown %", color="#52514e")
    ax1.set_xlim(right=port.index[-1] + pd.Timedelta(days=620))
    fig.tight_layout(); fig.savefig(OUT / "equity.png", dpi=150)

    try:
        import quantstats as qs
        for tag, (a, b) in {"oos": OOS, "full": (IS[0], last)}.items():
            rf_ann = float(cash.loc[a:b].mean() * 252)   # average T-bill yield of the period
            qs.reports.html(port["ret"].loc[a:b], benchmark=r["SPY"].loc[a:b], rf=rf_ann,
                            output=str(OUT / f"tearsheet_{tag}.html"),
                            title=f"Multi-sleeve portfolio ({tag})")
    except Exception as e:
        print("quantstats report skipped:", e)
    print("\nSaved results/ (summary.md, equity.png, daily_returns.csv, tearsheets)")


if __name__ == "__main__":
    main()
