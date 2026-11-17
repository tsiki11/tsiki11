"""Charts (matplotlib, PNG)."""
from __future__ import annotations

import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e0dfda", "#fcfcfb"
SEG_COL = {"In-sample": "#2a78d6", "Validation": "#eb6834", "Out-of-sample": "#1baf7a"}


def equity_chart(segments: dict, path, title: str, benchmark: pd.Series | None = None):
    """segments: {"In-sample": equity_series, "Validation": equity_series, ...} (each starting at 10k)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1.2]}, facecolor=SURF)
    for name, eq in segments.items():
        e = eq.copy(); e.index = e.index.tz_convert(None)
        col = SEG_COL.get(name, "#6250d6")
        ax1.plot(e.index, e.values, color=col, lw=1.6, label=f"{name}: {e.iloc[-1] / e.iloc[0] - 1:+.1%}")
        ax2.plot(e.index, (e / e.cummax() - 1) * 100, color=col, lw=1.1)
    for ax in (ax1, ax2):
        ax.set_facecolor(SURF); ax.grid(color=GRID, lw=0.6)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax1.axhline(segments[next(iter(segments))].iloc[0], color=MUTED, lw=0.8, ls="--")
    ax1.set_ylabel("Equity (USD), each segment restarts at initial capital", color=MUTED)
    ax2.set_ylabel("Drawdown %", color=MUTED)
    ax1.set_title(title, loc="left", color=INK, fontsize=12)
    ax1.legend(frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def r_histogram(trades: pd.DataFrame, path, title: str):
    fig, ax = plt.subplots(figsize=(8, 4), facecolor=SURF)
    ax.hist(trades["R"].clip(-3, 4), bins=40, color="#2a78d6", edgecolor=SURF, linewidth=1)
    ax.axvline(0, color=MUTED, lw=1)
    ax.axvline(trades["R"].mean(), color="#eb6834", lw=1.5, label=f"mean R = {trades['R'].mean():+.3f}")
    ax.set_facecolor(SURF); ax.grid(color=GRID, lw=0.6, axis="y")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.set_xlabel("Trade result in R (net P&L / planned risk), clipped to [-3, 4]", color=MUTED)
    ax.set_ylabel("Trades", color=MUTED); ax.set_title(title, loc="left", fontsize=11); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
