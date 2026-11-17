"""Robustness checks on in-sample data only (<= 2022)."""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from trading_system import data, engine, strategies as S, metrics as M

d = data.load(); IS_END = "2022-12-31"
o, h, l, c = (d[k].loc[:IS_END] for k in ("Open", "High", "Low", "Close"))
days = c["SPY"].dropna().index
o, h, l, c = (x.loc[days] for x in (o, h, l, c))
rcc, roo = c.pct_change(fill_method=None), o.pct_change(fill_method=None)
cash = rcc["BIL"].fillna(0.0)
EQ = ["SPY", "QQQ", "IWM"]
def sh(r, a="2013", b=IS_END): r = r.loc[a:b]; ex = r - cash.loc[r.index]; return ex.mean()/ex.std()*np.sqrt(252)

print("--- B: IBS parameter / cost grid (Sharpe 2013-2022, next-open fills)")
rows = []
for entry in (0.1, 0.15, 0.2, 0.25, 0.3):
    for sma in (0, 100, 200):
        pos = S.ibs_mean_reversion(o, h, l, c, EQ, entry=entry, sma=sma)
        for cost in (5, 10):
            b = engine.run(pos, roo, cash, lag=1, cost_bps=cost)
            rows.append((entry, sma, cost, round(sh(b.ret), 2)))
g = pd.DataFrame(rows, columns=["entry", "sma", "cost", "sharpe"])
print(g.pivot_table(index=["entry"], columns=["sma", "cost"], values="sharpe").to_string())

print("--- B per ETF (entry .2, sma200)")
for a in EQ:
    pos = S.ibs_mean_reversion(o, h, l, c, [a])
    print(a, round(sh(engine.run(pos, roo, cash, lag=1, cost_bps=5).ret), 2))
print("--- B on non-traded ETFs as out-of-universe check: EFA EEM")
for a in ["EFA", "EEM"]:
    pos = S.ibs_mean_reversion(o, h, l, c, [a])
    print(a, round(sh(engine.run(pos, roo, cash, lag=1, cost_bps=5).ret), 2))

A = ["SPY","QQQ","IWM","EFA","EEM","VNQ","TLT","IEF","LQD","TIP","GLD","DBC","UUP"]
print("--- A: trend, sub-periods & variants")
for f in ("W-FRI", "M"):
    w = S.multi_asset_trend(c, A, freq=f)
    b = engine.run(w, rcc, cash, lag=1, cost_bps=5)
    print(f, "2008-2012:", round(sh(b.ret, "2008", "2012"), 2), " 2013-2022:", round(sh(b.ret), 2),
          " 2008 yr:", round(M.yearly(b.ret.loc["2008"]).iloc[0]*100, 1), "%")
print("SPY 2008-2012 sharpe:", round(sh(rcc.SPY, "2008", "2012"), 2))

print("--- C: crypto, cost & sub-period")
cr = ["BTC-USD", "ETH-USD"]
w = S.crypto_trend(c[cr], cr)
for cost in (15, 30, 50):
    b = engine.run(w, rcc[cr], cash, lag=1, cost_bps=cost)
    print("cost", cost, "2016-2019:", round(sh(b.ret, "2016", "2019"), 2), "2020-2022:", round(sh(b.ret, "2020", "2022"), 2))
wb = S.crypto_trend(c[["BTC-USD"]], ["BTC-USD"])
print("BTC only 2016-2022:", round(sh(engine.run(wb, rcc[["BTC-USD"]], cash, lag=1, cost_bps=15).ret, "2016"), 2))
