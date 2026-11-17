"""Golden cross - IN-SAMPLE ONLY (data truncated at 2022-12-31)."""
import sys; sys.path.insert(0, ".")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from trading_system import data, engine, strategies as S, portfolio as P, metrics as M

END = "2022-12-31"; IS = ("2013-01-01", END)
d = data.load()
o, h, l, c = P.prepare(d, END)
rcc = c.pct_change(fill_method=None); cash = rcc["BIL"].fillna(0.0)
A, CR = P.CONFIG["trend_assets"], P.CONFIG["crypto_assets"]
per = lambda s: s.loc[IS[0]:IS[1]].dropna()

def gc_spy(fast=50, slow=200):
    w = S.golden_cross_score(c[["SPY"]], fast, slow).fillna(0)
    return engine.run(w, rcc, cash, lag=1, cost_bps=5)

res = {}
b1 = gc_spy(); res["GC1 SPY 50/200"] = b1
w2 = S.multi_asset_trend(c, A, 0.10, freq="M", signal="golden_cross")
res["GC2 13-ETF 50/200"] = engine.run(w2, rcc, cash, lag=1, cost_bps=5)
w3 = S.crypto_trend(c[CR], CR, 0.30, signal="golden_cross")
b3 = engine.run(w3, rcc[CR], cash, lag=1, cost_bps=15)
first = w3.abs().sum(axis=1).gt(0).idxmax(); b3 = b3.assign(ret=b3.ret.where(b3.index >= first))
res["GC3 Crypto 50/200"] = b3
port, sl, alloc, det = P.run_portfolio(d, END, P.GC_CONFIG)
res["GC-Portfolio"] = port
res["Original Portfolio (TSMOM)"] = P.run_portfolio(d, END, P.CONFIG)[0]
res["SPY buy&hold"] = pd.DataFrame({"ret": rcc["SPY"], "turnover": 0.0})

tbl = {k: M.stats(per(v["ret"]), cash, v["turnover"]) for k, v in res.items()}
print(M.table(tbl).to_string())
print("\nGC1 SPY: time invested %.0f%%, number of crosses 2013-2022: %d" % (
    S.golden_cross_score(c[["SPY"]]).loc[IS[0]:IS[1]].mean().iloc[0] * 100,
    int(S.golden_cross_score(c[["SPY"]]).loc[IS[0]:IS[1]].diff().abs().sum().iloc[0])))
print("GC-Portfolio avg alloc:", alloc.loc[IS[0]:IS[1]].mean().round(2).to_dict())
print("\nYearly (%):"); print((pd.DataFrame({k: M.yearly(per(v["ret"])) for k, v in res.items()}) * 100).round(1).to_string())

print("\n--- Robustness: Sharpe (excess) 2013-2022 for neighbouring SMA pairs")
rows = []
for f, s_ in [(20, 100), (50, 150), (50, 200), (100, 200), (100, 300)]:
    r1 = gc_spy(f, s_).ret
    w = S.multi_asset_trend(c, A, 0.10, freq="M", signal="golden_cross", fast=f, slow=s_)
    r2 = engine.run(w, rcc, cash, lag=1, cost_bps=5).ret
    w = S.crypto_trend(c[CR], CR, 0.30, signal="golden_cross", fast=f, slow=s_)
    r3 = engine.run(w, rcc[CR], cash, lag=1, cost_bps=15).ret.loc["2016":]
    shp = lambda r: M.stats(per(r), cash)["Sharpe (excess)"]
    rows.append({"SMA": f"{f}/{s_}", "GC1 SPY": shp(r1), "GC2 ETFs": shp(r2), "GC3 crypto (2016+)": shp(r3)})
print(pd.DataFrame(rows).set_index("SMA").round(2).to_string())
print("\nGC1 SPY 2008-2012 Sharpe:", round(M.stats(gc_spy().ret.loc["2008":"2012"], cash)["Sharpe (excess)"], 2),
      " vs SPY:", round(M.stats(rcc.SPY.loc["2008":"2012"], cash)["Sharpe (excess)"], 2))
