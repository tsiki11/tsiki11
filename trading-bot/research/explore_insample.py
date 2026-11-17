"""In-sample exploration (2013-2022 ONLY). Out-of-sample data is not touched."""
import sys; sys.path.insert(0, ".")
import pandas as pd
from trading_system import data, engine, strategies as S, metrics as M

IS_START, IS_END = "2013-01-01", "2022-12-31"
d = data.load()
o, h, l, c = d["Open"], d["High"], d["Low"], d["Close"]
cut = lambda x: x.loc[:IS_END]          # never let the strategy see post-2022 data
o, h, l, c = map(cut, (o, h, l, c))

etf_days = c["SPY"].dropna().index
c_etf = c.loc[etf_days]
rets_cc = c_etf.pct_change(fill_method=None)
cash = rets_cc["BIL"].fillna(0.0)
period = lambda s: s.loc[IS_START:IS_END]

ASSETS_A = ["SPY", "QQQ", "IWM", "EFA", "EEM", "VNQ", "TLT", "IEF", "LQD", "TIP", "GLD", "DBC", "UUP"]
res = {}

# A: trend
wA = S.multi_asset_trend(c_etf, ASSETS_A, target_vol=0.10)
bA = engine.run(wA, rets_cc, cash, lag=1, cost_bps=5)
res["A Trend"] = bA

# B: IBS mean reversion, next-open execution (realistic) -> open-to-open returns
EQ = ["SPY", "QQQ", "IWM"]
oe, he, le, ce = (x.loc[etf_days] for x in (o, h, l, c))
posB = S.ibs_mean_reversion(oe, he, le, ce, EQ)
rets_oo = oe.pct_change(fill_method=None)
# decided at close t, filled at open t+1 (lag=1 on open-to-open returns)
bB = engine.run(posB, rets_oo, cash, lag=1, cost_bps=5)
res["B IBS (next open)"] = bB
# Same, filled at the close of signal day (MOC) -- optimistic variant for reference
bB2 = engine.run(posB, rets_cc, cash, lag=0, cost_bps=5)  # lag=0 => fill at close t (MOC)
res["B IBS (close fill)"] = bB2

# C: crypto trend
c_cr = c.loc[etf_days, ["BTC-USD", "ETH-USD"]]
r_cr = c_cr.pct_change(fill_method=None)
wC = S.crypto_trend(c_cr, ["BTC-USD", "ETH-USD"], target_vol=0.30)
bC = engine.run(wC, r_cr, cash, lag=1, cost_bps=15)
res["C Crypto trend"] = bC

# Benchmarks
res["SPY buy&hold"] = pd.DataFrame({"ret": rets_cc["SPY"], "turnover": 0.0})
res["60/40 SPY/IEF"] = pd.DataFrame({"ret": 0.6 * rets_cc["SPY"] + 0.4 * rets_cc["IEF"], "turnover": 0.0})

rf = period(cash)
tbl = {k: M.stats(period(v["ret"]), rf, v["turnover"]) for k, v in res.items()}
print(M.table(tbl).to_string())
yr = pd.DataFrame({k: M.yearly(period(v["ret"])) for k, v in res.items()})
print((yr * 100).round(1).to_string())
rets_df = pd.DataFrame({k: period(v["ret"]) for k, v in res.items()})
print(rets_df.corr().round(2).to_string())
print("Exposure B:", (period(posB).sum(axis=1) > 0).mean().round(2))
