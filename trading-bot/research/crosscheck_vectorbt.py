"""Cross-check the custom engine against vectorbt (independent open-source library)."""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd, vectorbt as vbt
from trading_system import data, engine, strategies as S

d = data.load(); c = d["Close"].loc[:"2022-12-31"]
c = c.loc[c["SPY"].dropna().index]
A = ["SPY","QQQ","IWM","EFA","EEM","VNQ","TLT","IEF","LQD","TIP","GLD","DBC","UUP"]
w = S.multi_asset_trend(c, A, freq="M")
r = c.pct_change(fill_method=None)
mine = engine.run(w, r, pd.Series(0.0, index=r.index), lag=1, cost_bps=5)["ret"].loc["2013":]

# vectorbt: target-percent orders, executed one bar later at the close, 5 bps fees
px = c[A].loc["2012-06":].ffill()
tgt = w[A].shift(1).loc[px.index]
chg = tgt.diff().abs().sum(axis=1) > 1e-12
tgt = tgt.where(chg, np.nan)                 # only send orders on rebalance days
pf = vbt.Portfolio.from_orders(px, size=tgt, size_type="targetpercent", fees=0.0005,
                               group_by=True, cash_sharing=True, call_seq="auto", freq="1D")
vb = pf.returns().loc["2013":]
print("my engine CAGR %.2f%%  vectorbt CAGR %.2f%%" % (((1+mine).prod()**(252/len(mine))-1)*100,
                                                      ((1+vb).prod()**(252/len(vb))-1)*100))
print("daily return correlation: %.4f" % mine.corr(vb))
