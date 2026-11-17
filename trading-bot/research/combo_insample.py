import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from trading_system import data, portfolio as P, metrics as M
d = data.load()
END = "2022-12-31"
for lev in (1.0, 1.5):
    cfg = dict(P.CONFIG, max_leverage=lev)
    port, sl, alloc, det = P.run_portfolio(d, END, cfg)
    cash = det["cash"]
    r = port["ret"].loc["2013":END]
    res = {f"Portfolio L<={lev}": M.stats(r, cash, port["turnover"])}
    for k in sl: res[k] = M.stats(sl[k].loc["2013":END].dropna(), cash)
    print(M.table(res).to_string())
    print("avg alloc:", alloc.loc["2013":END].mean().round(2).to_dict(), "avg lev:", alloc.loc["2013":END].sum(axis=1).mean().round(2))
    print((M.yearly(r)*100).round(1).to_dict())
