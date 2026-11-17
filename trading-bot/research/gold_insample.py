"""Gold long/short - IN-SAMPLE ONLY (data truncated at 2022-12-31)."""
import sys; sys.path.insert(0, ".")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from trading_system import data, engine, gold as G, metrics as M, portfolio as P

END = "2022-12-31"; IS = ("2013-01-01", END)
d = data.load()
res, W, alloc, cash, rcc = G.run_gold(d, END)
per = lambda s: s.loc[IS[0]:IS[1]].dropna()
tbl = {k: M.stats(per(v["ret"]), cash, v["turnover"]) for k, v in res.items()}
tbl["GLD buy&hold"] = M.stats(per(rcc["GLD"]), cash)
print(M.table(tbl).to_string())
print("\nYearly (%):")
yr = pd.DataFrame({k: M.yearly(per(v["ret"])) for k, v in res.items()}); yr["GLD"] = M.yearly(per(rcc["GLD"]))
print((yr * 100).round(1).to_string())
print("\nTime long / short / flat (2013-2022):")
for k, w in W.items():
    w = per(w); print(f"  {k:12s} long {np.mean(w>0):.0%}  short {np.mean(w<0):.0%}  flat {np.mean(w==0):.0%}")
print("Combo avg alloc:", per(alloc).mean().round(2).to_dict())

print("\n--- Long vs short leg contribution (CAGR of each leg alone, IS)")
o, h, l, c = P.prepare(d, END); roo = o.pct_change(fill_method=None)
cfg = G.GOLD_CONFIG; kw = dict(cost_bps=5, short_fee_yr=0.005, short_rebate=False)
for k, w in W.items():
    r = roo if k == "IBS" else rcc
    for side, ww in (("long", w.clip(lower=0)), ("short", w.clip(upper=0))):
        b = engine.run(ww.to_frame("GLD"), r[["GLD"]], cash, lag=1, **kw)["ret"]
        s_ = M.stats(per(b), cash); print(f"  {k:12s} {side:5s}: CAGR {s_['CAGR']:6.1%}  Sharpe {s_['Sharpe (excess)']:.2f}")

print("\n--- Robustness grid (Sharpe excess, IS)")
sh = lambda b: M.stats(per(b["ret"]), cash)["Sharpe (excess)"]
for f in ("W-FRI", "M"):
    w = G.tsmom_ls(c["GLD"], 0.10, 1.0, f)
    print(f"  TSMOM {f:6s}: {sh(engine.run(w.to_frame('GLD'), rcc[['GLD']], cash, 1, **kw)):.2f}")
for fa, sl in ((20, 100), (50, 150), (50, 200), (100, 200), (100, 300)):
    w = G.golden_cross_ls(c["GLD"], fa, sl)
    print(f"  GC {fa}/{sl}: {sh(engine.run(w.to_frame('GLD'), rcc[['GLD']], cash, 1, **kw)):.2f}")
for lo, hi in ((0.1, 0.9), (0.15, 0.85), (0.2, 0.8), (0.25, 0.75), (0.3, 0.7)):
    for cost in (5, 10):
        w = G.ibs_ls(o["GLD"], h["GLD"], l["GLD"], c["GLD"], lo, hi)
        k2 = dict(kw, cost_bps=cost)
        print(f"  IBS {lo}/{hi} cost {cost}: {sh(engine.run(w.to_frame('GLD'), roo[['GLD']], cash, 1, **k2)):.2f}", end=" |")
    print()
# look-ahead test
full = G.run_gold(d, None)[0]
print("\nLook-ahead test max diff (truncated vs full data):",
      max(float((res[k]["ret"].loc[IS[0]:"2022-12-30"] - full[k]["ret"].loc[IS[0]:"2022-12-30"]).abs().max()) for k in res))
