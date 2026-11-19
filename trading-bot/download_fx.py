"""Download + cache hourly bid/ask for the configured period.

    python download_fx.py EURUSD            # FXCM archive (primary)
    python download_fx.py EURUSD dukascopy  # Dukascopy (throttles/blocks heavy use)
"""
import logging, sys, time
from trading_bot import data_loader as DL
from trading_bot.config import SplitConfig
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
sym = sys.argv[1] if len(sys.argv) > 1 else "EURUSD"
src = sys.argv[2] if len(sys.argv) > 2 else "fxcm"
loader = DL.load_fxcm_h1 if src == "fxcm" else DL.load_dukascopy_h1
s = SplitConfig(); t = time.time()
df = loader(sym, s.warmup_start, s.end)
print(f"done: {len(df)} rows, {df.index[0]} -> {df.index[-1]} in {time.time()-t:.0f}s")
