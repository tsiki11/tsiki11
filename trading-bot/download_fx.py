"""Download + cache Dukascopy hourly bid/ask for the configured period."""
import logging, sys, time
from trading_bot import data_loader as DL
from trading_bot.config import SplitConfig
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
s = SplitConfig(); t = time.time()
df = DL.load_dukascopy_h1(sys.argv[1] if len(sys.argv) > 1 else "EURUSD", s.warmup_start, s.end)
print(f"done: {len(df)} rows, {df.index[0]} -> {df.index[-1]} in {time.time()-t:.0f}s")
