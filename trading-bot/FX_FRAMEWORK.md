# FX research framework: Experiment 1 (EUR/USD baseline)

Modular, event-driven backtesting framework για FX και μέταλλα, που χρησιμοποιεί μόνο δωρεάν δεδομένα και open-source εργαλεία.
Κώδικας στο `trading_bot/`, tests στο `tests/` και αποτελέσματα στο `results/fx/`.

## Φάση 1–2: Πηγές δεδομένων (ελέγχθηκαν στην πράξη)

| Πηγή | Τι δίνει | Ιστορικό | Αποτέλεσμα ελέγχου | Επιλογή |
|---|---|---|---|---|
| **Dukascopy** datafeed | Ωριαία **bid και ask** OHLC σε UTC, ένα αρχείο ανά μήνα | από 2003 | ✅ Λειτουργεί. Κόβει τις γρήγορες σειριακές λήψεις (503 / connection reset), οπότε χρειάζεται περίπου 1 λήψη/δευτ. με backoff | **Κύρια** |
| HistData | 1-min OHLC (bid) σε EST χωρίς DST | από 2000 | ❌ Η λήψη χρειάζεται token που δημιουργεί JavaScript (anti-bot). Δεν έγινε παράκαμψη | Απορρίφθηκε |
| Yahoo `EURUSD=X` | Ημερήσια (πολλά χρόνια), 1h (μόνο ~730 μέρες) | 1h: 2 χρόνια | ⚠️ Δεν αρκεί για 1H backtest 5–10 ετών και υπόκειται σε rate limit (429) | Διασταύρωση |

**Περιορισμοί του Dukascopy:**
- Είναι τιμές **ενός** ECN παρόχου και όχι «της αγοράς». Το FX είναι αποκεντρωμένο, οπότε τα spreads και οι τιμές διαφέρουν από broker σε broker.
- Ο όγκος είναι ο όγκος του Dukascopy και **όχι** ο συνολικός όγκος της αγοράς.
- Οι ώρες που η αγορά είναι κλειστή γεμίζουν με «επίπεδα» κεριά μηδενικού όγκου. Ο validator τα αφαιρεί και τα μετράει.
- Όροι χρήσης: τα δεδομένα δίνονται δωρεάν για προσωπική και ερευνητική χρήση. Δεν τα αναδιανέμουμε: ο φάκελος `data/` δεν μπαίνει στο git.

**Περίοδος:** 2016-01 έως 2026-10 (130 πλήρεις μήνες), με χρονολογικό split 60/20/20. Για να ζεσταθούν οι δείκτες χρησιμοποιούνται επιπλέον δεδομένα από 2015-07.

| Segment | Περίοδος | Χρήση |
|---|---|---|
| In-sample (60%) | 2016-01-01 → 2022-06-30 | ανάπτυξη και sensitivity |
| Validation (20%) | 2022-07-01 → 2024-08-31 | έλεγχος development gate |
| Out-of-sample (20%) | 2024-09-01 → 2026-10-31 | **κλειδωμένο**. Ανοίγει μόνο αν περάσει το gate |

## Φάση 3: Baseline strategy S001 (κανόνες κλειδωμένοι πριν το backtest)

**Υπόθεση:** όταν το 1H MA20 περνά το MA50 προς την ίδια κατεύθυνση με την καθιερωμένη τάση του 4H (MA20 έναντι MA50, με ADX>20), η τιμή συνεχίζει αρκετά ώστε να πιάνει στόχο 2R συχνότερα από 1 στις 3 φορές. Το 1 στις 3 είναι το break-even win rate για R:R 1:2 πριν από τα κόστη.

| | Κανόνας (στο close του 1H κεριού t, με mid τιμές) |
|---|---|
| Long | MA20 cross πάνω από MA50 (1H) **και** RSI14 < 70 **και** στο τελευταίο **ολοκληρωμένο** 4H κερί: MA20 > MA50 και ADX14 > 20 |
| Short | Το συμμετρικό: cross κάτω, RSI14 > 30, 4H MA20 < MA50, ADX14 > 20 |
| Entry | Open του επόμενου 1H κεριού: στο ask για long, στο bid για short, + slippage 0.1 pip |
| Stop loss | 1.5 × ATR14(1H) από την τιμή εισόδου |
| Take profit | 2.0 × απόσταση stop (R:R 1:2) |
| Έξοδος με σήμα | Αντίθετο 1H cross, οπότε κλείσιμο στο επόμενο open (αν δεν έχει χτυπήσει πρώτα SL ή TP) |
| Θέσεις | Μέγιστο 1 θέση, χωρίς pyramiding |

Η χρονική λογική του 4H: ένα 4H κερί που ανοίγει στις L χρησιμοποιείται μόνο από 1H κεριά που κλείνουν στις L+4h ή αργότερα. Αυτό ελέγχεται με unit test.

## Position sizing (λογαριασμός σε USD, ζεύγος XXX/USD)

```
Risk $  = Equity × 0.5%          (μέγιστο 1%)
Units   = Risk $ / (Stop distance × V),   V = 1 USD ανά μονάδα για EURUSD
Pip value = 0.0001 × 100,000 = $10 ανά standard lot
Units στρογγυλοποιούνται ΠΡΟΣ ΤΑ ΚΑΤΩ σε micro lot (1,000). Όριο notional: 30:1 (υπόθεση ESMA retail)
```
Παράδειγμα: με equity $10,000 και stop 20 pips, το ρίσκο είναι $50 και η θέση 50 / 0.0020 = 25,000 units (0.25 lot).
Το **σχεδιασμένο** ρίσκο καταγράφεται χωριστά από την **πραγματική** ζημιά (R multiple), γιατί gaps και slippage μπορούν να ξεπεράσουν το −1R.

## Κόστη (υποθέσεις ECN λογαριασμού, παραμετροποιήσιμες στο `config.py`)

| Κόστος | Βάση | Stress tests |
|---|---|---|
| Spread | πραγματικό bid/ask Dukascopy ανά κερί | ×2, ×3, +0.8 pip (retail χωρίς προμήθεια) |
| Προμήθεια | $3.5 / lot / πλευρά | $5 |
| Slippage | 0.1 pip σε market και stop εντολές (όχι στο TP) | 0.5 και 1 pip |
| Swap | $6 / lot / νύχτα και στις **δύο** κατευθύνσεις, ×3 την Τετάρτη | συντηρητικό: αγνοεί το θετικό carry |

Δεν υπάρχουν δωρεάν ιστορικά swap rates, γι' αυτό το swap είναι σταθερή και συντηρητική υπόθεση.

## Execution model

- Εκτέλεση στο επόμενο κερί, ενώ το σήμα δημιουργείται στο close (no look-ahead, ελέγχεται με test σε κομμένα δεδομένα).
- Τα long κλείνουν σε τιμές bid και τα short σε τιμές ask.
- Αν το SL και το TP βρίσκονται μέσα στο ίδιο κερί, θεωρούμε **ότι χτύπησε πρώτα το SL** (συντηρητική υπόθεση).
- Αν το κερί ανοίξει πέρα από το SL λόγω gap, η εκτέλεση γίνεται στην τιμή του open.

## Acceptance criteria (κλειδωμένα πριν τα αποτελέσματα, `config.AcceptanceCriteria`)

- Sharpe (πάνω από T-bills) > 1
- PF > 1.20
- Max DD < 20%
- Καθαρή απόδοση > 0
- Τουλάχιστον 50 trades ανά segment
- Expectancy > 0R
- Τουλάχιστον 60% κερδοφόρα 6μηνα παράθυρα

**Development gate:** το OOS ανοίγει μόνο αν η στρατηγική περάσει **όλα** τα κριτήρια στο Validation **και** έχει θετική expectancy στο IS.

## Αρχιτεκτονική

```
trading_bot/
  config.py             instrument specs, costs, risk, split, acceptance criteria
  data_loader.py        Dukascopy bi5 decoder + cache, Yahoo, risk-free (BIL)
  data_validator.py     missing/duplicates/OHLC/spread/outliers/gaps/incomplete candles, cross-check
  indicators.py         SMA, RSI, ATR, ADX (Wilder), resample, look-ahead-safe HTF alignment
  strategy_base.py      interface: signal frame (entry, sl_dist, tp_dist, exits)
  strategies/           S001 ma_cross_baseline.py (+ μελλοντικές)
  risk_management.py    position sizing, leverage cap
  backtester.py         bid/ask execution, SL/TP/gaps, costs, swap, trade log, equity
  performance.py        όλα τα metrics με τους τύπους τους, acceptance check
  walk_forward.py       rolling-window stability + walk-forward re-optimisation (Φάση 8)
  monte_carlo.py        bootstrap των trades (drawdown και πιθανότητα ζημιάς)
  optimizer.py          parameter sensitivity (όχι επιλογή παραμέτρων)
  experiment_tracker.py append-only experiments.jsonl + strategy_comparison.csv
  visualization.py      equity/drawdown, κατανομή R
  main.py               εκτέλεση πειράματος: python -m trading_bot.main --strategy S001
```
