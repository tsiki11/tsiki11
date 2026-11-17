"""Walk-forward backtest: προσομοίωση του πώς θα είχε αποδώσει η μηχανή στο παρελθόν.

Κανόνας χωρίς look-ahead bias: για κάθε εβδομάδα εκπαιδεύουμε το μοντέλο ΜΟΝΟ με αγώνες
που είχαν ήδη παιχτεί, και προβλέπουμε τους αγώνες της εβδομάδας.

Μετρικές:
  - log-loss / Brier: πόσο καλές είναι οι πιθανότητες (χαμηλότερο = καλύτερο).
    Συγκρίνονται με την αγορά (closing odds χωρίς margin) - αυτό είναι το πραγματικό benchmark.
  - ROI: κέρδος / σύνολο πονταρισμάτων.
  - CLV (closing line value): πήραμε καλύτερη απόδοση από το κλείσιμο; Θετικό CLV σε βάθος
    χρόνου είναι η πιο αξιόπιστη ένδειξη ότι ένα μοντέλο έχει πραγματικό edge.
  - Max drawdown: η μεγαλύτερη πτώση από κορυφή σε πάτο.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config
from .markets import devig, kelly_fraction
from .model import DixonColes

OUTCOMES_1X2 = [("H", "p_home", "odds_h", "close_h"), ("D", "p_draw", "odds_d", "close_d"),
                ("A", "p_away", "odds_a", "close_a")]
OUTCOMES_OU = [("Over 2.5", "p_over25", "odds_over", "close_over"),
               ("Under 2.5", "p_under25", "odds_under", "close_under")]


@dataclass
class BacktestSettings:
    test_start: pd.Timestamp
    refit_days: int = 7
    train_window_days: int = config.TRAIN_WINDOW_DAYS
    min_team_matches: int = config.MIN_TEAM_MATCHES
    xi: float = config.XI
    l2: float = config.L2
    min_edge: float = config.MIN_EDGE
    max_odds: float = config.MAX_ODDS
    kelly_fraction: float = config.KELLY_FRACTION
    max_stake_pct: float = config.MAX_STAKE_PCT
    markets: tuple = config.DEFAULT_MARKETS
    market_weight: float = config.MARKET_WEIGHT


def walk_forward_predictions(matches: pd.DataFrame, s: BacktestSettings) -> pd.DataFrame:
    """Επιστρέφει έναν πίνακα με τις out-of-sample προβλέψεις κάθε αγώνα της test περιόδου."""
    out = []
    for league, lg in matches.groupby("league"):
        lg = lg.sort_values("date")
        test = lg[lg["date"] >= s.test_start]
        if test.empty:
            continue
        model_prev = None
        start = s.test_start
        end_all = test["date"].max()
        while start <= end_all:
            end = start + pd.Timedelta(days=s.refit_days)
            batch = test[(test["date"] >= start) & (test["date"] < end)]
            if not batch.empty:
                train = lg[(lg["date"] < start) & (lg["date"] >= start - pd.Timedelta(days=s.train_window_days))]
                if len(train) >= 100:
                    model = DixonColes(xi=s.xi, l2=s.l2).fit(train, ref_date=start, warm_start=model_prev)
                    model_prev = model
                    for _, r in batch.iterrows():
                        if not (model.knows(r.home, s.min_team_matches) and model.knows(r.away, s.min_team_matches)):
                            continue
                        out.append({**r.to_dict(), **model.predict(r.home, r.away).as_dict()})
            start = end
    return pd.DataFrame(out)


def blend_with_market(preds: pd.DataFrame, w: float) -> pd.DataFrame:
    """p_τελικό = (1-w) * p_μοντέλου + w * p_αγοράς (pre-match odds χωρίς margin).

    Οι αρχικές πιθανότητες του μοντέλου κρατιούνται στις στήλες model_p_*.
    Με w > 0 ποντάρουμε μόνο όταν το μοντέλο διαφωνεί έντονα με την αγορά.
    """
    df = preds.copy()
    for c in ("p_home", "p_draw", "p_away", "p_over25", "p_under25"):
        df[f"model_{c}"] = df[c]
    if w <= 0:
        return df
    groups = [(["p_home", "p_draw", "p_away"], ["odds_h", "odds_d", "odds_a"]),
              (["p_over25", "p_under25"], ["odds_over", "odds_under"])]
    for pcols, ocols in groups:
        odds = df[ocols].to_numpy(dtype=float)
        ok = np.isfinite(odds).all(axis=1) & (odds > 1).all(axis=1)
        if ok.any():
            mkt = devig(odds[ok])
            df.loc[ok, pcols] = (1 - w) * df.loc[ok, pcols].to_numpy() + w * mkt
    return df


def _actual(row) -> dict:
    res = "H" if row.hg > row.ag else ("A" if row.hg < row.ag else "D")
    over = row.hg + row.ag > 2.5
    return {"H": res == "H", "D": res == "D", "A": res == "A", "Over 2.5": over, "Under 2.5": not over}


def select_bets(preds: pd.DataFrame, s: BacktestSettings) -> pd.DataFrame:
    """Κρατά το πολύ ένα στοίχημα ανά αγορά ανά αγώνα: το outcome με το μεγαλύτερο edge."""
    groups = []
    if "1X2" in s.markets:
        groups.append(("1X2", OUTCOMES_1X2))
    if "OU25" in s.markets:
        groups.append(("OU 2.5", OUTCOMES_OU))
    bets = []
    for _, r in preds.iterrows():
        actual = _actual(r)
        for market, outcomes in groups:
            best = None
            for name, pcol, ocol, ccol in outcomes:
                odds, p = r.get(ocol), r.get(pcol)
                if pd.isna(odds) or odds <= 1 or odds > s.max_odds:
                    continue
                edge = p * odds - 1
                if edge >= s.min_edge and (best is None or edge > best["edge"]):
                    close = r.get(ccol)
                    best = {
                        "date": r.date, "league": r.league, "home": r.home, "away": r.away,
                        "market": market, "selection": name, "p_model": p, "odds": odds,
                        "fair_odds": 1 / p, "edge": edge, "close_odds": close,
                        "clv": (odds / close - 1) if pd.notna(close) and close > 1 else np.nan,
                        "score": f"{r.hg}-{r.ag}", "won": actual[name],
                    }
            if best:
                bets.append(best)
    bets = pd.DataFrame(bets)
    if bets.empty:
        return bets
    bets = bets.sort_values("date").reset_index(drop=True)
    bets["profit_flat"] = np.where(bets["won"], bets["odds"] - 1, -1.0)
    bets["cum_profit_flat"] = bets["profit_flat"].cumsum()
    # Προσομοίωση bankroll με fractional Kelly (αρχικό κεφάλαιο = 100 μονάδες)
    bank = 100.0
    stakes, banks = [], []
    for _, b in bets.iterrows():
        f = float(kelly_fraction(b.p_model, b.odds, s.kelly_fraction, s.max_stake_pct))
        stake = bank * f
        bank += stake * (b.odds - 1) if b.won else -stake
        stakes.append(stake)
        banks.append(bank)
    bets["stake_kelly"] = stakes
    bets["bankroll_kelly"] = banks
    return bets


def max_drawdown(equity: pd.Series) -> float:
    """Μέγιστη πτώση από κορυφή, στις ίδιες μονάδες με το equity."""
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    return float((peak - equity).max())


def probability_metrics(preds: pd.DataFrame) -> pd.DataFrame:
    """Log-loss & Brier για 1X2: μοντέλο vs αγορά (opening & closing χωρίς margin)."""
    df = preds.dropna(subset=["odds_h", "odds_d", "odds_a"]).copy()
    y = np.column_stack([(df.hg > df.ag), (df.hg == df.ag), (df.hg < df.ag)]).astype(float)
    sources = {"Μοντέλο (τελική πιθανότητα)": df[["p_home", "p_draw", "p_away"]].to_numpy(),
               "Αγορά - pre-match odds": devig(df[["odds_h", "odds_d", "odds_a"]].to_numpy())}
    closing = df[["close_h", "close_d", "close_a"]]
    mask = closing.notna().all(axis=1).to_numpy()
    rows = []
    if mask.any():
        sources["Αγορά - closing odds"] = None
    for name, p in sources.items():
        if p is None:
            p_use, y_use = devig(closing.to_numpy()[mask]), y[mask]
        else:
            p_use, y_use = p, y
        p_use = np.clip(p_use, 1e-9, 1)
        rows.append({"πηγή": name, "αγώνες": len(y_use),
                     "log_loss": float(-(y_use * np.log(p_use)).sum(axis=1).mean()),
                     "brier": float(((p_use - y_use) ** 2).sum(axis=1).mean())})
    # Ίδιο δείγμα για δίκαιη σύγκριση με το closing
    if mask.any() and mask.sum() < len(df):
        p_m = np.clip(sources["Μοντέλο (τελική πιθανότητα)"][mask], 1e-9, 1)
        rows.append({"πηγή": "Μοντέλο (ίδιο δείγμα με closing)", "αγώνες": int(mask.sum()),
                     "log_loss": float(-(y[mask] * np.log(p_m)).sum(axis=1).mean()),
                     "brier": float(((p_m - y[mask]) ** 2).sum(axis=1).mean())})
    return pd.DataFrame(rows)


def calibration_table(preds: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    """Όταν το μοντέλο λέει Χ%, πόσο συχνά συμβαίνει πράγματι;"""
    p = np.concatenate([preds.p_home, preds.p_draw, preds.p_away])
    y = np.concatenate([preds.hg > preds.ag, preds.hg == preds.ag, preds.hg < preds.ag]).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    cat = pd.cut(p, edges, include_lowest=True)
    t = pd.DataFrame({"bin": cat, "p": p, "y": y}).groupby("bin", observed=True).agg(
        προβλέψεις=("p", "size"), μέση_πιθανότητα=("p", "mean"), πραγματική_συχνότητα=("y", "mean"))
    t["απόκλιση"] = t["πραγματική_συχνότητα"] - t["μέση_πιθανότητα"]
    return t.reset_index().assign(bin=lambda d: d["bin"].astype(str))


def summarize_bets(bets: pd.DataFrame) -> pd.DataFrame:
    if bets.empty:
        return pd.DataFrame()

    def agg(g: pd.DataFrame) -> pd.Series:
        n = len(g)
        profit = g["profit_flat"].sum()
        # Τυπικό σφάλμα ROI: αν το |ROI| < ~2 SE, το αποτέλεσμα δεν ξεχωρίζει από την τύχη
        se = g["profit_flat"].std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
        return pd.Series({
            "στοιχήματα": n, "hit_rate": g["won"].mean(), "μέση_απόδοση": g["odds"].mean(),
            "μέσο_edge_μοντέλου": g["edge"].mean(), "κέρδος_μονάδες": profit, "ROI": profit / n,
            "ROI_SE": se, "μέσο_CLV": g["clv"].mean(), "CLV_θετικό_%": (g["clv"] > 0).mean(),
            "max_drawdown_μονάδες": max_drawdown(g["profit_flat"].cumsum()),
        })

    parts = [agg(bets).rename("ΣΥΝΟΛΟ")]
    for (lg, mk), g in bets.groupby(["league", "market"]):
        parts.append(agg(g).rename(f"{lg} | {mk}"))
    return pd.DataFrame(parts).reset_index().rename(columns={"index": "ομάδα"})


def run(matches: pd.DataFrame, s: BacktestSettings) -> dict:
    preds = walk_forward_predictions(matches, s)
    if preds.empty:
        raise RuntimeError("Καμία πρόβλεψη - λίγα δεδομένα πριν από την test περίοδο;")
    return evaluate(preds, s)


def evaluate(preds: pd.DataFrame, s: BacktestSettings) -> dict:
    """Αξιολόγηση έτοιμων προβλέψεων (χωρίς ξανά fit) - χρήσιμο για δοκιμές ρυθμίσεων."""
    preds = blend_with_market(preds, s.market_weight)
    bets = select_bets(preds, s)
    kelly_final = float(bets["bankroll_kelly"].iloc[-1]) if not bets.empty else 100.0
    kelly_dd = 0.0
    if not bets.empty:
        eq = pd.concat([pd.Series([100.0]), bets["bankroll_kelly"]], ignore_index=True)
        kelly_dd = float(((eq.cummax() - eq) / eq.cummax()).max())
    return {
        "predictions": preds, "bets": bets,
        "prob_metrics": probability_metrics(preds),
        "calibration": calibration_table(preds),
        "bet_summary": summarize_bets(bets),
        "kelly": {"αρχικό": 100.0, "τελικό": kelly_final, "max_drawdown_%": kelly_dd},
    }
