"""Προβλέψεις για μελλοντικούς αγώνες και εντοπισμός value bets."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .backtest import OUTCOMES_1X2, OUTCOMES_OU, blend_with_market
from .markets import kelly_fraction, overround
from .model import DixonColes


def fit_league_models(matches: pd.DataFrame, ref_date, window_days: int = config.TRAIN_WINDOW_DAYS,
                      xi: float = config.XI, l2: float = config.L2) -> dict[str, DixonColes]:
    ref_date = pd.Timestamp(ref_date)
    models = {}
    for lg, df in matches.groupby("league"):
        train = df[(df.date < ref_date) & (df.date >= ref_date - pd.Timedelta(days=window_days))]
        if len(train) >= 100:
            models[lg] = DixonColes(xi=xi, l2=l2).fit(train, ref_date=ref_date)
    return models


def fit_corner_models(matches: pd.DataFrame, ref_date, window_days: int = config.TRAIN_WINDOW_DAYS) -> dict:
    from .experiments import CornerPredictor
    ref_date = pd.Timestamp(ref_date)
    out = {}
    for lg, df in matches.groupby("league"):
        train = df[(df.date < ref_date) & (df.date >= ref_date - pd.Timedelta(days=window_days))]
        train = train.dropna(subset=["hc", "ac"])
        if len(train) >= 100:
            out[lg] = CornerPredictor(train, ref_date)
    return out


def predict_corners(fixtures: pd.DataFrame, corner_models: dict) -> pd.DataFrame:
    rows = []
    for _, r in fixtures.iterrows():
        cm = corner_models.get(r.league)
        p = cm.predict(r.home, r.away) if cm else None
        if p:
            rows.append({"date": r.date, "time": r.get("time", ""), "league": r.league,
                         "home": r.home, "away": r.away, **p,
                         "απόδοση_bookmaker_over": None, "απόδοση_bookmaker_under": None, "γραμμή": None})
    return pd.DataFrame(rows)


def predict_fixtures(fixtures: pd.DataFrame, models: dict[str, DixonColes], bankroll: float = 1000.0,
                     min_edge: float = config.MIN_EDGE, max_odds: float = config.MAX_ODDS,
                     kelly_frac: float = config.KELLY_FRACTION, max_stake_pct: float = config.MAX_STAKE_PCT,
                     min_team_matches: int = config.MIN_TEAM_MATCHES,
                     market_weight: float = config.MARKET_WEIGHT,
                     markets: tuple = config.DEFAULT_MARKETS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Επιστρέφει (πίνακας προβλέψεων, πίνακας value bets)."""
    rows, skipped = [], []
    for _, r in fixtures.iterrows():
        m = models.get(r.league)
        if m is None:
            continue
        base = {"date": r.date, "time": r.get("time", ""), "league": r.league, "home": r.home, "away": r.away,
                "odds_h": r.odds_h, "odds_d": r.odds_d, "odds_a": r.odds_a,
                "odds_over": r.odds_over, "odds_under": r.odds_under}
        missing = [t for t in (r.home, r.away) if not m.knows(t, min_team_matches)]
        if missing:
            skipped.append({**base, "σημείωση": f"Λίγα δεδομένα για: {', '.join(missing)}"})
            continue
        rows.append({**base, **m.predict(r.home, r.away).as_dict(), "σημείωση": ""})

    preds = pd.DataFrame(rows)
    bets = []
    if not preds.empty:
        preds = blend_with_market(preds, market_weight)
        for k, col in (("h", "p_home"), ("d", "p_draw"), ("a", "p_away"),
                       ("over25", "p_over25"), ("under25", "p_under25")):
            preds[f"fair_{k}"] = 1 / preds[col]
        preds["margin_1x2"] = [overround([a, b, c]) if pd.notna(a) else np.nan
                               for a, b, c in zip(preds.odds_h, preds.odds_d, preds.odds_a)]
        for _, r in preds.iterrows():
            base = {c: r[c] for c in ("date", "time", "league", "home", "away")}
            groups = [g for key, g in (("1X2", ("1X2", OUTCOMES_1X2)), ("OU25", ("OU 2.5", OUTCOMES_OU)))
                      if key in markets]
            for market, outcomes in groups:
                best = None
                for name, pcol, ocol, _ in outcomes:
                    odds, prob = r[ocol], r[pcol]
                    if pd.isna(odds) or odds <= 1 or odds > max_odds:
                        continue
                    edge = prob * odds - 1
                    if edge >= min_edge and (best is None or edge > best["edge"]):
                        f = float(kelly_fraction(prob, odds, kelly_frac, max_stake_pct))
                        best = {**base, "market": market, "selection": name, "p_model_raw": r[f"model_{pcol}"],
                                "p_model": prob, "fair_odds": 1 / prob, "odds": odds, "edge": edge,
                                "kelly_%": f, "stake": round(bankroll * f, 2)}
                if best:
                    bets.append(best)
    preds = pd.concat([preds, pd.DataFrame(skipped)], ignore_index=True)
    bets_df = pd.DataFrame(bets)
    if not bets_df.empty:
        bets_df = bets_df.sort_values("edge", ascending=False).reset_index(drop=True)
    return preds, bets_df
