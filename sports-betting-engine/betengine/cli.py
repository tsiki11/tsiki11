"""Command line interface.

    python -m betengine download  --leagues E0 G1 --seasons 6
    python -m betengine stats     --leagues G1
    python -m betengine backtest  --leagues E0 G1 --test-from 2024-07-01
    python -m betengine predict   --leagues E0 G1 --bankroll 1000
"""
from __future__ import annotations

import argparse
from datetime import date

import pandas as pd

from . import config, data, report
from .backtest import BacktestSettings, run as run_backtest
from .predict import fit_league_models, predict_fixtures
from .stats import team_table

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)


def _leagues(args) -> list[str]:
    bad = [l for l in args.leagues if l not in config.LEAGUES]
    if bad:
        raise SystemExit(f"Άγνωστες λίγκες: {bad}. Διαθέσιμες: {', '.join(config.LEAGUES)}")
    return args.leagues


def cmd_download(args):
    seasons = data.recent_seasons(args.seasons)
    for s in seasons:
        for lg in _leagues(args):
            try:
                p = data.download(lg, s, force=args.force)
                print(f"  ✓ {s} {lg} -> {p.relative_to(config.PROJECT_DIR)}")
            except RuntimeError as e:
                print(f"  ✗ {e}")
    data.download_fixtures(force=True)
    print("  ✓ fixtures.csv")


def cmd_stats(args):
    season = data.recent_seasons(1)[0]
    matches = data.load_matches(_leagues(args), [season])
    table = team_table(matches)
    hist = data.load_matches(_leagues(args), data.recent_seasons(4))
    models = fit_league_models(hist, pd.Timestamp(date.today()) + pd.Timedelta(days=1))
    ratings = pd.concat([m.ratings().assign(league=lg) for lg, m in models.items()], ignore_index=True)
    for lg, t in table.groupby("league"):
        print(f"\n=== {config.LEAGUES[lg]} — σεζόν {season} ===")
        print(t[["θέση", "team", "αγώνες", "βαθμοί", "γκολ_υπέρ", "γκολ_κατά", "εντός_εστίας_υπέρ",
                 "εντός_εστίας_κατά", "over25_%", "φόρμα"]].round(2).to_string(index=False))
    out = report.write_excel(args.out or config.REPORTS_DIR / f"stats_{season}.xlsx",
                             {"Βαθμολογία & στατιστικά": table, "Ratings μοντέλου": ratings})
    print(f"\nExcel: {out}")


def cmd_backtest(args):
    leagues = _leagues(args)
    matches = data.load_matches(leagues, data.recent_seasons(args.seasons))
    s = BacktestSettings(test_start=pd.Timestamp(args.test_from), min_edge=args.min_edge,
                         max_odds=args.max_odds, xi=args.xi, l2=args.l2,
                         kelly_fraction=args.kelly, max_stake_pct=args.max_stake,
                         market_weight=args.market_weight, markets=tuple(args.markets))
    print(f"Αγώνες: {len(matches)} | test από {s.test_start.date()} | min edge {s.min_edge:.0%} "
          f"| market weight {s.market_weight}")
    res = run_backtest(matches, s)
    print("\n--- Ποιότητα πιθανοτήτων 1X2 (χαμηλότερο = καλύτερο) ---")
    print(res["prob_metrics"].round(4).to_string(index=False))
    print("\n--- Calibration ---")
    print(res["calibration"].round(3).to_string(index=False))
    print("\n--- Value bets (flat 1 μονάδα) ---")
    if res["bet_summary"].empty:
        print("Κανένα στοίχημα δεν πέρασε τα κριτήρια.")
    else:
        cols = ["ομάδα", "στοιχήματα", "hit_rate", "μέση_απόδοση", "κέρδος_μονάδες", "ROI", "ROI_SE",
                "μέσο_CLV", "max_drawdown_μονάδες"]
        print(res["bet_summary"][cols].round(3).to_string(index=False))
    k = res["kelly"]
    print(f"\nKelly bankroll ({s.kelly_fraction:.2f} Kelly, max {s.max_stake_pct:.0%}): "
          f"100 -> {k['τελικό']:.1f} | max drawdown {k['max_drawdown_%']:.1%}")
    settings = pd.DataFrame([{"παράμετρος": k_, "τιμή": str(v)} for k_, v in vars(s).items()])
    out = report.write_excel(args.out or config.REPORTS_DIR / "backtest.xlsx", {
        "Σύνοψη στοιχημάτων": res["bet_summary"], "Ποιότητα πιθανοτήτων": res["prob_metrics"],
        "Calibration": res["calibration"], "Στοιχήματα": res["bets"],
        "Προβλέψεις": res["predictions"], "Ρυθμίσεις": settings,
    }, highlight={"Στοιχήματα": "won"})
    print(f"Excel: {out}")


def cmd_predict(args):
    leagues = _leagues(args)
    today = pd.Timestamp(date.today())
    hist = data.load_matches(leagues, data.recent_seasons(args.seasons))
    fixtures = data.load_fixtures(leagues, from_date=today)
    if fixtures.empty:
        print("Δεν βρέθηκαν επερχόμενοι αγώνες για αυτές τις λίγκες στο fixtures.csv.")
        return
    models = fit_league_models(hist, today)
    preds, bets = predict_fixtures(fixtures, models, bankroll=args.bankroll, min_edge=args.min_edge,
                                   max_odds=args.max_odds, kelly_frac=args.kelly, max_stake_pct=args.max_stake,
                                   market_weight=args.market_weight, markets=tuple(args.markets))
    show = ["date", "league", "home", "away", "xg_home", "xg_away", "p_home", "p_draw", "p_away",
            "p_over25", "p_btts", "top_score"]
    print("\n--- Προβλέψεις ---")
    ok = preds[preds["σημείωση"] == ""] if "σημείωση" in preds else preds
    print(ok[[c for c in show if c in ok]].round(3).to_string(index=False))
    skipped = preds[preds.get("σημείωση", "") != ""] if "σημείωση" in preds else pd.DataFrame()
    if not skipped.empty:
        print(f"\n({len(skipped)} αγώνες χωρίς πρόβλεψη λόγω λίγων δεδομένων)")
    print(f"\n--- Value bets (edge ≥ {args.min_edge:.0%}, bankroll {args.bankroll:.0f}) ---")
    if bets.empty:
        print("Κανένα.")
    else:
        print(bets[["date", "league", "home", "away", "market", "selection", "p_model", "fair_odds",
                    "odds", "edge", "stake"]].round(3).to_string(index=False))
    out = report.write_excel(args.out or config.REPORTS_DIR / f"predictions_{today.date()}.xlsx",
                             {"Value bets": bets, "Προβλέψεις": preds})
    print(f"\nExcel: {out}")
    print("Υπενθύμιση: πιθανότητες, όχι βεβαιότητες. Έλεγξε την απόδοση τη στιγμή του στοιχήματος.")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="betengine", description="Μηχανή στατιστικών & value betting ποδοσφαίρου")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, seasons):
        p.add_argument("--leagues", nargs="+", default=config.DEFAULT_LEAGUES,
                       help=f"κωδικοί: {' '.join(config.LEAGUES)}")
        p.add_argument("--seasons", type=int, default=seasons, help="πόσες σεζόν ιστορικού")
        p.add_argument("--out", help="αρχείο Excel εξόδου")

    def betting(p):
        p.add_argument("--min-edge", type=float, default=config.MIN_EDGE)
        p.add_argument("--max-odds", type=float, default=config.MAX_ODDS)
        p.add_argument("--kelly", type=float, default=config.KELLY_FRACTION, help="κλάσμα Kelly")
        p.add_argument("--max-stake", type=float, default=config.MAX_STAKE_PCT, help="max %% bankroll")
        p.add_argument("--market-weight", type=float, default=config.MARKET_WEIGHT,
                       help="blending με την αγορά: 0 = μόνο μοντέλο, 1 = μόνο αγορά")
        p.add_argument("--markets", nargs="+", default=list(config.DEFAULT_MARKETS), choices=["1X2", "OU25"])

    p = sub.add_parser("download", help="κατέβασμα δεδομένων")
    common(p, 6)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("stats", help="στατιστικά ομάδων τρέχουσας σεζόν")
    common(p, 1)
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("backtest", help="walk-forward backtest")
    common(p, 6)
    betting(p)
    p.add_argument("--test-from", default=f"{data.current_season_start() - 2}-07-01")
    p.add_argument("--xi", type=float, default=config.XI)
    p.add_argument("--l2", type=float, default=config.L2)
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("predict", help="προβλέψεις & value bets για επερχόμενους αγώνες")
    common(p, 4)
    betting(p)
    p.add_argument("--bankroll", type=float, default=1000.0)
    p.set_defaults(func=cmd_predict)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
