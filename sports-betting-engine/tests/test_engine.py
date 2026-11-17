import numpy as np
import pandas as pd
import pytest
from scipy.optimize import approx_fprime

from betengine.backtest import BacktestSettings, blend_with_market, evaluate, walk_forward_predictions
from betengine.markets import devig, expected_value, kelly_fraction, overround
from betengine.model import DixonColes
from betengine.stats import team_table


def synthetic_league(seasons=3, n_teams=12, seed=0):
    """Συνθετική λίγκα με γνωστές δυνάμεις ομάδων, για να ελέγξουμε ότι το μοντέλο τις ξαναβρίσκει."""
    rng = np.random.default_rng(seed)
    teams = [f"T{i:02d}" for i in range(n_teams)]
    att = rng.normal(0, 0.3, n_teams); att -= att.mean()
    dfn = rng.normal(0, 0.25, n_teams); dfn -= dfn.mean()
    rows, d0 = [], pd.Timestamp("2022-08-01")
    for s in range(seasons):
        k = 0
        for i in range(n_teams):
            for j in range(n_teams):
                if i == j:
                    continue
                lam = np.exp(0.15 + 0.25 + att[i] + dfn[j]); mu = np.exp(0.15 + att[j] + dfn[i])
                hg, ag = rng.poisson(lam), rng.poisson(mu)
                date = d0 + pd.Timedelta(days=365 * s + 2 * k)
                k += 1
                rows.append(dict(league="X1", season=str(s), date=date, home=teams[i], away=teams[j],
                                 hg=hg, ag=ag, odds_h=2.4, odds_d=3.3, odds_a=3.1, odds_over=1.95,
                                 odds_under=1.9, close_h=2.4, close_d=3.3, close_a=3.1,
                                 close_over=1.95, close_under=1.9,
                                 hs=10, as_=8, hst=4, ast=3, hc=5, ac=4, hy=1, ay=2, hr=0, ar=0))
    return pd.DataFrame(rows), dict(zip(teams, att))


# ---------- markets ----------
def test_devig_and_overround():
    odds = [2.0, 3.4, 4.0]
    assert overround(odds) == pytest.approx(0.0441, abs=1e-3)
    p = devig(odds)
    assert p.sum() == pytest.approx(1.0)
    assert p[0] == pytest.approx(0.479, abs=1e-3)


def test_ev_and_kelly():
    assert expected_value(0.55, 2.0) == pytest.approx(0.10)
    assert kelly_fraction(0.55, 2.0) == pytest.approx(0.10)          # (0.55*2-1)/(2-1)
    assert kelly_fraction(0.55, 2.0, 0.25) == pytest.approx(0.025)
    assert kelly_fraction(0.40, 2.0) == 0                             # αρνητικό EV -> 0
    assert kelly_fraction(0.9, 2.0, 1.0, cap=0.02) == pytest.approx(0.02)


# ---------- model ----------
def test_gradient_matches_numeric():
    df, _ = synthetic_league(seasons=1)
    m = DixonColes(xi=0.001, l2=0.5)
    m.teams = sorted(set(df.home))
    idx = {t: i for i, t in enumerate(m.teams)}
    args = (df.home.map(idx).values, df.away.map(idx).values,
            df.hg.values.astype(float), df.ag.values.astype(float), np.linspace(0.5, 1, len(df)))
    theta = np.random.default_rng(1).normal(0, 0.2, 2 * len(m.teams) + 3)
    theta[-1] = 0.05
    num = approx_fprime(theta, lambda t: m._nll(t, *args)[0], 1e-6)
    assert np.allclose(m._nll(theta, *args)[1], num, rtol=1e-3, atol=1e-2)


def test_fit_recovers_known_strengths():
    df, true_att = synthetic_league(seasons=4)
    m = DixonColes(xi=0.0, l2=0.0).fit(df)
    est = np.array([m.params["attack"][t] for t in true_att])
    assert np.corrcoef(est, list(true_att.values()))[0, 1] > 0.85
    assert 0.1 < m.params["home"] < 0.4


def test_prediction_probabilities_are_consistent():
    df, _ = synthetic_league()
    p = DixonColes().fit(df).predict("T00", "T01")
    assert p.score_matrix.sum() == pytest.approx(1.0)
    assert p.p_home + p.p_draw + p.p_away == pytest.approx(1.0)
    assert 0 < p.p_over(2.5) < 1 and 0 < p.p_btts < 1


# ---------- backtest ----------
def test_walk_forward_has_no_lookahead():
    """Κάθε πρόβλεψη πρέπει να βγαίνει από μοντέλο που δεν είδε τον ίδιο τον αγώνα:
    αν αλλάξουμε το σκορ ενός test αγώνα, η πρόβλεψή του δεν πρέπει να αλλάξει."""
    df, _ = synthetic_league(seasons=3)
    start = pd.Timestamp("2024-08-01")
    s = BacktestSettings(test_start=start, min_team_matches=1)
    a = walk_forward_predictions(df, s)
    target = a.iloc[0]
    df2 = df.copy()
    sel = (df2.date == target.date) & (df2.home == target.home) & (df2.away == target.away)
    df2.loc[sel, ["hg", "ag"]] = [9, 0]
    b = walk_forward_predictions(df2, s)
    assert b.iloc[0].p_home == pytest.approx(target.p_home)


def test_blend_and_evaluate_run():
    df, _ = synthetic_league(seasons=3)
    s = BacktestSettings(test_start=pd.Timestamp("2024-08-01"), min_team_matches=1, market_weight=0.5)
    preds = walk_forward_predictions(df, s)
    blended = blend_with_market(preds, 1.0)
    assert np.allclose(blended[["p_home", "p_draw", "p_away"]].sum(axis=1), 1.0)
    assert blended.p_home.iloc[0] == pytest.approx(devig([2.4, 3.3, 3.1])[0])
    res = evaluate(preds, s)
    assert {"bets", "prob_metrics", "calibration", "bet_summary"} <= set(res)


def test_team_table_points():
    df = pd.DataFrame(dict(league="X1", date=pd.to_datetime(["2024-01-01", "2024-01-08"]),
                           home=["A", "B"], away=["B", "A"], hg=[2, 1], ag=[0, 1],
                           hs=1, as_=1, hst=1, ast=1, hc=1, ac=1, hy=0, ay=0, hr=0, ar=0))
    t = team_table(df).set_index("team")
    assert t.loc["A", "βαθμοί"] == 4 and t.loc["B", "βαθμοί"] == 1
    assert t.loc["A", "φόρμα"] == "WD"
