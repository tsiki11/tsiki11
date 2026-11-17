"""Out-of-sample τεστ σεζόν: εκμάθηση σε κάποιες σεζόν, προβλέψεις σε άλλη, σύγκριση με τα πραγματικά.

Δύο τρόποι:
  A. frozen       - ένα fit στο τέλος των training σεζόν, καμία ενημέρωση κατά τη test σεζόν.
  B. walk-forward - ίδιο ξεκίνημα, αλλά κάθε εβδομάδα το μοντέλο μαθαίνει και τους αγώνες που έχουν ήδη παιχτεί.

Νεοφώτιστες ομάδες στο frozen: παίρνουν τα μέσα ratings των ομάδων που υποβιβάστηκαν
(ήταν στην τελευταία training σεζόν αλλά όχι στην test).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .backtest import BacktestSettings, evaluate, walk_forward_predictions
from .markets import devig
from .model import DixonColes


def frozen_predictions(train: pd.DataFrame, test: pd.DataFrame, xi: float = config.XI,
                       l2: float = config.L2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Επιστρέφει (προβλέψεις test σεζόν, πίνακας νεοφώτιστων με τα ratings που τους δόθηκαν)."""
    rows, promoted_info = [], []
    last_train_season = train["season"].max()
    for lg, tr in train.groupby("league"):
        te = test[test.league == lg]
        if te.empty:
            continue
        ref = te["date"].min()
        model = DixonColes(xi=xi, l2=l2).fit(tr, ref_date=ref)
        last = tr[tr.season == last_train_season]
        last_teams = set(last.home) | set(last.away)
        test_teams = set(te.home) | set(te.away)
        relegated = sorted(last_teams - test_teams)
        promoted = sorted(test_teams - set(model.teams))
        p = model.params
        if relegated:
            att_r = float(np.mean([p["attack"][t] for t in relegated]))
            def_r = float(np.mean([p["defence"][t] for t in relegated]))
        else:  # fallback: τα 3 πιο αδύναμα ratings
            worst = model.ratings().tail(3)
            att_r, def_r = float(worst.attack.mean()), float(worst.defence.mean())
        for t in promoted:
            p["attack"][t], p["defence"][t] = att_r, def_r
            promoted_info.append({"league": lg, "team": t, "attack": att_r, "defence": def_r,
                                  "βάση": "μέσος όρος υποβιβασθέντων: " + ", ".join(relegated)})
        promoted_set = set(promoted)
        for _, r in te.iterrows():
            rows.append({**r.to_dict(), **model.predict(r.home, r.away).as_dict(),
                         "νεοφώτιστη_στον_αγώνα": r.home in promoted_set or r.away in promoted_set})
    return pd.DataFrame(rows), pd.DataFrame(promoted_info)


def _ll(p, y):
    p = np.clip(p, 1e-9, 1)
    return float(-(y * np.log(p)).sum(axis=1).mean())


def match_report(preds: pd.DataFrame) -> pd.DataFrame:
    """Ανά αγώνα: πρόβλεψη μοντέλου vs αγορά vs πραγματικό αποτέλεσμα."""
    df = preds.copy()
    df["αποτέλεσμα"] = np.select([df.hg > df.ag, df.hg == df.ag], ["1", "X"], "2")
    probs = df[["p_home", "p_draw", "p_away"]].to_numpy()
    df["πρόβλεψη_μοντέλου"] = np.array(["1", "X", "2"])[probs.argmax(axis=1)]
    odds = df[["odds_h", "odds_d", "odds_a"]].to_numpy(dtype=float)
    ok = np.isfinite(odds).all(axis=1)
    fav = np.full(len(df), "", dtype=object)
    fav[ok] = np.array(["1", "X", "2"])[np.nanargmin(odds[ok], axis=1)]
    df["φαβορί_αγοράς"] = fav
    mk = np.full((len(df), 3), np.nan)
    mk[ok] = devig(odds[ok])
    df["mkt_p_home"], df["mkt_p_draw"], df["mkt_p_away"] = mk.T
    df["σωστό_μοντέλο"] = df["πρόβλεψη_μοντέλου"] == df["αποτέλεσμα"]
    df["σωστό_αγορά"] = np.where(ok, df["φαβορί_αγοράς"] == df["αποτέλεσμα"], np.nan)
    df["συνολικά_γκολ"] = df.hg + df.ag
    df["over25_πραγματικό"] = df["συνολικά_γκολ"] > 2.5
    df["σωστό_OU_μοντέλο"] = (df.p_over25 > 0.5) == df["over25_πραγματικό"]
    cols = ["date", "league", "home", "away", "hg", "ag", "αποτέλεσμα", "xg_home", "xg_away",
            "p_home", "p_draw", "p_away", "mkt_p_home", "mkt_p_draw", "mkt_p_away",
            "πρόβλεψη_μοντέλου", "φαβορί_αγοράς", "σωστό_μοντέλο", "σωστό_αγορά",
            "p_over25", "over25_πραγματικό", "σωστό_OU_μοντέλο", "top_score", "odds_h", "odds_d", "odds_a",
            "odds_over", "odds_under", "close_h", "close_d", "close_a"]
    if "νεοφώτιστη_στον_αγώνα" in df:
        cols.append("νεοφώτιστη_στον_αγώνα")
    return df[cols]


def summary(preds: pd.DataFrame, label: str) -> pd.DataFrame:
    """Συνοπτικές μετρικές ανά λίγκα και συνολικά."""
    out = []
    for name, g in [("ΣΥΝΟΛΟ", preds)] + sorted(preds.groupby("league")):
        g = g.dropna(subset=["odds_h", "odds_d", "odds_a"])
        if g.empty:
            continue
        y = np.column_stack([g.hg > g.ag, g.hg == g.ag, g.hg < g.ag]).astype(float)
        pm = g[["p_home", "p_draw", "p_away"]].to_numpy()
        pk = devig(g[["odds_h", "odds_d", "odds_a"]].to_numpy(dtype=float))
        over = (g.hg + g.ag > 2.5).to_numpy()
        ou_odds = g[["odds_over", "odds_under"]].to_numpy(dtype=float)
        ok_ou = np.isfinite(ou_odds).all(axis=1)
        mk_over = devig(ou_odds[ok_ou])[:, 0] if ok_ou.any() else np.array([])
        out.append({
            "τρόπος": label, "λίγκα": name, "αγώνες": len(g),
            "1X2_σωστά_μοντέλο": float((pm.argmax(1) == y.argmax(1)).mean()),
            "1X2_σωστά_φαβορί_αγοράς": float((pk.argmax(1) == y.argmax(1)).mean()),
            "log_loss_μοντέλο": _ll(pm, y), "log_loss_αγορά": _ll(pk, y),
            "OU_σωστά_μοντέλο": float(((g.p_over25.to_numpy() > 0.5) == over).mean()),
            "OU_σωστά_αγορά": float(((mk_over > 0.5) == over[ok_ou]).mean()) if ok_ou.any() else np.nan,
            "μέσα_γκολ_πρόβλεψη": float((g.xg_home + g.xg_away).mean()),
            "μέσα_γκολ_πραγματικά": float((g.hg + g.ag).mean()),
            "μέση_P(over)_μοντέλο": float(g.p_over25.mean()),
            "μέση_P(over)_αγορά": float(mk_over.mean()) if ok_ou.any() else np.nan,
            "over25_πραγματικό_%": float(over.mean()),
        })
    return pd.DataFrame(out)


def bets_for(preds: pd.DataFrame, label: str, market_weight: float, markets: tuple,
             min_edge: float = config.MIN_EDGE) -> tuple[pd.DataFrame, pd.DataFrame]:
    s = BacktestSettings(test_start=preds.date.min(), market_weight=market_weight,
                         markets=markets, min_edge=min_edge)
    res = evaluate(preds, s)
    summ = res["bet_summary"]
    if not summ.empty:
        summ.insert(0, "σενάριο", f"{label} | w={market_weight} | {'+'.join(markets)}")
    bets = res["bets"]
    if not bets.empty:
        bets.insert(0, "σενάριο", f"{label} | w={market_weight} | {'+'.join(markets)}")
    return summ, bets


def run(matches: pd.DataFrame, train_seasons: list[str], test_season: str) -> dict:
    train = matches[matches.season.isin(train_seasons)]
    test = matches[matches.season == test_season]
    if train.empty or test.empty:
        raise RuntimeError("Λείπουν δεδομένα για τις σεζόν που ζητήθηκαν.")

    frozen, promoted = frozen_predictions(train, test)
    wf_settings = BacktestSettings(test_start=test.date.min(), train_window_days=10 * 365)
    wf = walk_forward_predictions(pd.concat([train, test]), wf_settings)
    wf = wf[wf.season == test_season]

    summaries = pd.concat([summary(frozen, "A. frozen"), summary(wf, "B. walk-forward")], ignore_index=True)
    bet_summ, bet_rows = [], []
    for label, preds in (("A. frozen", frozen), ("B. walk-forward", wf)):
        for w in (0.0, 0.6):
            for mk in (("1X2",), ("OU25",)):
                s, b = bets_for(preds, label, w, mk)
                bet_summ.append(s)
                bet_rows.append(b)
    return {
        "summary": summaries,
        "bet_summary": pd.concat(bet_summ, ignore_index=True),
        "bets": pd.concat(bet_rows, ignore_index=True),
        "frozen_matches": match_report(frozen),
        "wf_matches": match_report(wf),
        "promoted": promoted,
    }
