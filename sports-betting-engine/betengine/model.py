"""Μοντέλο Dixon-Coles (1997) για τα γκολ ενός αγώνα.

Ιδέα σε μία γραμμή: τα γκολ κάθε ομάδας ακολουθούν κατανομή Poisson με μέσο όρο
που εξαρτάται από την επίθεσή της, την άμυνα του αντιπάλου και το πλεονέκτημα έδρας.

    λ (γκολ γηπεδούχου)  = exp(μ0 + home + att[γηπ.] + def[φιλ.])
    μ (γκολ φιλοξενούμ.) = exp(μ0 +        att[φιλ.] + def[γηπ.])

    att > 0  -> σκοράρει περισσότερο από τον μέσο όρο
    def > 0  -> δέχεται περισσότερα από τον μέσο όρο (αδύναμη άμυνα)

Δύο βελτιώσεις του Dixon-Coles πάνω στο απλό Poisson:
  1. ρ (rho): διορθώνει τις πιθανότητες των χαμηλών σκορ 0-0, 1-0, 0-1, 1-1,
     που στην πράξη δεν είναι ανεξάρτητα.
  2. time decay: οι πρόσφατοι αγώνες μετράνε περισσότερο, βάρος = exp(-ξ * ημέρες).

Οι παράμετροι βρίσκονται με maximum likelihood (L-BFGS-B, με αναλυτικό gradient).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

from . import config


@dataclass
class MatchPrediction:
    home: str
    away: str
    exp_home_goals: float
    exp_away_goals: float
    score_matrix: np.ndarray = field(repr=False)

    @property
    def p_home(self) -> float:
        return float(np.tril(self.score_matrix, -1).sum())

    @property
    def p_draw(self) -> float:
        return float(np.trace(self.score_matrix))

    @property
    def p_away(self) -> float:
        return float(np.triu(self.score_matrix, 1).sum())

    def p_over(self, line: float = 2.5) -> float:
        n = self.score_matrix.shape[0]
        total = np.add.outer(np.arange(n), np.arange(n))
        return float(self.score_matrix[total > line].sum())

    @property
    def p_btts(self) -> float:
        return float(self.score_matrix[1:, 1:].sum())

    def most_likely_score(self) -> tuple[int, int, float]:
        i, j = np.unravel_index(np.argmax(self.score_matrix), self.score_matrix.shape)
        return int(i), int(j), float(self.score_matrix[i, j])

    def as_dict(self) -> dict:
        hs, as_, ps = self.most_likely_score()
        return {
            "xg_home": self.exp_home_goals, "xg_away": self.exp_away_goals,
            "p_home": self.p_home, "p_draw": self.p_draw, "p_away": self.p_away,
            "p_over25": self.p_over(2.5), "p_under25": 1 - self.p_over(2.5),
            "p_btts": self.p_btts, "top_score": f"{hs}-{as_}", "p_top_score": ps,
        }


def _tau(x, y, lam, mu, rho):
    """Διόρθωση Dixon-Coles για τα σκορ 0-0, 0-1, 1-0, 1-1 (διανυσματικά)."""
    t = np.ones_like(lam, dtype=float)
    m00 = (x == 0) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0)
    m11 = (x == 1) & (y == 1)
    t[m00] = 1 - lam[m00] * mu[m00] * rho
    t[m01] = 1 + lam[m01] * rho
    t[m10] = 1 + mu[m10] * rho
    t[m11] = 1 - rho
    return t, (m00, m01, m10, m11)


class DixonColes:
    def __init__(self, xi: float = config.XI, l2: float = config.L2, max_goals: int = config.MAX_GOALS):
        self.xi = xi
        self.l2 = l2
        self.max_goals = max_goals
        self.teams: list[str] = []
        self.params: dict | None = None
        self.team_matches: dict[str, int] = {}

    # ---------- fitting ----------
    def _unpack(self, theta):
        n = len(self.teams)
        return theta[:n], theta[n:2 * n], theta[2 * n], theta[2 * n + 1], theta[2 * n + 2]

    def _nll(self, theta, hi, ai, x, y, w):
        att, dfn, mu0, home, rho = self._unpack(theta)
        n = len(self.teams)
        eta1 = mu0 + home + att[hi] + dfn[ai]
        eta2 = mu0 + att[ai] + dfn[hi]
        lam, mu = np.exp(eta1), np.exp(eta2)

        t, (m00, m01, m10, m11) = _tau(x, y, lam, mu, rho)
        t = np.clip(t, 1e-10, None)
        # log Poisson χωρίς τον σταθερό όρο log(x!)
        ll = np.log(t) + x * eta1 - lam + y * eta2 - mu
        nll = -(w * ll).sum() + self.l2 * (att @ att + dfn @ dfn)

        # ---- αναλυτικό gradient ----
        g1 = x - lam            # d ll / d eta1
        g2 = y - mu             # d ll / d eta2
        grho = np.zeros_like(lam)
        g1[m00] += -lam[m00] * mu[m00] * rho / t[m00]
        g2[m00] += -lam[m00] * mu[m00] * rho / t[m00]
        grho[m00] = -lam[m00] * mu[m00] / t[m00]
        g1[m01] += lam[m01] * rho / t[m01]
        grho[m01] = lam[m01] / t[m01]
        g2[m10] += mu[m10] * rho / t[m10]
        grho[m10] = mu[m10] / t[m10]
        grho[m11] = -1 / t[m11]

        wg1, wg2 = w * g1, w * g2
        g_att = np.bincount(hi, wg1, n) + np.bincount(ai, wg2, n)
        g_def = np.bincount(ai, wg1, n) + np.bincount(hi, wg2, n)
        grad = -np.concatenate([g_att, g_def, [wg1.sum() + wg2.sum(), wg1.sum(), (w * grho).sum()]])
        grad[:n] += 2 * self.l2 * att
        grad[n:2 * n] += 2 * self.l2 * dfn
        return nll, grad

    def fit(self, matches: pd.DataFrame, ref_date=None, warm_start: "DixonColes | None" = None) -> "DixonColes":
        """matches: DataFrame με στήλες date, home, away, hg, ag."""
        if matches.empty:
            raise ValueError("Κενό training set")
        ref_date = pd.Timestamp(ref_date) if ref_date is not None else matches["date"].max()
        self.teams = sorted(set(matches["home"]) | set(matches["away"]))
        idx = {t: i for i, t in enumerate(self.teams)}
        hi = matches["home"].map(idx).to_numpy()
        ai = matches["away"].map(idx).to_numpy()
        x = matches["hg"].to_numpy(dtype=float)
        y = matches["ag"].to_numpy(dtype=float)
        days = (ref_date - matches["date"]).dt.days.to_numpy(dtype=float)
        w = np.exp(-self.xi * np.clip(days, 0, None))

        counts = pd.concat([matches["home"], matches["away"]]).value_counts()
        self.team_matches = counts.to_dict()

        n = len(self.teams)
        theta0 = np.zeros(2 * n + 3)
        theta0[2 * n] = np.log(max(np.average((x + y) / 2, weights=w), 0.1))
        theta0[2 * n + 1] = 0.25
        if warm_start is not None and warm_start.params is not None:
            p = warm_start.params
            for t, i in idx.items():
                theta0[i] = p["attack"].get(t, 0.0)
                theta0[n + i] = p["defence"].get(t, 0.0)
            theta0[2 * n], theta0[2 * n + 1], theta0[2 * n + 2] = p["mu0"], p["home"], p["rho"]

        bounds = [(-3, 3)] * (2 * n) + [(-3, 3), (-1, 1), (-0.2, 0.2)]
        res = minimize(self._nll, theta0, args=(hi, ai, x, y, w), jac=True,
                       method="L-BFGS-B", bounds=bounds, options={"maxiter": 500})
        att, dfn, mu0, home, rho = self._unpack(res.x)
        self.params = {
            "attack": dict(zip(self.teams, att)), "defence": dict(zip(self.teams, dfn)),
            "mu0": float(mu0), "home": float(home), "rho": float(rho),
            "converged": bool(res.success),
        }
        return self

    # ---------- prediction ----------
    def knows(self, team: str, min_matches: int = 0) -> bool:
        return team in self.team_matches and self.team_matches[team] >= min_matches

    def expected_goals(self, home: str, away: str) -> tuple[float, float]:
        p = self.params
        lam = np.exp(p["mu0"] + p["home"] + p["attack"][home] + p["defence"][away])
        mu = np.exp(p["mu0"] + p["attack"][away] + p["defence"][home])
        return float(lam), float(mu)

    def predict(self, home: str, away: str) -> MatchPrediction:
        if self.params is None:
            raise RuntimeError("Το μοντέλο δεν έχει εκπαιδευτεί (fit)")
        lam, mu = self.expected_goals(home, away)
        g = np.arange(self.max_goals + 1)
        m = np.outer(poisson.pmf(g, lam), poisson.pmf(g, mu))
        rho = self.params["rho"]
        m[0, 0] *= 1 - lam * mu * rho
        m[0, 1] *= 1 + lam * rho
        m[1, 0] *= 1 + mu * rho
        m[1, 1] *= 1 - rho
        m = np.clip(m, 0, None)
        m /= m.sum()
        return MatchPrediction(home, away, lam, mu, m)

    def ratings(self) -> pd.DataFrame:
        """Πίνακας δυναμικότητας: αναμενόμενα γκολ υπέρ/κατά απέναντι σε μέση ομάδα, σε ουδέτερο γήπεδο."""
        p = self.params
        rows = []
        for t in self.teams:
            gf = np.exp(p["mu0"] + p["attack"][t])
            ga = np.exp(p["mu0"] + p["defence"][t])
            rows.append({"team": t, "attack": p["attack"][t], "defence": p["defence"][t],
                         "xGF_vs_avg": gf, "xGA_vs_avg": ga, "xGD_vs_avg": gf - ga,
                         "matches_in_window": self.team_matches.get(t, 0)})
        return pd.DataFrame(rows).sort_values("xGD_vs_avg", ascending=False).reset_index(drop=True)
