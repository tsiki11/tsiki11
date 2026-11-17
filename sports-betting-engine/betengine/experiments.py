"""Πειράματα: μοντέλο σουτ εντός εστίας (SoT) και πρόβλεψη κόρνερ.

1) SoT μοντέλο για γκολ
   Τα γκολ είναι "θορυβώδη": μια ομάδα μπορεί να έχει 8 σουτ εντός εστίας και 0 γκολ.
   Τα σουτ εντός εστίας δείχνουν πιο σταθερά την ποιότητα. Οπότε:
     - Fit Poisson μοντέλου (ίδια δομή με Dixon-Coles) στα SoT αντί για τα γκολ
     - Αναμενόμενα γκολ = αναμενόμενα SoT x ποσοστό μετατροπής της λίγκας (γκολ / SoT)
   Υβριδικό: γεωμετρικός μέσος των αναμενόμενων γκολ από το μοντέλο γκολ και το SoT μοντέλο.

2) Κόρνερ
   Poisson μοντέλο στα κόρνερ κάθε ομάδας (επιθετική "πίεση" & αμυντική "παραχώρηση" κόρνερ).
   Τα συνολικά κόρνερ έχουν μεγαλύτερη διασπορά από Poisson -> negative binomial
   με παράμετρο υπερδιασποράς που εκτιμάται από το training set.
   Baseline σύγκρισης: ο μέσος όρος κόρνερ της λίγκας (ίδια κατανομή, χωρίς πληροφορία ομάδων).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson

from . import config
from .markets import devig
from .model import DixonColes

CORNER_LINES = config.CORNER_LINES
MODELS = ("goals", "sot", "hybrid")


def _probs_from_xg(lam: float, mu: float, rho: float, max_goals: int = config.MAX_GOALS) -> dict:
    g = np.arange(max_goals + 1)
    m = np.outer(poisson.pmf(g, lam), poisson.pmf(g, mu))
    m[0, 0] *= 1 - lam * mu * rho
    m[0, 1] *= 1 + lam * rho
    m[1, 0] *= 1 + mu * rho
    m[1, 1] *= 1 - rho
    m = np.clip(m, 0, None)
    m /= m.sum()
    total = np.add.outer(g, g)
    return {"p_home": np.tril(m, -1).sum(), "p_draw": np.trace(m), "p_away": np.triu(m, 1).sum(),
            "p_over25": m[total > 2.5].sum()}


def _nb_over(mean: float, phi: float, line: float) -> float:
    """P(X > line) για negative binomial με E=mean, Var=phi*mean (phi<=1 -> Poisson)."""
    if phi <= 1.0001:
        return float(poisson.sf(np.floor(line), mean))
    n = mean / (phi - 1)
    p = n / (n + mean)
    return float(nbinom.sf(np.floor(line), n, p))


def _swap(df: pd.DataFrame, h: str, a: str) -> pd.DataFrame:
    """Αντιγράφει τις στήλες h/a στα hg/ag ώστε να χρησιμοποιηθεί το ίδιο μοντέλο για άλλα μεγέθη."""
    out = df.dropna(subset=[h, a]).copy()
    out["hg"], out["ag"] = out[h].astype(int), out[a].astype(int)
    return out


def walk_forward_multi(matches: pd.DataFrame, test_season: str, refit_days: int = 7,
                       min_team_matches: int = config.MIN_TEAM_MATCHES,
                       xi: float = config.XI, l2: float = config.L2) -> pd.DataFrame:
    """Out-of-sample προβλέψεις για τη test σεζόν: γκολ, SoT, υβριδικό και κόρνερ."""
    rows = []
    for league, lg in matches.groupby("league"):
        lg = lg.sort_values("date")
        test = lg[lg.season == test_season]
        train_all = lg[lg.season < test_season]
        if test.empty or train_all.empty:
            continue
        start, end_all = test.date.min(), test.date.max()
        prev = {}
        while start <= end_all:
            end = start + pd.Timedelta(days=refit_days)
            batch = test[(test.date >= start) & (test.date < end)]
            if not batch.empty:
                train = lg[lg.date < start]
                days = (start - train.date).dt.days.to_numpy(float)
                w = np.exp(-xi * days)
                mg = DixonColes(xi=xi, l2=l2).fit(train, start, prev.get("g"))
                ms = DixonColes(xi=xi, l2=l2, fit_rho=False).fit(_swap(train, "hst", "ast"), start, prev.get("s"))
                mc = DixonColes(xi=xi, l2=l2, fit_rho=False).fit(_swap(train, "hc", "ac"), start, prev.get("c"))
                prev = {"g": mg, "s": ms, "c": mc}
                # Μετατροπή SoT -> γκολ, ξεχωριστά για γηπεδούχους/φιλοξενούμενους (σταθμισμένα)
                conv_h = np.sum(w * train.hg) / np.sum(w * train.hst)
                conv_a = np.sum(w * train.ag) / np.sum(w * train.ast)
                # Υπερδιασπορά συνολικών κόρνερ στο training (με τις προβλέψεις του μοντέλου)
                tr_c = _swap(train, "hc", "ac")
                known = tr_c.home.isin(mc.teams) & tr_c.away.isin(mc.teams)
                tr_c = tr_c[known]
                exp_tot = np.array([sum(mc.expected_goals(h, a)) for h, a in zip(tr_c.home, tr_c.away)])
                wt = np.exp(-xi * (start - tr_c.date).dt.days.to_numpy(float))
                act_tot = (tr_c.hg + tr_c.ag).to_numpy(float)
                phi = max(1.0, np.sum(wt * (act_tot - exp_tot) ** 2 / exp_tot) / np.sum(wt))
                base_mean = np.sum(wt * act_tot) / np.sum(wt)
                phi_base = max(1.0, np.sum(wt * (act_tot - base_mean) ** 2) / np.sum(wt) / base_mean)
                for _, r in batch.iterrows():
                    if not all(m.knows(t, min_team_matches) for m in (mg, ms, mc) for t in (r.home, r.away)):
                        continue
                    lg_, mu_g = mg.expected_goals(r.home, r.away)
                    ls, mu_s = ms.expected_goals(r.home, r.away)
                    xg_s = (ls * conv_h, mu_s * conv_a)
                    xg_h = (np.sqrt(lg_ * xg_s[0]), np.sqrt(mu_g * xg_s[1]))
                    rho = mg.params["rho"]
                    row = {**r.to_dict(), "xg_goals_h": lg_, "xg_goals_a": mu_g, "xsot_h": ls, "xsot_a": mu_s,
                           "xg_sot_h": xg_s[0], "xg_sot_a": xg_s[1]}
                    for name, (a, b) in (("goals", (lg_, mu_g)), ("sot", xg_s), ("hybrid", xg_h)):
                        for k, v in _probs_from_xg(a, b, rho).items():
                            row[f"{name}_{k}"] = v
                    ch, ca = mc.expected_goals(r.home, r.away)
                    row.update({"xc_h": ch, "xc_a": ca, "xc_total": ch + ca, "corner_phi": phi,
                                "xc_base": base_mean, "corner_phi_base": phi_base})
                    for line in CORNER_LINES:
                        row[f"c_over_{line}"] = _nb_over(ch + ca, phi, line)
                        row[f"cbase_over_{line}"] = _nb_over(base_mean, phi_base, line)
                    rows.append(row)
            start = end
    return pd.DataFrame(rows)


# ---------------- αξιολόγηση ----------------
def _ll_multi(p, y):
    return float(-(y * np.log(np.clip(p, 1e-9, 1))).sum(axis=1).mean())


def _ll_bin(p, y):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def goal_models_report(preds: pd.DataFrame, weights=(0, 0.1, 0.2, 0.3, 0.5, 1.0)) -> pd.DataFrame:
    """Για κάθε μοντέλο: log-loss μόνο του, και log-loss όταν αναμειγνύεται με την αγορά.
    Αν κάποιο βάρος w > 0 δίνει μικρότερο log-loss από την αγορά μόνη της, το μοντέλο έχει
    πληροφορία που η αγορά δεν έχει πλήρως ενσωματώσει."""
    d = preds.dropna(subset=["odds_h", "odds_d", "odds_a", "odds_over", "odds_under"])
    y = np.column_stack([d.hg > d.ag, d.hg == d.ag, d.hg < d.ag]).astype(float)
    yo = (d.hg + d.ag > 2.5).to_numpy(float)
    pk = devig(d[["odds_h", "odds_d", "odds_a"]].to_numpy(float))
    ko = devig(d[["odds_over", "odds_under"]].to_numpy(float))[:, 0]
    rows = []
    for name in MODELS:
        pm = d[[f"{name}_p_home", f"{name}_p_draw", f"{name}_p_away"]].to_numpy()
        mo = d[f"{name}_p_over25"].to_numpy()
        for w in weights:
            rows.append({"μοντέλο": name, "βάρος_μοντέλου": w, "αγώνες": len(d),
                         "log_loss_1X2": _ll_multi(w * pm + (1 - w) * pk, y),
                         "log_loss_OU25": _ll_bin(w * mo + (1 - w) * ko, yo)})
    out = pd.DataFrame(rows)
    base = out[out.βάρος_μοντέλου == 0].iloc[0]
    out["βελτίωση_1X2_vs_αγορά"] = base.log_loss_1X2 - out.log_loss_1X2
    out["βελτίωση_OU_vs_αγορά"] = base.log_loss_OU25 - out.log_loss_OU25
    return out


def corners_report(preds: pd.DataFrame) -> pd.DataFrame:
    """Κόρνερ: μοντέλο ομάδων vs baseline (μέσος όρος λίγκας)."""
    out = []
    preds = preds.dropna(subset=["hc", "ac"])
    for name, d in [("ΣΥΝΟΛΟ", preds)] + sorted(preds.groupby("league")):
        actual = (d.hc + d.ac).to_numpy(float)
        row = {"λίγκα": name, "αγώνες": len(d), "μέσα_κόρνερ_πραγματικά": actual.mean(),
               "μέσα_κόρνερ_πρόβλεψη": d.xc_total.mean(),
               "MAE_μοντέλο": np.abs(actual - d.xc_total).mean(),
               "MAE_baseline": np.abs(actual - d.xc_base).mean(),
               "συσχέτιση_πρόβλεψης_με_πραγματικά": np.corrcoef(d.xc_total, actual)[0, 1]}
        for line in CORNER_LINES:
            y = (actual > line).astype(float)
            row[f"LL_over{line}_μοντέλο"] = _ll_bin(d[f"c_over_{line}"].to_numpy(), y)
            row[f"LL_over{line}_baseline"] = _ll_bin(d[f"cbase_over_{line}"].to_numpy(), y)
        out.append(row)
    return pd.DataFrame(out)


def fit_corner_shrink(preds: pd.DataFrame) -> float:
    """Συντελεστής k: πόσο από την απόκλιση (πρόβλεψη - μέσος λίγκας) επιβεβαιώνεται στην πράξη.
    OLS κλίση: πραγματικά - μέσος = k * (πρόβλεψη - μέσος). k=1 -> σωστή κλίμακα, k<1 -> υπερβολική σιγουριά."""
    d = preds.dropna(subset=["hc", "ac"])
    x = (d.xc_total - d.xc_base).to_numpy()
    y = (d.hc + d.ac - d.xc_base).to_numpy()
    return float(np.sum(x * y) / np.sum(x * x))


def apply_corner_shrink(preds: pd.DataFrame, k: float) -> pd.DataFrame:
    """Μαζεύει τις προβλέψεις κόρνερ προς τον μέσο όρο της λίγκας και ξαναϋπολογίζει τις πιθανότητες."""
    d = preds.copy()
    d["xc_total"] = d.xc_base + k * (d.xc_total - d.xc_base)
    for line in CORNER_LINES:
        d[f"c_over_{line}"] = [_nb_over(m, phi, line) for m, phi in zip(d.xc_total, d.corner_phi_base)]
    return d


def corners_calibration(preds: pd.DataFrame, line: float = 9.5) -> pd.DataFrame:
    preds = preds.dropna(subset=["hc", "ac"])
    p = preds[f"c_over_{line}"].to_numpy()
    y = ((preds.hc + preds.ac) > line).astype(float).to_numpy()
    bins = pd.cut(p, [0, .3, .4, .5, .6, .7, 1], include_lowest=True)
    t = pd.DataFrame({"bin": bins, "p": p, "y": y}).groupby("bin", observed=True).agg(
        αγώνες=("p", "size"), μέση_πρόβλεψη=("p", "mean"), πραγματικό=("y", "mean"))
    return t.reset_index().assign(bin=lambda x: x["bin"].astype(str), γραμμή=line)


def corners_extremes(preds: pd.DataFrame) -> pd.DataFrame:
    """Οι αγώνες όπου το μοντέλο αποκλίνει πολύ από τον μέσο όρο: εκεί θα υπήρχε value αν υπάρχει."""
    d = preds.dropna(subset=["hc", "ac"]).copy()
    d["απόκλιση_από_μέσο"] = d.xc_total - d.xc_base
    d["κόρνερ_πραγματικά"] = d.hc + d.ac
    d["ομάδα_απόκλισης"] = pd.qcut(d["απόκλιση_από_μέσο"], 5,
                                    labels=["πολύ λιγότερα", "λιγότερα", "μέσο", "περισσότερα", "πολύ περισσότερα"])
    return d.groupby("ομάδα_απόκλισης", observed=True).agg(
        αγώνες=("xc_total", "size"), πρόβλεψη_μοντέλου=("xc_total", "mean"), μέσος_λίγκας=("xc_base", "mean"),
        πραγματικά=("κόρνερ_πραγματικά", "mean"), over95_πρόβλεψη=("c_over_9.5", "mean"),
        over95_πραγματικό=("κόρνερ_πραγματικά", lambda x: (x > 9.5).mean())).reset_index()


class CornerPredictor:
    """Πρόβλεψη συνολικών κόρνερ για μια λίγκα (για χρήση στο predict)."""

    def __init__(self, train: pd.DataFrame, ref_date, shrink: float = config.CORNER_SHRINK,
                 xi: float = config.XI, l2: float = config.L2):
        ref_date = pd.Timestamp(ref_date)
        tr = _swap(train, "hc", "ac")
        self.model = DixonColes(xi=xi, l2=l2, fit_rho=False).fit(tr, ref_date)
        w = np.exp(-xi * (ref_date - tr.date).dt.days.to_numpy(float))
        tot = (tr.hg + tr.ag).to_numpy(float)
        self.base = float(np.sum(w * tot) / np.sum(w))
        self.phi = max(1.0, float(np.sum(w * (tot - self.base) ** 2) / np.sum(w) / self.base))
        self.shrink = shrink

    def predict(self, home: str, away: str, min_matches: int = config.MIN_TEAM_MATCHES) -> dict | None:
        if not (self.model.knows(home, min_matches) and self.model.knows(away, min_matches)):
            return None
        h, a = self.model.expected_goals(home, away)
        xc = self.base + self.shrink * (h + a - self.base)
        out = {"xc_total": xc, "xc_league_avg": self.base}
        for line in CORNER_LINES:
            p = _nb_over(xc, self.phi, line)
            out[f"corners_over_{line}"] = p
            out[f"fair_corners_over_{line}"] = 1 / p if p > 0 else np.inf
            out[f"fair_corners_under_{line}"] = 1 / (1 - p) if p < 1 else np.inf
        return out


def run_all(matches: pd.DataFrame, folds: list[tuple[list[str], str]]) -> dict:
    """Τρέχει όλα τα πειράματα σε διαδοχικά folds. Το k των κόρνερ εκτιμάται στο πρώτο fold
    και εφαρμόζεται στα επόμενα (out-of-sample)."""
    goal_tables, corner_tables, calib, extremes, preds_all = [], [], [], [], []
    k = None
    for train, test in folds:
        p = walk_forward_multi(matches[matches.season.isin(train + [test])], test)
        g = goal_models_report(p).assign(test_σεζόν=test)
        goal_tables.append(g)
        raw = corners_report(p).assign(test_σεζόν=test, εκδοχή="χωρίς διόρθωση")
        corner_tables.append(raw)
        if k is not None:
            ps = apply_corner_shrink(p, k)
            corner_tables.append(corners_report(ps).assign(test_σεζόν=test, εκδοχή=f"με διόρθωση k={k:.3f}"))
            calib.append(corners_calibration(ps).assign(test_σεζόν=test))
        else:
            calib.append(corners_calibration(p).assign(test_σεζόν=test, σημείωση="χωρίς διόρθωση"))
        extremes.append(corners_extremes(p).assign(test_σεζόν=test))
        k_fold = fit_corner_shrink(p)
        k = k_fold if k is None else k
        preds_all.append(p.assign(test_σεζόν=test))
    return {"goals": pd.concat(goal_tables, ignore_index=True),
            "corners": pd.concat(corner_tables, ignore_index=True),
            "corners_calibration": pd.concat(calib, ignore_index=True),
            "corners_by_deviation": pd.concat(extremes, ignore_index=True),
            "predictions": pd.concat(preds_all, ignore_index=True), "k": k}
