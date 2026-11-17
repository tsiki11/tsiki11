"""Στατιστικά ομάδων: βαθμολογία, γκολ, σουτ, κόρνερ, φόρμα, έδρα/εκτός."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _long(matches: pd.DataFrame) -> pd.DataFrame:
    """Μετατρέπει κάθε αγώνα σε 2 γραμμές (μία ανά ομάδα), για εύκολο groupby."""
    cols = {"hs": "as_", "hst": "ast", "hc": "ac", "hy": "ay", "hr": "ar"}
    home = pd.DataFrame({"date": matches.date, "league": matches.league, "team": matches.home,
                         "opp": matches.away, "venue": "H", "gf": matches.hg, "ga": matches.ag})
    away = pd.DataFrame({"date": matches.date, "league": matches.league, "team": matches.away,
                         "opp": matches.home, "venue": "A", "gf": matches.ag, "ga": matches.hg})
    for h, a in cols.items():
        name = h[1:]  # s, st, c, y, r
        home[f"{name}_for"], home[f"{name}_against"] = matches[h], matches[a]
        away[f"{name}_for"], away[f"{name}_against"] = matches[a], matches[h]
    df = pd.concat([home, away], ignore_index=True).sort_values("date")
    df["pts"] = np.select([df.gf > df.ga, df.gf == df.ga], [3, 1], 0)
    df["res"] = np.select([df.gf > df.ga, df.gf == df.ga], ["W", "D"], "L")
    return df


def team_table(matches: pd.DataFrame, form_n: int = 5) -> pd.DataFrame:
    """Πίνακας ανά ομάδα για τους αγώνες που δίνονται (συνήθως: τρέχουσα σεζόν μιας λίγκας)."""
    if matches.empty:
        return pd.DataFrame()
    df = _long(matches)
    g = df.groupby(["league", "team"])
    t = g.agg(αγώνες=("gf", "size"), βαθμοί=("pts", "sum"), γκολ_υπέρ=("gf", "sum"), γκολ_κατά=("ga", "sum"))
    t["διαφορά"] = t["γκολ_υπέρ"] - t["γκολ_κατά"]
    t["βαθμοί_ανά_αγώνα"] = t["βαθμοί"] / t["αγώνες"]
    t["GF_ανά_αγώνα"] = t["γκολ_υπέρ"] / t["αγώνες"]
    t["GA_ανά_αγώνα"] = t["γκολ_κατά"] / t["αγώνες"]
    means = g[["s_for", "s_against", "st_for", "st_against", "c_for", "c_against"]].mean()
    means.columns = ["σουτ_υπέρ", "σουτ_κατά", "εντός_εστίας_υπέρ", "εντός_εστίας_κατά", "κόρνερ_υπέρ", "κόρνερ_κατά"]
    t = t.join(means)
    # Conversion: γκολ ανά σουτ εντός εστίας (υψηλό = πιθανή "τύχη" που μπορεί να μη διαρκέσει)
    t["γκολ_ανά_SoT"] = t["γκολ_υπέρ"] / g["st_for"].sum().replace(0, np.nan)
    t["over25_%"] = g.apply(lambda x: ((x.gf + x.ga) > 2.5).mean(), include_groups=False)
    t["btts_%"] = g.apply(lambda x: ((x.gf > 0) & (x.ga > 0)).mean(), include_groups=False)
    t["clean_sheet_%"] = g.apply(lambda x: (x.ga == 0).mean(), include_groups=False)
    for v, label in (("H", "έδρα"), ("A", "εκτός")):
        sub = df[df.venue == v].groupby(["league", "team"])
        t[f"βαθμοί_ανά_αγώνα_{label}"] = sub["pts"].mean()
    t["φόρμα"] = g["res"].apply(lambda r: "".join(r.tail(form_n)))
    t["βαθμοί_τελ_" + str(form_n)] = g["pts"].apply(lambda p: p.tail(form_n).sum())
    t = t.reset_index().sort_values(["league", "βαθμοί", "διαφορά", "γκολ_υπέρ"], ascending=[True, False, False, False])
    t.insert(2, "θέση", t.groupby("league").cumcount() + 1)
    return t.reset_index(drop=True)


def head_to_head(matches: pd.DataFrame, a: str, b: str, n: int = 6) -> pd.DataFrame:
    m = matches[((matches.home == a) & (matches.away == b)) | ((matches.home == b) & (matches.away == a))]
    return m.sort_values("date").tail(n)[["date", "league", "home", "away", "hg", "ag"]]
