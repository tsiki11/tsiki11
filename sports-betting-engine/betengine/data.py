"""Λήψη, cache και καθαρισμός δεδομένων από το football-data.co.uk.

Κάθε CSV έχει ένα αγώνα ανά γραμμή. Κρατάμε ένα καθαρό, σταθερό σχήμα:
    league, season, date, home, away, hg, ag, (στατιστικά), odds_*, close_*
"""
from __future__ import annotations

import time
import urllib.request
from datetime import date, datetime
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

# Στήλες αποδόσεων με σειρά προτίμησης (η πρώτη που υπάρχει και έχει τιμή κερδίζει).
# "Avg" = μέσος όρος αγοράς, B365 = Bet365, PS = Pinnacle (το πιο "έξυπνο" βιβλίο).
PRE_1X2 = [("AvgH", "AvgD", "AvgA"), ("BbAvH", "BbAvD", "BbAvA"), ("B365H", "B365D", "B365A")]
PRE_OU = [("Avg>2.5", "Avg<2.5"), ("BbAv>2.5", "BbAv<2.5"), ("B365>2.5", "B365<2.5")]
CLOSE_1X2 = [("PSCH", "PSCD", "PSCA"), ("AvgCH", "AvgCD", "AvgCA"), ("B365CH", "B365CD", "B365CA")]
CLOSE_OU = [("PC>2.5", "PC<2.5"), ("AvgC>2.5", "AvgC<2.5"), ("B365C>2.5", "B365C<2.5")]

STAT_COLS = {
    "HS": "hs", "AS": "as_", "HST": "hst", "AST": "ast",
    "HC": "hc", "AC": "ac", "HY": "hy", "AY": "ay", "HR": "hr", "AR": "ar",
}


def current_season_start(today: date | None = None) -> int:
    today = today or date.today()
    return today.year if today.month >= 7 else today.year - 1


def season_code(start_year: int) -> str:
    """2024 -> '2425'"""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def recent_seasons(n: int, today: date | None = None) -> list[str]:
    start = current_season_start(today)
    return [season_code(y) for y in range(start - n + 1, start + 1)]


def _fetch(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "betengine/0.1"})
    last_err = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                dest.write_bytes(r.read())
            return
        except Exception as e:  # pragma: no cover - δίκτυο
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Αποτυχία λήψης {url}: {last_err}")


def download(league: str, season: str, force: bool = False, max_age_hours: float = 12) -> Path:
    """Κατεβάζει ένα CSV (ή το παίρνει από cache). Η τρέχουσα σεζόν ανανεώνεται κάθε 12 ώρες."""
    dest = config.DATA_DIR / season / f"{league}.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    is_current = season == season_code(current_season_start())
    stale = dest.exists() and is_current and (time.time() - dest.stat().st_mtime) > max_age_hours * 3600
    if force or not dest.exists() or stale:
        _fetch(f"{config.BASE_URL}/mmz4281/{season}/{league}.csv", dest)
    return dest


def download_fixtures(force: bool = False, max_age_hours: float = 6) -> Path:
    dest = config.DATA_DIR / "fixtures.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if force or not dest.exists() or (time.time() - dest.stat().st_mtime) > max_age_hours * 3600:
        _fetch(f"{config.BASE_URL}/fixtures.csv", dest)
    return dest


def _read_csv(path: Path) -> pd.DataFrame:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    df = pd.read_csv(StringIO(text), on_bad_lines="skip", low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    return df.dropna(how="all")


def _pick(df: pd.DataFrame, candidates: list[tuple[str, ...]], names: list[str]) -> pd.DataFrame:
    """Για κάθε γραμμή παίρνει την πρώτη ομάδα στηλών που έχει πλήρεις τιμές."""
    out = pd.DataFrame(np.nan, index=df.index, columns=names)
    for cols in candidates:
        if not all(c in df.columns for c in cols):
            continue
        block = df[list(cols)].apply(pd.to_numeric, errors="coerce")
        block.columns = names
        missing = out.isna().any(axis=1) & block.notna().all(axis=1)
        out.loc[missing] = block.loc[missing].values
    return out


def _parse_dates(s: pd.Series) -> pd.Series:
    d = pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")
    d2 = pd.to_datetime(s, format="%d/%m/%y", errors="coerce")
    return d.fillna(d2)


def normalize(df: pd.DataFrame, season: str | None = None, results: bool = True) -> pd.DataFrame:
    out = pd.DataFrame({
        "league": df["Div"].astype(str).str.strip(),
        "season": season,
        "date": _parse_dates(df["Date"].astype(str).str.strip()),
        "home": df["HomeTeam"].astype(str).str.strip(),
        "away": df["AwayTeam"].astype(str).str.strip(),
        "time": df["Time"].astype(str).str.strip() if "Time" in df.columns else "",
    })
    if results:
        out["hg"] = pd.to_numeric(df.get("FTHG"), errors="coerce")
        out["ag"] = pd.to_numeric(df.get("FTAG"), errors="coerce")
        for src, dst in STAT_COLS.items():
            out[dst] = pd.to_numeric(df[src], errors="coerce") if src in df.columns else np.nan
    out = pd.concat([
        out,
        _pick(df, PRE_1X2, ["odds_h", "odds_d", "odds_a"]),
        _pick(df, PRE_OU, ["odds_over", "odds_under"]),
        _pick(df, CLOSE_1X2, ["close_h", "close_d", "close_a"]),
        _pick(df, CLOSE_OU, ["close_over", "close_under"]),
    ], axis=1)
    out = out.dropna(subset=["date"])
    if results:
        out = out.dropna(subset=["hg", "ag"])
        out[["hg", "ag"]] = out[["hg", "ag"]].astype(int)
    return out.reset_index(drop=True)


def load_matches(leagues: list[str], seasons: list[str], download_missing: bool = True) -> pd.DataFrame:
    frames = []
    for season in seasons:
        for lg in leagues:
            path = config.DATA_DIR / season / f"{lg}.csv"
            if download_missing:
                try:
                    path = download(lg, season)
                except RuntimeError as e:
                    print(f"  ! {e}")
                    continue
            if not path.exists():
                continue
            df = _read_csv(path)
            if df.empty or "HomeTeam" not in df.columns:
                continue
            frames.append(normalize(df, season))
    if not frames:
        raise RuntimeError("Δεν βρέθηκαν δεδομένα αγώνων.")
    return pd.concat(frames, ignore_index=True).sort_values(["date", "league"]).reset_index(drop=True)


def load_fixtures(leagues: list[str], from_date: date | None = None) -> pd.DataFrame:
    df = _read_csv(download_fixtures())
    df = df[df["Div"].astype(str).str.strip().isin(leagues)]
    if df.empty:
        return pd.DataFrame()
    fx = normalize(df, results=False)
    if from_date is not None:
        fx = fx[fx["date"] >= pd.Timestamp(from_date)]
    return fx.sort_values(["date", "league"]).reset_index(drop=True)
