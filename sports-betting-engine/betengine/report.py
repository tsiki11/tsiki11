"""Εξαγωγή αποτελεσμάτων σε Excel με βασική μορφοποίηση."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

PCT_HINTS = ("p_", "edge", "roi", "hit_rate", "clv", "%", "margin", "kelly", "απόκλιση",
             "πιθανότητα", "συχνότητα")
HEADER_FILL = PatternFill("solid", fgColor="1F3864")
GOOD_FILL = PatternFill("solid", fgColor="E2EFDA")


def _is_pct(col: str) -> bool:
    c = col.lower()
    return any(h in c for h in PCT_HINTS) and "odds" not in c and "μονάδες" not in c


def write_excel(path: Path, sheets: dict[str, pd.DataFrame], highlight: dict[str, str] | None = None) -> Path:
    """sheets: {όνομα φύλλου: DataFrame}. highlight: {φύλλο: στήλη bool} -> πράσινες γραμμές."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    highlight = highlight or {}
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for name, df in sheets.items():
            if df is None or df.empty:
                df = pd.DataFrame({"info": ["Δεν υπάρχουν δεδομένα"]})
            df = df.copy()
            for c in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[c]):
                    df[c] = df[c].dt.date
            sheet = name[:31]
            df.to_excel(xw, sheet_name=sheet, index=False)
            ws = xw.sheets[sheet]
            ws.freeze_panes = "A2"
            for j, col in enumerate(df.columns, start=1):
                cell = ws.cell(row=1, column=j)
                cell.font = Font(bold=True, color="FFFFFF")
                cell.fill = HEADER_FILL
                cell.alignment = Alignment(wrap_text=True, vertical="center")
                letter = get_column_letter(j)
                width = max([len(str(col))] + [len(str(v)) for v in df[col].head(200)]) + 2
                ws.column_dimensions[letter].width = min(max(width, 8), 40)
                if pd.api.types.is_float_dtype(df[col]):
                    fmt = "0.0%" if _is_pct(str(col)) else "0.00"
                    for i in range(2, len(df) + 2):
                        ws.cell(row=i, column=j).number_format = fmt
            hcol = highlight.get(name)
            if hcol and hcol in df.columns:
                for i, flag in enumerate(df[hcol].tolist(), start=2):
                    if bool(flag):
                        for j in range(1, len(df.columns) + 1):
                            ws.cell(row=i, column=j).fill = GOOD_FILL
            ws.auto_filter.ref = ws.dimensions
    return path
