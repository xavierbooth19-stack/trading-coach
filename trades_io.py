"""Layer 1 — load a NinjaTrader-style trade export.

load_trades() reads the CSV, validates the required columns, parses
timestamps and currency-formatted numbers, and backfills any optional
columns that are missing. A bad file produces one clear error line.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "Instrument",
    "Market pos.",
    "Qty",
    "Entry price",
    "Exit price",
    "Entry time",
    "Exit time",
    "Profit",
]

# Optional columns and the default used when the export doesn't have them.
# None means "computed after load" (trade numbering, cumulative profit).
OPTIONAL_COLUMNS = {
    "Trade number": None,
    "Account": "Sim101",
    "Strategy": "Manual",
    "Commission": 0.0,
    "MAE": 0.0,
    "MFE": 0.0,
    "ETD": 0.0,
    "Bars": 0,
    "Exit name": "",
    "Cum. net profit": None,
}

NUMERIC_COLUMNS = ["Qty", "Entry price", "Exit price", "Profit",
                   "Commission", "MAE", "MFE", "ETD", "Cum. net profit"]

COLUMN_ORDER = [
    "Trade number", "Instrument", "Account", "Strategy", "Market pos.",
    "Qty", "Entry price", "Exit price", "Entry time", "Exit time",
    "Exit name", "Profit", "Cum. net profit", "Commission",
    "MAE", "MFE", "ETD", "Bars",
]


class TradesFileError(ValueError):
    """Raised with a single human-readable line describing what's wrong."""


def load_trades(path) -> pd.DataFrame:
    """Load a NinjaTrader-style CSV into a clean, typed DataFrame."""
    try:
        trades = pd.read_csv(path)
    except Exception as exc:
        raise TradesFileError(f"Could not read trades file '{path}': {exc}") from None

    trades.columns = [str(c).strip() for c in trades.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in trades.columns]
    if missing:
        raise TradesFileError(
            f"Trades file '{path}' is missing required column(s): {', '.join(missing)}"
        )
    if trades.empty:
        raise TradesFileError(f"Trades file '{path}' contains no trades")

    for col in ("Entry time", "Exit time"):
        parsed = pd.to_datetime(trades[col], errors="coerce")
        if parsed.isna().any():  # exports sometimes mix timestamp formats
            parsed = pd.to_datetime(trades[col], format="mixed", errors="coerce")
        bad = int(parsed.isna().sum())
        if bad:
            raise TradesFileError(
                f"Trades file '{path}': could not parse {bad} value(s) in '{col}'"
            )
        trades[col] = parsed

    for col in NUMERIC_COLUMNS:
        if col in trades.columns:
            trades[col] = _to_number(trades[col])
    for col in ("Qty", "Entry price", "Exit price", "Profit"):
        bad = int(trades[col].isna().sum())
        if bad:
            raise TradesFileError(
                f"Trades file '{path}': could not parse {bad} value(s) in '{col}'"
            )

    trades["Market pos."] = trades["Market pos."].astype(str).str.strip().str.capitalize()
    trades["Instrument"] = trades["Instrument"].astype(str).str.strip()

    trades = trades.sort_values("Entry time", kind="stable").reset_index(drop=True)

    for col, default in OPTIONAL_COLUMNS.items():
        if col not in trades.columns:
            trades[col] = default
    if trades["Trade number"].isna().all():
        trades["Trade number"] = np.arange(1, len(trades) + 1)
    if trades["Cum. net profit"].isna().all():
        trades["Cum. net profit"] = (trades["Profit"] - trades["Commission"]).cumsum()
    trades["Bars"] = pd.to_numeric(trades["Bars"], errors="coerce").fillna(0).astype("int64")
    trades["Qty"] = trades["Qty"].astype("int64")

    return trades[COLUMN_ORDER]


def _to_number(series: pd.Series) -> pd.Series:
    """Parse numbers that may be currency-formatted: '$1,234.50', '($20.00)', '-$5'."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    text = series.astype(str).str.strip()
    negative = text.str.startswith("(") & text.str.endswith(")")
    text = text.str.replace(r"[()$,\s]", "", regex=True).str.replace("%", "")
    numbers = pd.to_numeric(text, errors="coerce")
    return numbers.where(~negative, -numbers)
