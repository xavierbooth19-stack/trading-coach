"""Layer 1 — load a trade export.

load_trades() reads the CSV, validates the required columns, parses
timestamps and currency-formatted numbers, and backfills any optional
columns that are missing. A bad file produces one clear error line.

NinjaTrader exports load as-is. TopstepX/ProjectX-style exports (and
similar platforms) are recognized by header synonyms — ContractName,
EnteredAt/ExitedAt, Size, Type, PnL, Fees, ... — mapped onto the same
schema, with UTC timestamps converted to naive US/Eastern and Buy/Sell
sides normalized to Long/Short.
"""

from __future__ import annotations

import re

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


# Header synonyms (normalized: lowercase, alphanumerics only) for other
# platforms' exports — TopstepX/ProjectX, Tradovate-style, etc.
COLUMN_SYNONYMS = {
    "Instrument": ("contractname", "contract", "symbol", "symbolname", "product"),
    "Market pos.": ("type", "side", "direction", "position", "marketposition", "buysell"),
    "Qty": ("size", "quantity", "positionsize", "contracts", "filledqty"),
    "Entry price": ("entryprice", "avgentryprice", "priceentry", "buyprice", "openprice"),
    "Exit price": ("exitprice", "avgexitprice", "priceexit", "sellprice", "closeprice"),
    "Entry time": ("enteredat", "entrytime", "opentime", "openedat", "entrydate", "boughttimestamp"),
    "Exit time": ("exitedat", "exittime", "closetime", "closedat", "exitdate", "soldtimestamp"),
    "Profit": ("pnl", "pl", "netpnl", "grosspnl", "profitloss", "realizedpnl", "netprofit"),
    "Commission": ("fees", "fee", "totalfees", "commissions"),
    "Trade number": ("id", "tradeid", "tradenumber"),
}

SIDE_MAP = {"long": "Long", "buy": "Long", "b": "Long", "bot": "Long",
            "short": "Short", "sell": "Short", "s": "Short", "sld": "Short"}

EASTERN = "US/Eastern"


class TradesFileError(ValueError):
    """Raised with a single human-readable line describing what's wrong."""


def load_trades(path) -> pd.DataFrame:
    """Load a NinjaTrader-style CSV into a clean, typed DataFrame."""
    try:
        trades = pd.read_csv(path)
    except Exception as exc:
        raise TradesFileError(f"Could not read trades file '{path}': {exc}") from None

    trades.columns = [str(c).strip() for c in trades.columns]
    trades = _map_platform_columns(trades)

    missing = [c for c in REQUIRED_COLUMNS if c not in trades.columns]
    if missing:
        raise TradesFileError(
            f"Trades file '{path}' is missing required column(s): {', '.join(missing)}"
        )
    if trades.empty:
        raise TradesFileError(f"Trades file '{path}' contains no trades")

    for col in ("Entry time", "Exit time"):
        parsed = _parse_times(trades[col])
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

    sides = trades["Market pos."].astype(str).str.strip().str.lower()
    trades["Market pos."] = sides.map(SIDE_MAP).fillna(sides.str.capitalize())
    unknown = ~trades["Market pos."].isin(["Long", "Short"])
    if unknown.any():
        raise TradesFileError(
            f"Trades file '{path}': {int(unknown.sum())} row(s) have an unrecognized "
            f"side in 'Market pos.' (expected Long/Short or Buy/Sell)"
        )
    trades["Instrument"] = trades["Instrument"].astype(str).str.strip()

    trades = trades.sort_values("Entry time", kind="stable").reset_index(drop=True)

    for col, default in OPTIONAL_COLUMNS.items():
        if col not in trades.columns:
            trades[col] = default
    numbers = pd.to_numeric(trades["Trade number"], errors="coerce")
    if numbers.isna().any() or numbers.duplicated().any():
        trades["Trade number"] = np.arange(1, len(trades) + 1)  # GUID/absent ids -> renumber
    else:
        trades["Trade number"] = numbers.astype("int64")
    if trades["Cum. net profit"].isna().all():
        trades["Cum. net profit"] = (trades["Profit"] - trades["Commission"]).cumsum()
    trades["Bars"] = pd.to_numeric(trades["Bars"], errors="coerce").fillna(0).astype("int64")
    trades["Qty"] = trades["Qty"].astype("int64")

    return trades[COLUMN_ORDER]


def _map_platform_columns(trades: pd.DataFrame) -> pd.DataFrame:
    """Rename TopstepX/ProjectX-style headers onto the NinjaTrader schema.

    Only fills gaps: a column that already has its NinjaTrader name is
    never touched, and the first matching synonym wins.
    """
    normalized = {re.sub(r"[^a-z0-9]", "", str(c).lower()): c for c in trades.columns}
    rename = {}
    for target, synonyms in COLUMN_SYNONYMS.items():
        if target in trades.columns:
            continue
        for syn in synonyms:
            src = normalized.get(syn)
            if src is not None and src not in rename:
                rename[src] = target
                break
    return trades.rename(columns=rename) if rename else trades


def _parse_times(series: pd.Series) -> pd.Series:
    """Timestamps via pd.to_datetime; tz-aware values (TopstepX exports
    are UTC ISO) are converted to naive US/Eastern to match the bars."""
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.isna().any():  # exports sometimes mix timestamp formats
        parsed = pd.to_datetime(series, format="mixed", errors="coerce")
    if parsed.isna().any():  # mixed/UTC offsets need utc=True to parse at all
        utc = pd.to_datetime(series, errors="coerce", utc=True)
        if utc.isna().sum() < parsed.isna().sum():
            parsed = utc
    if getattr(parsed.dt, "tz", None) is not None:
        parsed = parsed.dt.tz_convert(EASTERN).dt.tz_localize(None)
    return parsed


def _to_number(series: pd.Series) -> pd.Series:
    """Parse numbers that may be currency-formatted: '$1,234.50', '($20.00)', '-$5'."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    text = series.astype(str).str.strip()
    negative = text.str.startswith("(") & text.str.endswith(")")
    text = text.str.replace(r"[()$,\s]", "", regex=True).str.replace("%", "")
    numbers = pd.to_numeric(text, errors="coerce")
    return numbers.where(~negative, -numbers)
