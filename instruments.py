"""Futures contract identification and dollar point values.

One shared place to answer two questions so every layer agrees:
  - which contract root is this instrument string? ("MES 06-25", "ESH25",
    "F.US.EP", "CON.F.US.MNQ.H25" -> MES / ES / ES / MNQ)
  - how many dollars is a 1.0 price-unit move worth per contract?

Equities (and anything unrecognized) get a point value of 1.0, which
keeps all the existing stock math exact.
"""

from __future__ import annotations

import re

# Dollars per 1.0 price-unit move, per contract (CME/CBOT/NYMEX/COMEX).
POINT_VALUES = {
    # equity index
    "ES": 50.0, "MES": 5.0,
    "NQ": 20.0, "MNQ": 2.0,
    "YM": 5.0, "MYM": 0.5,
    "RTY": 50.0, "M2K": 5.0,
    "NKD": 5.0,
    # energy
    "CL": 1000.0, "MCL": 100.0, "QM": 500.0,
    "NG": 10000.0, "QG": 2500.0,
    "RB": 42000.0, "HO": 42000.0,
    # metals
    "GC": 100.0, "MGC": 10.0,
    "SI": 5000.0, "SIL": 1000.0,
    "HG": 25000.0, "MHG": 2500.0,
    "PL": 50.0,
    # rates
    "ZB": 1000.0, "ZN": 1000.0, "ZF": 1000.0, "ZT": 2000.0,
    "UB": 1000.0, "TN": 1000.0,
    # FX
    "6E": 125000.0, "6B": 62500.0, "6A": 100000.0, "6C": 100000.0,
    "6J": 12500000.0, "6S": 125000.0, "M6E": 12500.0, "M6A": 10000.0,
}

# CQG/ProjectX (TopstepX) aliases for CME roots.
ROOT_ALIASES = {"EP": "ES", "ENQ": "NQ", "EMD": "ES", "QO": "GC", "MQO": "MGC"}

MONTH_CODES = "FGHJKMNQUVXZ"

_NT_RE = re.compile(r"^([A-Z0-9]{1,4})\s+\d{2}-\d{2}$")               # "MES 06-25"
_CODE_RE = re.compile(rf"^/?([A-Z0-9]{{1,4}}?)([{MONTH_CODES}])(\d{{1,2}})$")  # "ESH25", "/MNQZ5"
_DOTTED_RE = re.compile(r"(?:CON\.)?F\.[A-Z]{2}\.([A-Z0-9]{1,4})(?:\.[A-Z0-9]+)*$")  # "F.US.EP", "CON.F.US.MES.H25"


def contract_root(instrument) -> str:
    """The futures root of an instrument string, or the string unchanged.

    Recognizes NinjaTrader ("ES 03-26"), month-code ("ESH25", "/MNQZ5"),
    and CQG/ProjectX dotted forms ("F.US.EP", "CON.F.US.MES.H25"), and
    resolves CQG aliases (EP -> ES). Equities pass through untouched.
    """
    text = str(instrument).strip().upper()
    for pattern in (_NT_RE, _DOTTED_RE, _CODE_RE):
        m = pattern.match(text)
        if m:
            root = m.group(1)
            root = ROOT_ALIASES.get(root, root)
            # month-code regex can swallow a bare equity like "F" or "GE";
            # only trust it when the root is a known future
            if pattern is _CODE_RE and root not in POINT_VALUES:
                continue
            return root
    return ROOT_ALIASES.get(text, text) if text in ROOT_ALIASES or text in POINT_VALUES else str(instrument).strip()


def is_futures(instrument) -> bool:
    return contract_root(instrument) in POINT_VALUES


def point_value(instrument) -> float:
    """Dollars per 1.0 price move per contract; 1.0 for equities/unknown."""
    return POINT_VALUES.get(contract_root(instrument), 1.0)


def yahoo_symbol(instrument) -> str:
    """Yahoo ticker: continuous contract for futures, unchanged otherwise."""
    root = contract_root(instrument)
    if root in POINT_VALUES:
        return f"{root}=F"
    return str(instrument).strip()
