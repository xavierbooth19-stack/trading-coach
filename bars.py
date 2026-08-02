"""Layer 1 — real intraday price bars from Yahoo Finance.

fetch_bars() pulls a hybrid feed for any list of symbols:

  * the last ~60 days of 5-minute bars (Yahoo's cap for 5m data), plus
  * the last ~7 days again at 1-minute (Yahoo's cap for 1m data).

Both resolutions are merged into one frame; where a 5m and a 1m bar share
the same timestamp the finer 1m bar wins. Only the regular US session
(09:30-16:00 Eastern) is kept, and timestamps come back as naive Eastern.

A symbol or date range that can't be fetched is skipped with a plain
warning string — a data gap never crashes the fetch.
"""

from __future__ import annotations

import datetime as dt
import pandas as pd
import yfinance as yf

import instruments

EASTERN = "US/Eastern"
SESSION_START = dt.time(9, 30)
SESSION_END = dt.time(16, 0)

# Yahoo's practical lookback windows (a little inside the hard caps).
FIVE_MIN_DAYS = 59
ONE_MIN_DAYS = 7

BARS_COLUMNS = ["instrument", "datetime", "open", "high", "low", "close", "volume"]


def to_yahoo_symbol(instrument: str) -> str:
    """Map a trade-log instrument name to a Yahoo ticker.

    Futures in any common form — NinjaTrader "ES 03-26", month-code
    "MNQZ5", CQG/TopstepX "F.US.EP" — map to Yahoo's continuous contract
    ("ES=F", "MNQ=F", ...). Anything else passes through unchanged.
    """
    return instruments.yahoo_symbol(instrument)


def fetch_bars(symbols, start=None, end=None, save_path="bars.csv"):
    """Fetch hybrid 5m + 1m regular-session bars for ``symbols``.

    Parameters
    ----------
    symbols : list of ticker strings, or dict of {instrument: yahoo_symbol}
        With a list, each entry is both the output instrument name and
        (via ``to_yahoo_symbol``) the ticker to fetch.
    start, end : optional date-likes bounding the wanted range (e.g. the
        span of an uploaded trade log). The range is clipped to what
        Yahoo can serve; anything entirely outside it is skipped with a
        warning.
    save_path : where to write the merged CSV, or None to skip saving.

    Returns
    -------
    (bars, warnings) : a DataFrame with columns
        instrument, datetime, open, high, low, close, volume
        (naive Eastern timestamps, deduped, sorted) and a list of plain
        warning strings for anything that could not be fetched.
    """
    if isinstance(symbols, dict):
        mapping = {str(k): str(v) for k, v in symbols.items()}
    else:
        mapping = {str(s): to_yahoo_symbol(s) for s in symbols}

    now = pd.Timestamp.now()
    start = None if start is None else pd.Timestamp(start)
    end = now if end is None else pd.Timestamp(end)
    if end.time() == dt.time(0, 0):  # a bare date means "through that day"
        end = end + pd.Timedelta(days=1)

    warnings: list[str] = []
    frames: list[pd.DataFrame] = []

    for instrument, ticker in mapping.items():
        got_any = False
        for interval, max_days in (("5m", FIVE_MIN_DAYS), ("1m", ONE_MIN_DAYS)):
            floor = now - pd.Timedelta(days=max_days)
            req_start = floor if start is None else max(start, floor)
            req_end = min(end, now)
            if req_start >= req_end:
                if interval == "5m" and start is not None:
                    warnings.append(
                        f"{instrument}: requested range {start.date()} → {end.date()} is "
                        f"older than Yahoo's ~{max_days}-day window for {interval} bars; skipped"
                    )
                continue
            if interval == "5m" and start is not None and start < floor:
                warnings.append(
                    f"{instrument}: 5m history clipped to {req_start.date()} "
                    f"(Yahoo keeps ~{max_days} days of 5m bars)"
                )
            try:
                raw = yf.download(
                    ticker,
                    start=req_start.floor("D"),
                    end=req_end,
                    interval=interval,
                    auto_adjust=False,
                    prepost=False,
                    progress=False,
                )
            except Exception as exc:  # network, delisted ticker, etc.
                warnings.append(f"{instrument}: {interval} download failed ({exc}); skipped")
                continue
            norm = _normalize(raw, instrument, interval)
            if norm.empty:
                if interval == "5m":
                    warnings.append(f"{instrument}: Yahoo returned no {interval} bars; skipped")
                continue
            got_any = True
            frames.append(norm)
        if not got_any:
            warnings.append(f"{instrument}: no usable bars at any resolution")

    bars = _merge(frames)
    if save_path and not bars.empty:
        bars.to_csv(save_path, index=False)
    return bars, warnings


def fetch_bars_for_trades(trades: pd.DataFrame, save_path="bars.csv"):
    """Fetch exactly the symbols + date range an uploaded trade log needs.

    ``trades`` is a frame from trades_io.load_trades. Instruments are
    mapped to Yahoo tickers (futures contracts collapse to one continuous
    ticker) and the fetch is bounded by the log's first entry / last exit.
    """
    mapping = {inst: to_yahoo_symbol(inst) for inst in trades["Instrument"].astype(str).unique()}
    start = pd.Timestamp(trades["Entry time"].min()).normalize()
    end = pd.Timestamp(trades["Exit time"].max()).normalize()
    return fetch_bars(mapping, start=start, end=end, save_path=save_path)


def load_bars(path="bars.csv") -> pd.DataFrame:
    """Read a bars.csv written by fetch_bars back into a typed frame."""
    bars = pd.read_csv(path)
    bars["datetime"] = pd.to_datetime(bars["datetime"])
    return bars


# ---------------------------------------------------------------- internals


def _normalize(raw: pd.DataFrame, instrument: str, interval: str) -> pd.DataFrame:
    """Yahoo download -> instrument/datetime/OHLCV rows, naive Eastern, RTH only."""
    if raw is None or raw.empty:
        return pd.DataFrame()
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):  # yfinance returns (Price, Ticker)
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]

    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_convert(EASTERN).tz_localize(None)
    df.index = idx

    times = df.index.time
    df = df[(times >= SESSION_START) & (times < SESSION_END)]
    df = df[df["close"].notna()]
    if df.empty:
        return pd.DataFrame()

    df = df.reset_index(names="datetime")
    df.insert(0, "instrument", instrument)
    df["volume"] = df["volume"].fillna(0).astype("int64")
    df["_interval"] = interval
    return df


def _merge(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Concat per-symbol frames; finer bars win on timestamp collisions."""
    if not frames:
        return pd.DataFrame(columns=BARS_COLUMNS)
    merged = pd.concat(frames, ignore_index=True)
    # "1m" sorts before "5m", so keep="first" keeps the finer bar.
    merged = merged.sort_values(["instrument", "datetime", "_interval"])
    merged = merged.drop_duplicates(["instrument", "datetime"], keep="first")
    merged = merged.drop(columns="_interval").reset_index(drop=True)
    return merged[BARS_COLUMNS]
