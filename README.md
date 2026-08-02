# trading-coach

A trade auditor built in five layers. **This is Layer 1: the data** — real
intraday price bars plus the trade log.

## Layer 1 modules

| File | What it does |
| --- | --- |
| `bars.py` | `fetch_bars(symbols, ...)` pulls real intraday bars from Yahoo: the last ~60 days of 5-minute bars **plus** the last ~7 days again at 1-minute (Yahoo's caps). Both are merged into one frame — finer bars win where they exist. Regular session only (09:30–16:00 US/Eastern, naive Eastern timestamps), deduped, sorted, saved to `bars.csv`. Symbols/ranges that can't be fetched are skipped with plain warnings, never a crash. `fetch_bars_for_trades(trades)` reads the distinct symbols + date range straight from an uploaded trade log and fetches exactly that. |
| `trades_io.py` | `load_trades(path)` loads a NinjaTrader-style export. Required: `Instrument, Market pos., Qty, Entry price, Exit price, Entry time, Exit time, Profit`. Optional columns (`Trade number, Account, Strategy, Commission, MAE, MFE, ETD, Bars, Exit name, Cum. net profit`) are backfilled when missing. Timestamps go through `pd.to_datetime` so any sane format works; currency-formatted numbers (`$1,234.50`, `($20.00)`) parse fine. A bad file raises one clear error line. |
| `fetch_and_gen.py` | Demo data: fetches the real bars, then generates a believable **winning** long-biased momentum day-trader from them — 300+ trades across all ~60 sessions, written to `trades.csv` in the exact NinjaTrader column format. Every entry price is a real bar's open and every exit a real bar's close, at their real timestamps; only *which* bar a trade exits at is chosen. Realistic leaks are baked in for the upper layers to find: occasional oversizing right after a loss, a few late-session entries, some `Manual` exits. The script ends by reloading both CSVs and verifying every trade against the tape. |

## Quick start

```bash
pip install -r requirements.txt
python fetch_and_gen.py            # default: SPY QQQ NVDA TSLA
python fetch_and_gen.py SPY QQQ    # or your own symbols
```

This writes `bars.csv` and `trades.csv` and prints a verification summary.
(Needs network access to `query1/query2.finance.yahoo.com`.)

## Tests (no network needed)

```bash
python tests/test_layer1.py
```

Runs 34 offline checks against synthetic tapes shaped exactly like the
yfinance output: session filtering, the 5m/1m hybrid merge, the trade
generator (count, cadence, profitability, leaks, bar-level verification)
and the NinjaTrader loader round trip.
