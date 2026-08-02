# trading-coach

A trade auditor built in five layers. So far: **Layer 1 (the data)** — real
intraday price bars plus the trade log; **Layer 2 (the strategy)** —
structured rules the engine applies mechanically; **Layer 3 (the engine)** —
the deterministic behavioral audit and rule replay.

## Layer 1 modules

| File | What it does |
| --- | --- |
| `bars.py` | `fetch_bars(symbols, ...)` pulls real intraday bars from Yahoo: the last ~60 days of 5-minute bars **plus** the last ~7 days again at 1-minute (Yahoo's caps). Both are merged into one frame — finer bars win where they exist. Regular session only (09:30–16:00 US/Eastern, naive Eastern timestamps), deduped, sorted, saved to `bars.csv`. Symbols/ranges that can't be fetched are skipped with plain warnings, never a crash. `fetch_bars_for_trades(trades)` reads the distinct symbols + date range straight from an uploaded trade log and fetches exactly that. |
| `trades_io.py` | `load_trades(path)` loads a NinjaTrader-style export. Required: `Instrument, Market pos., Qty, Entry price, Exit price, Entry time, Exit time, Profit`. Optional columns (`Trade number, Account, Strategy, Commission, MAE, MFE, ETD, Bars, Exit name, Cum. net profit`) are backfilled when missing. Timestamps go through `pd.to_datetime` so any sane format works; currency-formatted numbers (`$1,234.50`, `($20.00)`) parse fine. A bad file raises one clear error line. |
| `fetch_and_gen.py` | Demo data: fetches the real bars, then generates a believable **winning** long-biased momentum day-trader from them — 300+ trades across all ~60 sessions, written to `trades.csv` in the exact NinjaTrader column format. Every entry price is a real bar's open and every exit a real bar's close, at their real timestamps; only *which* bar a trade exits at is chosen. Realistic leaks are baked in for the upper layers to find: occasional oversizing right after a loss, a few late-session entries, some `Manual` exits. The script ends by reloading both CSVs and verifying every trade against the tape. |

## Layer 2 modules

| File | What it does |
| --- | --- |
| `rules.py` | The rules schema, one JSON object: `direction` (`long_only`/`short_only`/`both`), `stop_pct`, `target_pct` (null = no fixed target), `session_start`/`session_end` ("HH:MM" Eastern), `max_size`, `exit_style` (`fixed`/`breakeven`/`trailing`/`time`), `slippage_bps` (default 2). Active rules persist to `rules.json`; a named-strategy library lives in `strategies.json` (save current rules under a name, reload with one click). A dozen one-click **presets** (momentum, mean reversion, breakout, trend, scalp, pullback, fade, opening range, VWAP reversion, gap-and-go, range, power hour) pre-fill the same schema — everything stays editable. `readout(rules)` produces the **live readout**: reward-to-risk and the breakeven win rate `1/(1+R)` in plain english, cheap enough to recompute on every keystroke. |
| `strategy_ai.py` | The plain-english path (label it **AI**, purple — happens once per strategy). The user picks one of ~15 trading families and types how they trade; family + text go to the model with a system prompt demanding strict JSON matching the schema (no prose, no backticks). The reply is parsed defensively and every field lands in an editable review form. Nothing runs until the user clicks **approve** — approving locks and saves the rules and stores the free-text playbook (`playbook.json`) for the coach to use later. |
| `ai_client.py` + `ai_config.py` | Provider client: key from the environment first (`AI_API_KEY` / `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`), `ai_config.py` as fallback. Routed by prefix — `sk-ant-...` → Anthropic Messages API (default `claude-opus-5`, with server-side refusal fallback enabled), plain `sk-...` → OpenAI (default `gpt-4o-mini`). `AI_MODEL` overrides the model either way. 90s timeout, one clear error line per failure mode. **No key at all?** The AI path greys out with a short note; presets and manual entry still work, and everything deterministic downstream runs keyless. |

## Layer 3 modules

| File | What it does |
| --- | --- |
| `metrics.py` | The shared metrics library — every number the engine and the dashboard show comes from here, so they can never disagree. Win rate, avg win/loss, payoff, profit factor, expectancy in $ and in R (1R = stop_pct% × entry × qty), P&L without the top-N winners, top-decile concentration of gross winning P&L (bounded 0–100%), hold times, and the plan-adherence check (direction / session window / max size) shared by both engines. |
| `auditor_engine.py` | `audit(trades, rules)` — the behavioral audit from the log alone, written into **one findings dict** the other layers reuse: core stats, luck (P&L without top 1/3/5 + concentration), revenge sizing (avg size after a same-day loss vs otherwise + every oversized trade), disposition (hold times winners vs losers, MFE capture, avg give-back/ETD), P&L by hour and by instrument, and per-trade plan adherence — with 'Manual' exits reported as a separate info line, never a violation. |
| `discipline.py` | `replay(trades, bars, rules)` — what your rules would have done, trade by trade, on the real bars. Conservative by design; the exact 9 assumptions ship in the aggregate and get printed on the dashboard: no intra-entry-bar lookahead, stop-before-target in the same bar, gap-aware stops (fill at the open, then slippage), targets as exact resting limits never gap-improved, slippage only on stop/market/time exits, the real trade's commission, explicit exit_style modeling (fixed / breakeven at +1R / trailing by the stop distance / time), plan eligibility first ('outside your plan, not taken' = 0 to the benchmark), and session-end close-outs. Output: per-trade rule exit price/time/kind + P&L, and actual vs rule-managed vs signed difference. |

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
python tests/test_layer2.py
python tests/test_layer3.py
```

Layer 1: 34 offline checks against synthetic tapes shaped exactly like the
yfinance output — session filtering, the 5m/1m hybrid merge, the trade
generator (count, cadence, profitability, leaks, bar-level verification)
and the NinjaTrader loader round trip. Layer 2: 53 offline checks with the
AI client stubbed — schema validation, persistence, the strategy library,
all presets, the readout math, provider routing, keyless degradation, and
the draft → review → approve flow. Layer 3: 60 offline checks — metric
identities on a hand-made ledger, every fill mechanic verified against
hand-computed prices (gap fills, stop-before-target, breakeven arming,
trailing stops, slippage to the tick), eligibility, and the full
audit + replay pipeline on the synthetic tape.
