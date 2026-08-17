# Simple Precision V1.0

The production bot now runs one deterministic strategy per symbol. Legacy
strategies remain available for comparison, but Railway does not instantiate
them.

## Decision flow

1. **4H direction** — EMA20/50 stack (40), EMA20 slope (30), price vs EMA20
   (30). The EMA stack is mandatory and the score must be at least 70.
2. **1H quality** — ADX (25), Choppiness (25), DI direction (20), MACD
   histogram (15), price vs EMA20 (15). Default pass score is 55. Only
   ADX below 15, Choppiness at or above 62, or full three-factor opposition
   are hard blocks.
3. **15M entry** — one closed-bar trigger is sufficient: a fresh EMA8/13
   cross, EMA13 pullback/reclaim, or a seven-bar structure breakout. Entries
   are rejected when price is more than 1.5 ATR from EMA20 or less than 1.2R
   from confirmed opposing structure.

All decisions use closed candles. The same 15M candle cannot emit two entries.

## Risk and exit

- Stop distance: volatility bounded to 0.7–1.4 ATR and informed by the last
  seven 15M candles.
- T1: +1.0R, close 40%, move the remaining stop to breakeven.
- TP2: +2.0R on the remaining 60%.
- Early exit: a confirmed reverse EMA8/13 cross plus close beyond EMA13, or
  two consecutive closes beyond EMA13.
- Re-entry pause: two closed 15M candles after an exit.
- Existing account-level limits remain active: position cap, portfolio heat,
  loss-streak cooldown, drawdown stop, leverage, exchange-side SL/TP, and the
  configured FX-week sleep gate.

## Railway variables

Defaults work without new variables. Optional tuning controls:

| Variable | Default | Purpose |
|---|---:|---|
| `SIMPLE_PRECISION_SYMBOLS` | `SYMBOLS` | Production symbol list |
| `SP_QUALITY_THRESHOLD` | 55 | Minimum 1H quality score |
| `SP_ADX_MIN` | 15 | Non-trend hard block |
| `SP_CHOP_MAX` | 62 | Sideways hard block |
| `SP_MAX_ENTRY_DISTANCE_ATR` | 1.50 | Anti-chase limit |
| `SP_MIN_ROOM_R` | 1.20 | Minimum room to opposing structure |
| `SP_STOP_ATR_MIN` | 0.70 | Minimum stop distance |
| `SP_STOP_ATR_MAX` | 1.40 | Maximum stop distance |
| `SP_TP1_R` | 1.0 | First target |
| `SP_TP1_TRIM_PCT` | 0.40 | T1 reduction fraction |
| `SP_TARGET_R` | 2.0 | Final target |
| `SP_EXIT_COOLDOWN_BARS` | 2 | Closed 15M bars before re-entry |

Do not tune several variables at once. Validate changes in paper mode and
compare at least trade count, win rate, profit factor, expectancy, max
drawdown, and results by symbol.
