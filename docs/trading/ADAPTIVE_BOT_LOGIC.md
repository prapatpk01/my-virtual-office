# Adaptive Trading Bot — SMC MTF V7.0

Branch: `claude/my-virtual-office-setup-0ndmyj`

Live path:
- `app/trading/indicator_engine.py` — deterministic multi-timeframe signal engine
- `app/trading/adaptive_trading_bot.py` — execution/risk/position state
- `app/run_bot.py` — market-data loop, OKX execution, Telegram, reconciliation

## Core idea

Each timeframe has one job. The bot does not add 1H as another hard gate.

| Timeframe | Engine | Job |
|---|---|---|
| 4H | TSS-style trend tunnel | Choose LONG / SHORT / NEUTRAL only |
| 15M | Market Structure | Confirm HH/HL, LH/LL, BOS/CHOCH |
| 5M | AMD | Find accumulation, liquidity manipulation/sweep, then distribution |
| 1M | IFVG | Precision execution after an inverted FVG retest/rejection |

All decision inputs are **closed candles**. The runner may poll more often, but it evaluates a new setup only once per newly closed 1M candle.

## 1. 4H — TSS-style direction

This is an internal, deterministic trend-tunnel approximation; it does not claim to clone a proprietary TSS indicator.

Inputs:
- EMA20 vs EMA50
- EMA20 slope
- HMA16 slope

LONG requires at least 2 of 3 bullish votes; SHORT requires at least 2 bearish votes. Otherwise bias is NEUTRAL and no new entry can fire.

## 2. 15M — Market Structure

Confirmed fractal swing highs/lows are used to classify:
- `HH/HL` → bullish structure
- `LH/LL` → bearish structure
- mixed swings → `TRANSITION`

The engine also detects BOS/CHOCH. A CHOCH can temporarily allow the new direction even before two full swing pairs have rebuilt.

## 3. 5M — AMD setup

A fixed recent window is split into:
- older block = candidate Accumulation range
- recent block = candidate Manipulation and Distribution

The accumulation range must be compact relative to ATR. Then:
- LONG: price sweeps below the range, closes back inside/above it, then displaces upward
- SHORT: price sweeps above the range, closes back inside/below it, then displaces downward

The manipulation extreme becomes the structural stop reference.

## 4. 1M — IFVG execution

LONG execution:
1. a bearish FVG exists,
2. price later closes above it, inverting the gap,
3. a fresh retest/rejection occurs within the last 3 closed 1M bars.

SHORT is the inverse.

A trade is emitted only when all four layers align:

`4H Direction → M15 Structure → M5 AMD Distribution → M1 IFVG retest`

## 5. Stop, target and size

SL is placed beyond the 5M manipulation extreme with a small ATR buffer. The bot never moves an overly wide structure stop back inside the manipulation just to make the trade fit; it rejects setups wider than `SMC_MAX_SL_PCT` instead.

Defaults:
- minimum SL distance: `SMC_MIN_SL_PCT=0.25%`
- maximum accepted SL distance: `SMC_MAX_SL_PCT=3.0%`
- TP1: `SMC_TP1_R=1.0R`, close 50%
- after TP1: SL → entry +/− `SMC_BE_LOCK_R=0.10R`
- TP2: `SMC_TP2_R=2.0R`, close all remaining
- size: min(risk-based size, margin × leverage cap)

The remaining 50% after TP1 is treated as a runner. Before TP2 it can exit early if M15 produces an opposing CHOCH or M5 confirms opposite distribution.

## 6. OKX safety and restart behavior

- The opening order carries the structure SL and TP2 to OKX.
- After TP1 the bot amends the real exchange-side SL to the locked BE level when the SL algo id is available.
- Before managing a local position, the runner checks whether OKX is already flat so exchange-side TP/SL cannot leave a ghost local position.
- If Railway restarts and local `/tmp` position state is missing, the runner can adopt a live OKX position **only when it can recover the real attached SL**. It never invents a stop for an unknown live position.
- Existing pre-V7 positions can still load. Legacy runner positions retain their old 15M EMA20 runner-exit behavior.

## 7. API-load control

M1 is polled at the configured loop interval. M5, M15 and H4 candle sets are cached and refreshed only when their own time buckets advance. This keeps M1 execution responsive without multiplying exchange requests unnecessarily.

## 8. What V7 intentionally removes from entry logic

The new entry pipeline does not require 1H confirmation, MACD, ADX, CHOP, Bollinger Bands, RSI crossing, or EMA crossover. Those extra gates made earlier versions harder to audit and could delay/kill valid entries. V7 focuses on directional context, structure, liquidity manipulation and execution location.

RSI/ATR may still be calculated for telemetry/risk context, but they are not entry gates.

## 9. Validation requirement

This is a rule rewrite, not proof of an edge. Before increasing live risk, compare V7 against the previous Adaptive logic using the same symbols, fees/slippage assumptions, and time window. Track at minimum: trades/month, win rate, average R, profit factor, max drawdown, MAE/MFE, and results by symbol/session.
