# Adaptive Trading Bot — Current Logic

Branch: `claude/my-virtual-office-setup-0ndmyj`
Code: `app/trading/adaptive_trading_bot.py`, `app/trading/strategies/mean_reversion.py`, `app/run_bot.py`

This describes what the bot actually does today, not the original design —
several rewrites replaced the original two-strategy (SwingReversalPro +
MeanReversion) architecture with a single unified pipeline. `SwingReversalPro`
the class is no longer imported or used; its trend-following role is filled by
`EntryHealthScorer`, described below.

## 1. Market state (4H) — which regime are we in?

Every tick, `_step1_market_state_engine` classifies the 4H candle into one of
8 states, checked in priority order:

| State | Condition (first match wins) |
|---|---|
| HIGH_VOL | ATR expansion > 1.8 and BB width > 0.7 |
| LOW_VOL | BB width < 0.2 and ATR expansion < 0.8 — **skipped, never traded** |
| BREAKOUT | BB width > 0.3, ATR exp > 1.2, volume > 1.4× avg, ADX < 24 |
| STRONG_TREND | ADX > 25, efficiency > 0.55, \|+DI − −DI\| > 8 |
| TRENDING | ADX > 18, efficiency > 0.35, \|+DI − −DI\| > 3 |
| REVERSAL | RSI > 68 or < 32, ADX > 15, efficiency < 0.42 |
| EXHAUSTION | ADX > 20, efficiency < 0.28 |
| SIDEWAY | default (none of the above) |

`_TRADEABLE_STATES` = everything except LOW_VOL.

## 2. Unified signal pipeline — one score, not two competing strategies

For each 15M bar, for each candidate direction (LONG, SHORT), `_generate_signal`
combines three components into a single 0-100 score and gates on ONE threshold
per state (`ADAPTIVE_THRESHOLDS[state]["total_min"]`). This replaced an earlier
design with five separate stacked AND-gates (bias / ADX / health / confidence
/ RSI-MACD re-check) that produced near-zero trade throughput in production.

```
total = entry_score × 0.40 + context_score × 0.30 + direction_fit × 0.30
```

### 2a. Chop-zone veto (checked first, before scoring)

Trend-state entries (not SIDEWAY/EXHAUSTION/REVERSAL) are rejected outright if
price sits within `MIN_EMA_DIST_ATR` (default **0.8**, env `ADAPTIVE_MIN_EMA_
DIST_ATR`) of the 15M EMA20, measured in ATR units. This was added after
instrumenting the backtest and finding that trades entered in this zone had
**WR 48.6%, +$0.97/trade** vs **WR 54.0%, +$10.13/trade** outside it — the
composite scores themselves had ~zero power to separate these ("fake" straight-
to-SL trades from real ones), but this single raw feature did. MR states are
exempt — they deliberately enter at mean extremes, which are far from EMA by
design.

### 2b. Entry score (15M) — style depends on market state

- **SIDEWAY / EXHAUSTION / REVERSAL** (`_MR_STATES`): mean-reversion scoring,
  using individual step methods from `MeanReversionStrategy`
  (`mean_reversion.py`) folded into one weighted score instead of a sequential
  gate cascade:
  - overextension 25%, liquidity sweep 20%, reversal structure 20%, momentum
    reversal 20%, candle quality 10%, volume 5%.
  - SL comes from `_step14_sl` (max of ATR×1.5 or sweep-low/high ± buffer).
- **All other states**: `EntryHealthScorer.compute()` — EMA(20) + MACD(15) +
  ADX(15) + Volume(15) + ATR(10) + RSI-zone(10) + Pattern(15) + State-quality
  (10), 0-100.
  - For **STRONG_TREND / BREAKOUT** only (`_LOCATION_STATES`), blended 70/30
    with `_entry_location_score`: a pullback-preference term that penalizes
    ~40 pts per ATR of extension above/below EMA20. Backtesting showed this
    location term helps efficient trends (pullbacks reliably resume) but
    *hurts* choppier TRENDING (continuation beats pullback there) — so it's
    gated to just these two states.
  - SL = `pattern_low`/`pattern_high` from the 15M candle.
- Hard floor: `entry["score"] < ENTRY_SCORE_FLOOR` (45) rejects regardless of
  how strong the 4H/1H alignment is, so a weak entry bar can't pass purely on
  HTF strength.

### 2c. Context score (1H)

`_context_score`: does the 1H timeframe support the candidate direction?
60% directional alignment (EMA5 vs EMA20 + RSI, scaled ±100), 20% MACD
momentum agreement, 20% ADX strength.

### 2d. Direction fit (4H)

`_direction_fit`: does the candidate direction fit the 4H regime?
- SIDEWAY: no bias required (flat 70).
- Counter-trend states (REVERSAL, EXHAUSTION): want the *opposite* of the 4H
  regime direction (fading an exhausted move).
- All other states: want *agreement* with the 4H regime direction.

4H regime direction itself (`_regime_direction`) is a continuous -100..+100
score from EMA5-vs-EMA20 spread, EMA20 slope, and RSI.

### 2e. Per-state total_min (the single gate)

| State | total_min | Note |
|---|---|---|
| STRONG_TREND | 55 | best historical performer, most permissive |
| BREAKOUT | 55 | |
| TRENDING | 62 | raised after backtest showed WR ~41% at looser settings |
| REVERSAL | 62 | counter-trend, made selective |
| HIGH_VOL | 62 | |
| SIDEWAY | 68 | tightened — was ~$0.60/trade edge at 62, strict win raising it |
| EXHAUSTION | 68 | fades 4H trend, lost money in every tested config, kept maximally selective rather than fully removed |
| LOW_VOL | 999 | never trades |

## 3. Risk engine — SL/TP/sizing (`_step5_risk_engine`)

- SL: from the signal's `sl_price`, floored at 2.0% of entry price
  (`min_sl_dist`) to cap notional size.
- TP1 = entry + `TP1_R` (default **0.7**) × SL-distance, closes
  `tp1_close_pct` (50%) of the position, moves SL to breakeven.
- TP2 = entry + `TP2_R` (default **1.5**) × SL-distance, closes the remainder.
  Both are env-tunable (`ADAPTIVE_TP1_R`/`ADAPTIVE_TP2_R`) as a live WR↔profit
  dial — lowering TP1_R raises win-rate (more trades reach it) at the cost of
  average win size; sweeping showed 0.7→0.4 only adds ~+1.7pp WR while costing
  ~$740 PnL, so this isn't a free lunch.
- Size multiplier = confidence-based multiplier (`ConfidenceScorer.
  get_size_multiplier`, ≥85→1.0, ≥75→1.0, ≥60→0.65, else skip) × health
  multiplier (1.0 if health≥75 else 0.65) × per-entry-type learning weight
  (`PatternLearningEngine`, adjusts ±15% every 20 trades based on rolling
  win-rate per entry type).
- **TP2 is also attached as a real OKX exchange-side order** (`stopLoss` +
  `takeProfit` on the entry order) so the position has downside AND upside
  protection even if the bot process is offline. TP1's 50% partial stays
  bot-managed only — OKX's attach mechanism supports one TP leg per order,
  sized to the full position, so it can't express a partial-size leg.

## 4. Position management (`_manage_open_position`) — checked every tick

In priority order:
1. **Reversal spike**: a 15M candle with a dominant wick (>60% of range,
   opposite-direction close, wick ≥1.5× the other side) → immediate full close.
2. **Trend fade**: after ≥4 bars held, if ≥2 of {ADX<14, EMA gap <0.25%, MACD
   hist against direction} → tighten SL to `price − ATR×0.8` (does not close,
   just protects).
3. **Emergency signals**: opposite CHOCH, ATR collapse, momentum collapse,
   volume collapse, invalid structure (from the indicator engine) → immediate
   full close.
4. **TP2 hit** → full close (unified TP has no TP3 runner, so this always
   closes 100% of whatever remains).
5. **TP1 hit** → partial close (50%) + SL to breakeven.
6. **Health-tiered management** (`PositionHealthCalculator.calculate`, 0-100):
   - ≥80: hold.
   - ≥60 (or already at breakeven): trail SL to `price ∓ ATR×2`.
   - ≥40: force SL to breakeven if TP1 already hit.
   - ≥20: close 50% early if TP1 already hit (`HEALTH_REDUCE`).
   - <20: full close (`POOR_HEALTH_EXIT`) — this is the "close before hitting
     SL" behavior for a trade whose momentum has genuinely reversed.
7. SL check (price crossed the current SL level) → full close.

## 5. Exchange reconciliation & safety nets (`run_bot.py`)

- **Startup**: `reconcile_with_exchange` adopts or clears position state from
  the real exchange position (reads actual `side`/`posSide`, not the sign of
  ccxt's unsigned `contracts` field — an earlier bug always inferred LONG).
- **Every 5 min** (all env-tunable, default 300s):
  - Re-check cooldown expiry independent of new-candle ticks.
  - Re-run exchange reconciliation (catches positions closed externally,
    e.g. manually or via the exchange-side TP/SL).
  - Sync `account_balance` from the real exchange balance (position sizing
    used to run off a fixed default that never matched the real account).
  - Log `[Adaptive][sym] state=...` regardless of whether a new bar closed.
  - Log `[FilterStats]` rejection tallies (bias/health/confidence buckets) at
    INFO level, visible without `LOG_LEVEL=DEBUG`.
- Telegram: `/stats`, `/positions`, `/log`, `/status` on demand; trade open/
  close notifications are event-driven (no periodic auto-digest — removed
  because it spammed the chat every 30 min regardless of activity).

## 6. What was tried and reverted

A pullback-and-resume entry trigger (require RSI to have pulled back and MACD
histogram to be turning in-direction on the current bar, before allowing
entry) was implemented and backtested. It cut trades from 421→282, dropped
WR 51.3%→47.5%, and collapsed PnL to $187 from $2359. This strategy's edge
comes from entering *with* trend strength, not waiting for dips — the location
term above already captures the useful part of "don't chase" without this
trade-off. Do not re-attempt without new evidence.

## 7. Backtest results (BTC, Jan–May 2026, 15M bars, single symbol)

| Stage | Trades | WR | Net PnL | PF | MaxDD | Sharpe |
|---|---|---|---|---|---|---|
| Pre-rewrite baseline | 133 | 55.6% | +$520 | — | 2.8% | 2.09 |
| Unified pipeline | 606 | 50.5% | +$2369 | 1.45 | 3.2% | 1.81 |
| + entry-quality tuning (location/floor/drop EXHAUSTION) | 421 | 51.3% | +$2359 | 1.59 | 2.8% | 2.35 |
| + chop-zone filter | 371 | 55.0% | +$2102 | 1.80 | 2.0% | 2.97 |
| + SIDEWAY selectivity (current) | **318** | **56.9%** | **+$2113** | **1.92** | **1.2%** | **3.37** |

Per-month breakdown of the current config shows profit in every one of the
5 months tested (Jan +$398, Feb +$760, Mar +$458, Apr +$116, May +$382) — no
single lucky month is carrying the average, which is a reasonable (not
conclusive) sign against gross overfitting.

**Known limitation**: all tuning is in-sample on one symbol (BTC) over one
5-month window; other symbols' backtest data directories are currently empty.
No out-of-sample or cross-symbol validation has been done yet. Recommended
before scaling up real capital: run paper/demo or small live sizing for a few
weeks and compare live WR/PF against the ~57% WR / 1.9 PF this document
reports, rather than tuning further against the same historical window.
