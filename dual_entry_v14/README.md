# DUAL ENTRY PRECISION V1.4
**FAST SYSTEMATIC HTF STRUCTURE, PULLBACK & MOMENTUM ENGINE**

Modular, stateful, event-driven, restart-safe trading system for OKX
perpetual futures. Top-down analysis like an expert trader: 4H macro
structure → 1H bias/structure → 15M entry timing, with **two independent
entry engines** (Fast Pullback + Momentum Breakout) and structure-first
risk management.

> ⚠️ No win rate or profit is guaranteed. Trading leveraged futures can
> lose more than the risked amount. `RISK_PER_TRADE` above 2% triggers an
> explicit warning at startup.

---

## 1. Architecture

```
Market Data
  ↓ Data Quality Gate           (completeness, OHLC sanity, staleness, shock)
  ↓ 4H Macro Structure Engine   (macro direction, major zones, premium/discount, conflicts)
  ↓ 1H Bias Engine              (structure 0-35, trend 0-25, momentum 0-15, DMI 0-15, ADX 0-10; Soft Bias Mode)
  ↓ HTF S/R + Supply/Demand     (zones, not lines — scored 0-100, freshness, flips)
  ↓ HTF Pattern Engine          (BREAK&RETEST / FLAGS / COMPRESSION / DOUBLE TOP-BOTTOM; advanced = shadow)
  ↓ 15M Regime Engine           (HMA10/16 + DMI + CHOP; PULLBACK_TRANSITION states; chop = hard no-trade)
  ↓ ┌──────────────────────────┬──────────────────────────┐
    │ Fast Pullback Engine     │ Momentum Breakout Engine │
    │ zone touch + reclaim,    │ breakout candle or next  │
    │ sweep+reclaim, HMA10     │ bar ONLY; close beyond   │
    │ reclaim, micro BOS…      │ level, never wick-only   │
    └──────────────────────────┴──────────────────────────┘
  ↓ Candidate Selection         (edge score; pullback priority; ambiguity reject)
  ↓ Portfolio Risk Gate         (max 2 positions, total open risk, correlation clusters)
  ↓ Structure Risk Manager      (deterministic stop selection, effective-risk sizing, structure targets)
  ↓ Execution Quality Gate      (slippage/deviation/spread/expiry; momentum rejects chasing sooner)
  ↓ Execution Engine            (idempotent clOrdId, journaled intent, native SL/TP, emergency protection)
  ↓ Position Manager            (setup-specific exits, BE moves, cooldowns; HMA flip alone never exits)
  ↓ Performance & Diagnostics   (trade records, module gate, reason codes, shadow stats)
```

**Key design principles** (spec §2):
- **Structure first**: structure → location → room → trigger → momentum → candle.
  Indicators can't compensate for broken structure or bad location.
- **Hard gates only where necessary** (data, duplicates, chop, strong HTF
  conflict, opposite 1H CHOCH, invalidation, opposing zone, room, stop
  sanity, RR, wick-only breaks, overextension, execution quality, paused
  module). ADX / DI / ROC / volume / patterns / slope / compression are
  **score modifiers, never blanket hard gates**.
- **Fast entry without chasing**: pullback fires on a single sufficient
  trigger at a valid location (Tier-1 at strong zones); momentum only on
  the breakout candle or the very next bar.
- **One position per symbol, max 2 total**; exchange is the source of
  truth; restart-safe reconciliation; deterministic `clOrdId` idempotency.

## 2. Project structure

```
dual_entry_v14/
  config.py            all defaults (spec §35), env overrides, validation
  enums.py  models.py  interfaces.py
  market_data.py  data_quality_gate.py  indicator_engine.py
  swing_engine.py  structure_engine.py  support_resistance_engine.py
  supply_demand_engine.py  pattern_engine.py  candle_engine.py  liquidity_engine.py
  macro_context_engine.py  bias_engine.py  regime_engine.py
  pullback_engine.py  momentum_engine.py  candidate_selector.py
  portfolio_risk_manager.py  risk_manager.py
  execution_quality_gate.py  okx_exchange.py  execution_engine.py
  position_manager.py  performance_engine.py  diagnostic_engine.py
  state_store.py  notifier.py
  backtest_engine.py  walk_forward.py  monte_carlo.py
  main.py
  tests/test_core.py
```

No god classes; strategy logic never imports ccxt; backtest and live run
the **same Bot pipeline** through `ExchangeInterface` (OKXExchange vs
SimulatedExchange).

## 3. Running

### Live / paper
```bash
export SYMBOLS="BTC/USDT:USDT,ETH/USDT:USDT"
export PAPER_TRADING=true            # false for live
export OKX_API_KEY=... OKX_SECRET_KEY=... OKX_PASSPHRASE=...
export TELEGRAM_BOT_TOKEN=... TELEGRAM_CHAT_ID=...
export RISK_PER_TRADE=0.01           # 0.03 default; >0.02 warns; 0.05 allowed with warning
python -m dual_entry_v14.main
```

### Backtest
```python
from dual_entry_v14.config import Config
from dual_entry_v14.backtest_engine import run_backtest
# data = {symbol: {"15m": [Candle...], "1h": [...], "4h": [...]}}
res = await run_backtest(cfg, data)
```
The sim exposes only candles whose **close time ≤ the simulated clock**
(no lookahead; swings confirm after right-side bars; the quality gate,
expiry and cooldowns all run on the sim clock). Note: the first
`min_4h_candles × 16` 15m bars of the dataset are warm-up for the 4H
window — supply history accordingly.

### Walk-forward & Monte Carlo
```python
from dual_entry_v14.walk_forward import walk_forward
from dual_entry_v14.monte_carlo import monte_carlo
wf = await walk_forward(cfg, data, n_windows=4)          # OOS stability of fixed config
mc = monte_carlo([t.result_r for t in trades], risk_per_trade=0.01)
```

### Tests
```bash
python dual_entry_v14/tests/test_core.py
```

## 4. State & restart safety (spec §4)

- Per-symbol `SymbolState` persisted **atomically** (tmp + rename) with a
  monotonically increasing `state_version`.
- Execution journal (`execution_journal.jsonl`) records ORDER_INTENT
  **before** every send; POSITION_OPEN / CLOSED / errors after.
- `signal_key = (symbol, 15m, candle_ts, setup, direction)` →
  `client_order_id = sha256(key)[:24]` — the same signal can never order
  twice, even across restarts (exchange is queried for the clOrdId first).
- On startup/every loop the reconciler adopts any exchange position that
  memory doesn't know (and clears local state for positions closed on the
  exchange). Memory being empty is never treated as proof of being flat.

## 5. Risk (spec §23)

- `risk_cash = equity × RISK_PER_TRADE × modifiers` (bias soft-pass,
  4H mild conflict, deep pullback, correlated exposure, module gate).
- Deterministic stop: nearest setup-relevant structure level (setup low /
  sweep low / zone boundary / confirmed swing / breakout base / pattern
  invalidation) + ATR buffer, validated to `[0.45, 1.60] ATR`, widened to
  minimum if too tight — never auto-deepest.
- **Effective risk** = stop distance + entry/exit slippage + round-trip
  fees; quantity = risk_cash / effective-risk-per-unit, floored to lot
  step, checked vs min qty/notional/margin.
- Target = min(base RR target computed off effective risk, structure
  target in front of the opposing zone). `MIN_ACCEPTABLE_RR = 1.08` after
  costs, re-checked after the actual fill (abort policy).

## 6. Exits (spec §26)

| | Pullback | Momentum |
|---|---|---|
| Break-even | at +0.65R → BE+0.05R | at +0.75R → BE+0.05R |
| Hard exit | invalidation break, 15M/1H opposite CHOCH, demand break | close back under breakout level + displacement / CHOCH (false breakout) |
| Soft exit | held ≥2 bars AND close < HMA10 AND ≥2 of 4 weakness signals | close back inside + ROC against |
| HMA flip alone | **never exits** | **never exits** while level holds |

Cooldowns: normal 1 bar, SL 2, early exit 1, TP 0, false breakout 2;
3 consecutive losses → max(30 min, 2 bars). No same-bar re-entry.

## 7. Module performance gate (spec §29)

Each engine (pullback / momentum) is independently ACTIVE →
REDUCED_RISK (PF 0.90–1.05 over ≥30 trades) → PAUSED (PF <0.90 &
negative expectancy over ≥50) → SHADOW_MODE re-qualification (15–20
shadow signals with improving PF) → back to REDUCED_RISK. Thresholds are
never loosened after losses; risk is reduced instead.

## 8. Behavioral targets

- Enter pullbacks **near the zone**, not after the move has left.
- Momentum entries only on the breakout candle or the next bar.
- ~8–15 trades/month per active symbol *when conditions cooperate* — the
  system never forces trades in markets without an edge (chop is a hard
  no-trade).
