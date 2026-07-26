# Regime Bias Bot — runtime package (MODE=regime)

Version: DUALCORE V3.0.1 OKX NETWORK RESILIENCE
Packaged: 482fe2a on branch claude/railway-trading-bot-kvn686

These are the files the LIVE bot actually loads (traced from main.py).
Backtest scripts, change-logs and the unused price_action/context_engine/
style_engines/early_booster/report cluster are intentionally excluded — they
are NOT imported when the bot runs.

## Entry point
- main.py ............. Bot loop, Telegram cmds (/stats /status /trades /restats), journal
- config.py ........... all tunables (TP1_R, RUNNER_LOCK_R, fee_rate, symbols, risk, thresholds)

## Signal pipeline (4H → 1H → 15M → 5M)
- pipeline.py ......... orchestrates the layers, weekend-halt gate
- regime_engine.py .... 4H/1H macro regime classification
- bias_engine.py ...... 1H/15M/5M directional bias
- entry_engine.py ..... DUALCORE V3.0 Expert Multi-Mode setups (cross/pullback/SMC/breakout/sweep/continuation/range)
- indicators.py ....... EMA/ATR/ADX/structure/swing helpers

## Execution / risk
- position_manager.py . open/close, TP1 partial, runner stop (breakeven+0.2R lock), BE, exits
- risk_manager.py ..... sizing, daily limits, loss-streak cooldown, max positions
- exchange_client.py .. OKX orders plus timeout-safe retries and native REST candle fallback
- spike_guard.py ...... fast 5m/15m V-reversal force-close

## Data / IO
- data_engine.py ...... per-candle cache, closed-bar handling and bounded stale-data recovery
- telegram_notifier.py  alerts, charts, command polling (409-safe)
- chart_engine.py ..... entry-signal candlestick charts

## Deploy
- Dockerfile, railway.json, requirements.txt, .env.example, VERSION.txt

## Key current settings (config.py)
- fee_rate = 0.0005 (verified 0.05% OKX)
- tp1_r = 0.80, tp1_fraction = 0.50, tp2_r = 2.40
- runner_lock_r = 0.2 (runner stop = breakeven + 0.2R after TP1)
- risk_per_trade = 0.05, leverage 20x, max 2 positions
- env overrides: TP1_R, RUNNER_LOCK_R, SYMBOLS, RISK_PER_TRADE, MAX_POSITIONS, STATS_SINCE_DATE

## V3.0.2 patch
- `entry_engine.py`: populate EMA8/EMA13/MACD before duplicate-bar return.
- `main.py`: preserve last meaningful status result during same-bar polling.
- `EMA_STATUS_FIX.md`: patch notes.

## V3.1 AI Exit Engine
- `ai_exit_engine.py` — stateful multi-factor WATCH → CONFIRM → CLOSE engine.
- `main.py` — integrates AI Exit before legacy TP1-gated EMA runner exit.
- `config.py` — embedded defaults; no new Railway variables required.
- `telegram_notifier.py` — confirmed/emergency AI exit notifications.

- market_schedule.py ... global FX weekly Sleep Mode for all symbols; DST-safe 4h pre-open wake
