# DUALCORE V3.0 — Validation Report

## Method

- Production `Pipeline`, `RegimeEngine`, `BiasEngine` and `EntryEngine` were used directly.
- Entries use the next 5M open with adverse slippage.
- Round-trip fees use the configured 0.10% per fill.
- Intrabar ordering is conservative: stop is evaluated before profit target when both occur in the same candle.
- TP1 partial close, fee-aware runner stop and post-TP1 signal exits are included.
- Historical data root: the supplied `Crypto(1).zip` extraction.

## Sample results

| Symbol | Period | Trades | Win rate | PF | Net R | Max DD R |
|---|---|---:|---:|---:|---:|---:|
| BTC | 2026-04-01 → 2026-04-08 | 6 | 66.67% | 1.31 | +0.81R | -2.62R |
| BTC | 2026-04-08 → 2026-04-15 | 7 | 28.57% | 0.07 | -6.11R | -6.56R |
| SOL | 2026-03-01 → 2026-03-08 | 4 | 50.00% | 0.73 | -0.63R | -2.36R |

## Interpretation

The rewrite solves the inactivity problem in historical replay: BTC generated 6–7 completed trades per week in the two April samples. However, the desired **67%+ win rate and PF >1.20 were not stable across all samples**. One BTC window reached the target area; the following window failed badly during repeated trend-failure conditions.

This means V3.0 should be treated as an expert multi-mode execution framework and a forward-test candidate—not as a proven 67% system. Static indicator/SMC rules cannot guarantee those KPIs in every market regime.

## Required next validation

- Run demo/paper for at least 50–100 completed trades.
- Review results separately by setup type, symbol and regime.
- Disable any setup with at least 10 trades and PF below 0.90.
- Do not judge performance from Telegram entry count alone; use net PnL after all OKX fees.
- Consider reducing live risk from 5% while the forward sample is still small.
