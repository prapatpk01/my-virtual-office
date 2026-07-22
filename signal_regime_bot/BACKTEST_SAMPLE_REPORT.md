# DUALCORE V2.0 — Frequency Verification Sample

## Test method

- Production pipeline and entry-engine code.
- Native 5M data for BTC and SOL.
- Closed-candle signals; fill at the next 5M open.
- Adverse slippage: 0.05%.
- Fee: 0.10% per fill, including partial exits.
- TP1, TP2, fee-adjusted runner stop and conservative same-candle SL-first ordering.

Two separate seven-day windows were used as a fast verification sample. This is not a full out-of-sample proof.

## V2.0 results

| Window | Symbol | Trades | Win rate* | PF | Net R |
|---|---|---:|---:|---:|---:|
| 1–8 Feb 2026 | BTC | 4 | 100.0% | ∞ | +2.73R |
| 1–8 Feb 2026 | SOL | 6 | 66.7% | 2.05 | +2.65R |
| 1–8 Mar 2026 | BTC | 1 | 0.0% | 0.00 | -1.31R |
| 1–8 Mar 2026 | SOL | 1 | 100.0% | ∞ | +1.15R |
| **Combined sample** | **BTC + SOL** | **12** | **75.0%** | **about 2.36** | **+5.22R** |

\*Small fee-adjusted runner gains are counted as positive trades by the test harness.

## Same-window frequency comparison

| Version | Trades | Net R | Observation |
|---|---:|---:|---|
| V1.9 | 6 | +6.74R | Very selective; higher average R per trade |
| **V2.0** | **12** | **+5.22R** | **Trade count doubled; lower average R per trade** |

V2.0 produced twice as many trades in the sampled windows. The trade-off is expected: average quality per trade is lower than V1.9, although the combined sample remained profitable.

## Frequency interpretation

Observed during 14 sampled days:

- BTC: 5 trades.
- SOL: 7 trades.

This is consistent with the objective of moving toward roughly 8–15 trades per month per actively trending symbol, but it does not guarantee that every symbol will reach that number in every month. Range-bound months will still generate fewer entries because structure, fee, room and R:R gates remain hard blocks.

## Important limitation

The full February–June continuous run is included in `backtest_exact_5m.py` for local/Railway execution, but this packaged verification report intentionally does not claim a full-period result. Only BTC and SOL had native 5M files suitable for this exact execution test in the uploaded dataset.
