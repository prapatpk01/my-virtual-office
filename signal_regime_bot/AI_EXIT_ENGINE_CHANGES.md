# V3.1 AI Exit Engine

Replaced one-signal Spike Guard closing with a stateful multi-factor exit engine.

- Live acceleration alone = WATCH only.
- Normal close requires score, multiple independent confirmations, adverse depth, and persistence across 2 closed 5M bars.
- Strong-trend and high-entry-score positions receive stricter close thresholds.
- 2-bar post-entry grace prevents fresh-position noise exits.
- True emergency only at >=0.82R plus >=2.8 ATR live acceleration, or >=0.94R adverse.
- Evidence: 5M/15M reversal candle, EMA8/13 invalidation, 15M EMA20 loss, 5M/15M structure break, MACD+ROC flip, volume expansion.
- Native SL/TP remains first priority and unchanged.
