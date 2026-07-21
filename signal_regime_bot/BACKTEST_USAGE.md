# Exact 5M backtest runner

```bash
pip install -r requirements_backtest.txt
python backtest_exact_5m.py \
  --data-root /path/to/recursively-extracted-data \
  --symbols BTC,SOL \
  --start 2026-02-01 \
  --end 2026-06-01
```

The runner reuses the live Pipeline/Regime/Bias/Entry logic. It enters at the next 5M open, models 0.05% adverse slippage, 0.10% fee per fill, partial TP1, fee-aware runner stop, and assumes the stop occurs first when TP and SL are both inside one candle.
