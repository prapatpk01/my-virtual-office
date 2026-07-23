# Deploy

Replace the existing project files with the files in this bundle and redeploy Railway. Existing environment variable names are preserved; no new Railway variables are required for the V3 strategy defaults.

Production files:

```text
main.py
config.py
pipeline.py
regime_engine.py
bias_engine.py
entry_engine.py
indicators.py
risk_manager.py
position_manager.py
telegram_notifier.py
```

Backtest:

```bash
pip install -r requirements_backtest.txt
BACKTEST_DATA_ROOT=/path/to/extracted/data \
python backtest_expert_multimode.py \
  --symbols BTC,SOL \
  --start 2026-04-01 \
  --end 2026-04-08
```

Before deployment, remove an old incompatible strategy-state file only when intentionally resetting all entry locks:

```text
state/entry_engine_state.json
```

The V3 loader can restore JSON signal keys correctly, so routine upgrades do not require deleting state.
