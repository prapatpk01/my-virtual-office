# V3.2 Persistent Position State

- Atomically stores open position lifecycle in `STATE_DIR/open_positions.json`.
- Restores TP1 hit, remaining amount, banked realized PnL, entry fee, setup metadata, entry time and last exit-check bar.
- Stores AI Exit WATCH persistence in `STATE_DIR/ai_exit_state.json`.
- On startup, restores local state first and then validates it against live OKX.
- OKX remains authoritative for live position existence, amount, entry, SL and TP2.
- Stale local positions are removed automatically.
- TP1 state is never reset merely because Railway restarts.
- Writes are atomic (`fsync` + `os.replace`) to reduce corruption risk.

For persistence across a full Railway redeploy/rebuild, `STATE_DIR` should point to a mounted Railway Volume. Without a volume, the bot still safely reconstructs positions from OKX, but local-only fields can be unavailable after the container filesystem is replaced.
