# V3.2.1 Startup Resume Fix

- Positions found during startup are now classified as **resumed from OKX**.
- Telegram shows `Position resumed from OKX` instead of `Adopted untracked position`.
- Positions that appear unexpectedly after startup are still labelled as adopted/unexpected.
- Live OKX entry, amount, SL and TP remain authoritative.
- Recovered state is saved immediately to `STATE_DIR/open_positions.json`.
- With a Railway Volume, the next restart restores full lifecycle metadata.
- Without a Volume, the bot still safely resumes from OKX and no longer uses misleading wording.
