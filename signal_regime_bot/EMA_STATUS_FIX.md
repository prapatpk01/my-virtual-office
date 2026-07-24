# V3.0.2 EMA Status Display Fix

## Symptom
Railway status displayed `EMA8/13=0.0000/0.0000` while the reason was
`5M bar already processed`.

## Root cause
The duplicate-candle safety guard returned `EntryResult` before populating the
EMA fields, so the dataclass defaults (`0.0`) were printed. This was a display
problem, not an EMA-calculation or market-data failure.

## Fix
- Calculate EMA8, EMA13 and MACD histogram before the duplicate-bar guard.
- Return the real values even when duplicate evaluation is blocked.
- Preserve the last meaningful setup/score/reason in the 5-minute status log
  instead of overwriting it on every poll with `5M bar already processed`.
- Increased status display precision to six decimal places.

The duplicate-order protection remains active and trading logic is unchanged.
