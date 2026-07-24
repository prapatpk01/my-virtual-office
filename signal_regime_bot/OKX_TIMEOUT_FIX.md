# DUALCORE V3.0.1 — OKX timeout fix

## What the error meant

`ccxt.base.errors.RequestTimeout` on
`GET /api/v5/market/candles` means Railway/CCXT did not receive the OKX public
candle response before the HTTP timeout. The URL and BTC-USDT-SWAP instrument
were valid; this was a transport/latency failure, not an EMA/SMC strategy error
and not an invalid API key.

## Changes

- Increased CCXT request timeout to 30 seconds.
- Added four bounded retries with exponential backoff and jitter for network,
  timeout, rate-limit and temporary exchange-availability errors.
- Added an independent native OKX REST candle fallback after CCXT retries are
  exhausted.
- Added one public-request lock to prevent overlapping market-data requests.
- Market candles are now cached by exchange candle bucket. A 4H candle is no
  longer downloaded every 30 seconds; each timeframe is refreshed only when a
  new candle bucket begins.
- Added bounded last-known-good cache fallback. New entries are skipped when no
  complete usable dataset exists; stale data is never used indefinitely.
- Removed duplicate 5M/15M downloads from SpikeGuard. It now reuses the frames
  already fetched for the symbol.
- A single timeout no longer produces a Telegram stack trace. The bot alerts
  only after three complete failed symbol cycles and sends a recovery message
  when OKX data returns.
- Existing native OKX SL/TP orders remain active during a market-data outage.

## Deployment

Replace the existing project files with this package and redeploy. No new
Railway environment variables are required.
