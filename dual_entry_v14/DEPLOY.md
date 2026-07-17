# Railway Deployment — DUAL ENTRY PRECISION V1.4

## 1. Files

`Dockerfile.dual_entry` at the repo root builds this package:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY dual_entry_v14/ dual_entry_v14/
RUN pip install --no-cache-dir numpy ccxt aiohttp
# Persistent state dir — mount a Railway volume here (see §3)
ENV STATE_DIR=/data/state
CMD ["python", "-m", "dual_entry_v14.main"]
```

## 2. Environment variables (Railway → Variables)

| Var | Example | Notes |
|---|---|---|
| `SYMBOLS` | `BTC/USDT:USDT,ETH/USDT:USDT` | ccxt unified symbols |
| `PAPER_TRADING` | `true` | **start paper**; `false` for live |
| `OKX_API_KEY` / `OKX_SECRET_KEY` / `OKX_PASSPHRASE` | … | trade-enabled key |
| `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` | … | alerts |
| `RISK_PER_TRADE` | `0.01` | default 0.03; >0.02 logs a loud warning |
| `MAX_POSITIONS` | `2` | portfolio cap |
| `LEVERAGE` | `10` | margin only — risk comes from stops |
| `STATE_DIR` | `/data/state` | must point at the volume |
| `POLL_INTERVAL_SEC` | `20` | main-loop cadence |

## 3. Persistent state (important)

Railway containers are ephemeral. Attach a **Railway Volume** mounted at
`/data` so `SymbolState`, the execution journal and the trade log survive
redeploys. The system still recovers positions without it (exchange
reconciliation is the source of truth), but journal/idempotency history
and trade statistics would reset on every deploy.

## 4. OKX account prerequisites

- Hedge mode (`long_short_mode`) on the futures account, isolated margin.
- API key with trade permission (no withdrawal), IP-whitelist Railway's
  egress if possible.
- The bot sets leverage per symbol on first order.

## 5. Rollout checklist

1. Deploy with `PAPER_TRADING=true`, watch the startup Telegram message.
2. Confirm view-log lines appear per symbol (SCANNING / regimes).
3. Let it paper-trade ≥ 1–2 weeks; check `/data/state/trades.jsonl` stats
   vs expectations (8–15 trades/mo/symbol in cooperative regimes).
4. Flip `PAPER_TRADING=false` with small `RISK_PER_TRADE` (0.005–0.01).
5. Scale risk only after the module gate keeps both engines ACTIVE.

## 6. Operational notes

- Every rejection carries a reason code — check Railway logs before
  assuming "the bot isn't trading" (usually REJECT_CHOP / thresholds).
- 3 consecutive losses → automatic ≥30-min cooldown per symbol.
- A module whose PF decays is automatically risk-reduced then paused
  (shadow mode keeps evaluating it; it re-opens only on improvement).
- Emergency protection: if SL/TP can't be attached after retries, the
  position is closed immediately and a CRITICAL alert is sent.
