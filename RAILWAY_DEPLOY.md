# Railway trading bot deployment

This project includes `railway.json` and `Dockerfile.bot` for running `app/run_bot.py` as a Railway worker. The worker does not need a public web port; it runs continuously and sends operational alerts to Telegram.

## Required Railway variables

Start safely in paper mode first:

```env
PAPER_TRADING=true
LIVE_TRADING_CONFIRMED=false
EXCHANGE=binance
SYMBOLS=BTC/USDT
INTERVAL_SECONDS=60
TELEGRAM_BOT_TOKEN=123456:your_bot_token
TELEGRAM_CHAT_ID=123456789
```

For live mode, set all exchange credentials and explicitly confirm live trading:

```env
PAPER_TRADING=false
LIVE_TRADING_CONFIRMED=true
EXCHANGE_API_KEY=...
EXCHANGE_API_SECRET=...
# OKX only:
EXCHANGE_PASSPHRASE=...
```

## Telegram checks

On every Railway start, the bot sends a `Railway Trading Bot Deploy Started` message with mode, exchange, symbols, and Railway environment. It also sends bot start/stop, signal, order, SL/TP, drawdown halt, and fatal error alerts.

Supported Telegram commands:

- `/status`
- `/positions`
- `/trades`
- `/balance`
- `/stats`
- `/insights`
- `/start_bot`
- `/stop_bot`
- `/help`

## Deploy steps

1. Create a Railway project from this repository.
2. Railway will use `railway.json`, build with `Dockerfile.bot`, and start `python run_bot.py`.
3. Add the variables above in Railway Variables.
4. Deploy with `PAPER_TRADING=true` and verify the Telegram startup message.
5. Only after paper testing, switch to live mode by setting `PAPER_TRADING=false` and `LIVE_TRADING_CONFIRMED=true`.
