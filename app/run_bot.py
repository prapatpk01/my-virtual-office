"""Adaptive Bot v12 runner. Paper mode is the safe default."""
from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from datetime import datetime, timezone

import ccxt

from trading.connectors.binance_conn import BinanceConnector
from trading.adaptive_trading_bot import TradingBot
from trading.indicator_engine import compute


BUILD_ID = "adaptive-v12-stdout-logging-2026-08-02"

# Railway visually marks stderr output in red. Python logging defaults to stderr,
# even for INFO messages. Route the bot's complete log stream to stdout so
# normal status messages such as SLEEP_MODE appear as ordinary INFO logs.
# force=True also replaces handlers that may have been installed by imports.
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger("adaptive_v12")


def env_bool(key: str, default: bool = False) -> bool:
    value = os.getenv(key)
    return default if value is None else value.lower() in ("1", "true", "yes", "on")


def fx_entry_window_open(now: datetime) -> bool:
    try:
        from zoneinfo import ZoneInfo
        new_york = now.astimezone(ZoneInfo("America/New_York"))
    except Exception:
        new_york = now
    weekday = new_york.weekday()
    hour = new_york.hour
    if weekday < 4:
        return True
    if weekday == 4:
        return hour < 17
    if weekday == 5:
        return False
    return hour >= 13


async def main() -> None:
    paper = env_bool("PAPER_TRADING", True) or os.getenv("TRADING_MODE", "").lower() == "paper"
    symbols = [s.strip() for s in os.getenv(
        "SYMBOLS", "BTC/USDT:USDT,ETH/USDT:USDT,SOL/USDT:USDT,XRP/USDT:USDT"
    ).split(",") if s.strip()]
    leverage = int(os.getenv("LEVERAGE", "20"))
    margin_usdt = float(os.getenv("ADAPTIVE_MARGIN_USDT", "20"))
    interval_seconds = int(os.getenv("INTERVAL_SECONDS", "60"))

    connector = BinanceConnector(
        api_key="" if paper else os.getenv("EXCHANGE_API_KEY", ""),
        api_secret="" if paper else os.getenv("EXCHANGE_API_SECRET", ""),
        paper=True,
        exchange_id=os.getenv("EXCHANGE", "okx"),
        passphrase="" if paper else os.getenv("EXCHANGE_PASSPHRASE", ""),
        leverage=leverage,
    )

    live_adapter = None
    if not paper:
        from trading.connectors.okx_adapter import OKXAdapter
        live_adapter = OKXAdapter(
            api_key=os.getenv("EXCHANGE_API_KEY", ""),
            api_secret=os.getenv("EXCHANGE_API_SECRET", ""),
            api_passphrase=os.getenv("EXCHANGE_PASSPHRASE", ""),
            paper=False,
            leverage=leverage,
        )

    def executor(order_type, payload):
        if paper:
            logger.info("[PAPER] %s %s", order_type, payload)
            return {"paper": True}
        return live_adapter.execute(order_type, payload)

    state_directory = os.getenv("BOT_STATE_DIR", "/tmp/adaptive_v12")
    bots = {
        symbol: TradingBot(
            symbol=symbol,
            margin_usdt=margin_usdt,
            leverage=leverage,
            paper=paper,
            state_file=os.path.join(
                state_directory, symbol.replace("/", "_").replace(":", "_") + ".json"
            ),
            execution_callback=executor,
        )
        for symbol in symbols
    }

    last_timestamp = {}
    disabled_symbols = set()
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for stop_signal in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(stop_signal, stop_event.set)
        except NotImplementedError:
            pass

    logger.info("=" * 64)
    logger.info("Adaptive Bot v12 | build=%s | mode=%s", BUILD_ID, "PAPER" if paper else "LIVE")
    logger.info("margin=$%.2f leverage=%dx notional≈$%.2f", margin_usdt, leverage, margin_usdt * leverage)
    logger.info("symbols=%s", symbols)
    logger.info("entries: Trend Pullback | CDC Transition | BB Breakout")
    logger.info("=" * 64)

    while not stop_event.is_set():
        entries_allowed = fx_entry_window_open(datetime.now(timezone.utc))
        for symbol in symbols:
            if symbol in disabled_symbols:
                continue
            try:
                candles_15m = await connector.fetch_ohlcv(symbol, "15m", 300)
                candles_1h = await connector.fetch_ohlcv(symbol, "1h", 200)
                candles_4h = await connector.fetch_ohlcv(symbol, "4h", 200)
                if not candles_15m or not candles_1h or not candles_4h:
                    logger.warning("[%s] waiting for candle data", symbol)
                    continue

                timestamp = getattr(candles_15m[-1], "timestamp", 0)
                if timestamp == last_timestamp.get(symbol):
                    continue
                last_timestamp[symbol] = timestamp

                indicators_15m = compute(candles_15m)
                indicators_1h = compute(candles_1h)
                indicators_4h = compute(candles_4h)
                bot = bots[symbol]

                if not entries_allowed and not bot.position_open:
                    logger.info("[%s] SLEEP_MODE — no new position", symbol)
                    continue

                event = bot.on_bar(
                    indicators_15m, indicators_1h, indicators_4h,
                    float(indicators_15m.get("close", 0.0)),
                )
                logger.info("[%s] %s", symbol, event if event else bot.last_signal)

            except ccxt.BadSymbol as error:
                disabled_symbols.add(symbol)
                logger.error("[%s] unsupported symbol; disabled: %s", symbol, error)
            except (ccxt.NetworkError, asyncio.TimeoutError) as error:
                logger.warning("[%s] transient market-data error: %s", symbol, error)
            except Exception:
                logger.exception("[%s] tick failed", symbol)

        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval_seconds)
        except asyncio.TimeoutError:
            pass

    try:
        exchange = getattr(connector, "exchange", None) or getattr(connector, "_exchange", None)
        if exchange is not None and hasattr(exchange, "close"):
            await exchange.close()
    except Exception:
        pass


if __name__ == "__main__":
    asyncio.run(main())
