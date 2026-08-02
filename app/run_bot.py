"""Adaptive Bot v12 runner. Paper mode is the safe default."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict

import ccxt

from trading.connectors.binance_conn import BinanceConnector
from trading.adaptive_trading_bot import TradingBot
from trading.indicator_engine import compute


BUILD_ID = "adaptive-v12-telegram-2026-08-03"

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


def first_env(*keys: str) -> str:
    for key in keys:
        value = os.getenv(key, "").strip()
        if value:
            return value
    return ""


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


def format_trade_message(order_type: str, payload: Dict[str, Any], paper: bool) -> str:
    symbol = payload.get("symbol", "UNKNOWN")
    direction = payload.get("direction", "")
    mode = "PAPER" if paper else "LIVE"

    if order_type.startswith("OPEN_"):
        entry = float(payload.get("entry", 0.0))
        sl = float(payload.get("sl", 0.0))
        tp = float(payload.get("tp", 0.0))
        size = float(payload.get("size", 0.0))
        strategy = payload.get("strategy", "unknown")
        return (
            f"🟢 [{mode}] OPEN {direction} {symbol}\n"
            f"Strategy: {strategy}\n"
            f"Entry: {entry:,.6f}\n"
            f"SL: {sl:,.6f}\n"
            f"TP: {tp:,.6f}\n"
            f"Size: {size:.6f}"
        )

    price = float(payload.get("price", 0.0))
    reason = payload.get("reason", "unknown")
    return (
        f"🔴 [{mode}] CLOSE {direction} {symbol}\n"
        f"Price: {price:,.6f}\n"
        f"Reason: {reason}"
    )


async def main() -> None:
    paper = env_bool("PAPER_TRADING", True) or os.getenv("TRADING_MODE", "").lower() == "paper"
    symbols = [s.strip() for s in os.getenv(
        "SYMBOLS", "BTC/USDT:USDT,ETH/USDT:USDT,SOL/USDT:USDT,XRP/USDT:USDT"
    ).split(",") if s.strip()]
    leverage = int(os.getenv("LEVERAGE", "20"))
    margin_usdt = float(os.getenv("ADAPTIVE_MARGIN_USDT", "20"))
    interval_seconds = int(os.getenv("INTERVAL_SECONDS", "60"))

    telegram_token = first_env("TELEGRAM_BOT_TOKEN", "TELEGRAM_TOKEN", "TG_BOT_TOKEN")
    telegram_chat_id = first_env("TELEGRAM_CHAT_ID", "TG_CHAT_ID", "CHAT_ID")
    telegram_enabled = bool(telegram_token and telegram_chat_id)
    telegram_queue: asyncio.Queue[str] = asyncio.Queue()

    def send_telegram_blocking(text: str) -> None:
        url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
        body = urllib.parse.urlencode({
            "chat_id": telegram_chat_id,
            "text": text,
            "disable_web_page_preview": "true",
        }).encode("utf-8")
        request = urllib.request.Request(url, data=body, method="POST")
        with urllib.request.urlopen(request, timeout=15) as response:
            raw = response.read().decode("utf-8", errors="replace")
            result = json.loads(raw)
            if not result.get("ok"):
                raise RuntimeError(result.get("description", "Telegram send failed"))

    async def telegram_worker() -> None:
        while True:
            text = await telegram_queue.get()
            try:
                await asyncio.to_thread(send_telegram_blocking, text)
                logger.info("Telegram notification sent")
            except asyncio.CancelledError:
                raise
            except Exception as error:
                logger.warning("Telegram notification failed: %s", error)
            finally:
                telegram_queue.task_done()

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

    def executor(order_type: str, payload: Dict[str, Any]):
        if telegram_enabled:
            telegram_queue.put_nowait(format_trade_message(order_type, payload, paper))

        if paper:
            logger.info("[PAPER] %s %s", order_type, payload)
            return {"paper": True}
        if live_adapter is None:
            raise RuntimeError("Live adapter is not initialized")
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

    last_timestamp: Dict[str, Any] = {}
    disabled_symbols = set()
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for stop_signal in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(stop_signal, stop_event.set)
        except NotImplementedError:
            pass

    telegram_task = asyncio.create_task(telegram_worker()) if telegram_enabled else None

    logger.info("=" * 64)
    logger.info("Adaptive Bot v12 | build=%s | mode=%s", BUILD_ID, "PAPER" if paper else "LIVE")
    logger.info("margin=$%.2f leverage=%dx notional≈$%.2f", margin_usdt, leverage, margin_usdt * leverage)
    logger.info("symbols=%s", symbols)
    logger.info("entries: Trend Pullback | CDC Transition | BB Breakout")
    logger.info("telegram=%s", "CONNECTED" if telegram_enabled else "DISABLED (missing token/chat id)")
    logger.info("=" * 64)

    if telegram_enabled:
        telegram_queue.put_nowait(
            "🤖 Adaptive Bot v12 started\n"
            f"Mode: {'PAPER' if paper else 'LIVE'}\n"
            f"Margin: ${margin_usdt:.2f}\n"
            f"Leverage: {leverage}x\n"
            f"Symbols: {', '.join(symbols)}"
        )

    try:
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
                        indicators_15m,
                        indicators_1h,
                        indicators_4h,
                        float(indicators_15m.get("close", 0.0)),
                    )
                    logger.info("[%s] %s", symbol, event if event else bot.last_signal)

                except ccxt.BadSymbol as error:
                    disabled_symbols.add(symbol)
                    logger.error("[%s] unsupported symbol; disabled: %s", symbol, error)
                    if telegram_enabled:
                        telegram_queue.put_nowait(f"⚠️ {symbol} disabled: unsupported symbol")
                except (ccxt.NetworkError, asyncio.TimeoutError) as error:
                    logger.warning("[%s] transient market-data error: %s", symbol, error)
                except Exception as error:
                    logger.exception("[%s] tick failed", symbol)
                    if telegram_enabled:
                        telegram_queue.put_nowait(f"❌ Adaptive v12 error\nSymbol: {symbol}\n{type(error).__name__}: {error}")

            try:
                await asyncio.wait_for(stop_event.wait(), timeout=interval_seconds)
            except asyncio.TimeoutError:
                pass
    finally:
        if telegram_enabled:
            telegram_queue.put_nowait("⏹ Adaptive Bot v12 stopped")
            try:
                await asyncio.wait_for(telegram_queue.join(), timeout=5)
            except asyncio.TimeoutError:
                pass

        if telegram_task:
            telegram_task.cancel()
            try:
                await telegram_task
            except asyncio.CancelledError:
                pass

        try:
            exchange = getattr(connector, "exchange", None) or getattr(connector, "_exchange", None)
            if exchange is not None and hasattr(exchange, "close"):
                await exchange.close()
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())
