"""Adaptive Bot v13 runner: closed bars, Telegram charts, stats, paper/live execution."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
import time
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

import ccxt

from trading.connectors.binance_conn import BinanceConnector
from trading.adaptive_trading_bot import TradingBot
from trading.indicator_engine import compute

BUILD_ID = "adaptive-v13-structure-trend-2026-08-04"

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger("adaptive_v13")


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
        ny = now.astimezone(ZoneInfo("America/New_York"))
    except Exception:
        ny = now
    if ny.weekday() < 4:
        return True
    if ny.weekday() == 4:
        return ny.hour < 17
    if ny.weekday() == 5:
        return False
    return ny.hour >= 13


def load_json(path: str, default: Any) -> Any:
    try:
        with open(path, encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return default


def save_json(path: str, value: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temp = path + ".tmp"
    with open(temp, "w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False)
    os.replace(temp, path)


def candle_field(candle: Any, name: str, index: int) -> float:
    value = getattr(candle, name, None)
    if value is None and isinstance(candle, dict):
        value = candle.get(name)
    if value is None and isinstance(candle, (list, tuple)) and len(candle) > index:
        value = candle[index]
    return float(value or 0.0)


def candle_timestamp(candle: Any) -> Any:
    value = getattr(candle, "timestamp", None)
    if value is None and isinstance(candle, dict):
        value = candle.get("timestamp")
    if value is None and isinstance(candle, (list, tuple)) and candle:
        value = candle[0]
    return value


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes = seconds // 60
    if days:
        return f"{days}d {hours}h {minutes}m"
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def format_trade(order_type: str, payload: Dict[str, Any], paper: bool) -> str:
    mode = "PAPER" if paper else "LIVE"
    direction = str(payload.get("direction", ""))
    symbol = str(payload.get("symbol", ""))
    if order_type.startswith("OPEN_"):
        return (
            f"{'🟢' if direction == 'LONG' else '🔴'} [{mode}] OPEN {direction} {symbol}\n"
            f"Strategy: {payload.get('strategy', 'structure_trend')}\n"
            f"Entry: {float(payload.get('entry', 0)):,.6f}\n"
            f"SL: {float(payload.get('sl', 0)):,.6f}\n"
            f"TP: {float(payload.get('tp', 0)):,.6f} (2.00R)\n"
            f"Size: {float(payload.get('size', 0)):.6f}"
        )
    pnl = float(payload.get("pnl", 0.0))
    return (
        f"{'✅' if pnl >= 0 else '❌'} [{mode}] CLOSE {direction} {symbol}\n"
        f"Price: {float(payload.get('price', 0)):,.6f}\n"
        f"Reason: {payload.get('reason', 'unknown')}\n"
        f"PnL: ${pnl:+.2f} ({float(payload.get('r_multiple', 0)):+.2f}R)"
    )


def render_stats(
    trades: List[Dict[str, Any]],
    bots: Dict[str, TradingBot],
    current_prices: Dict[str, float],
    paper: bool,
    margin_usdt: float,
) -> str:
    closed = [trade for trade in trades if trade.get("event") == "CLOSE"]
    wins = [trade for trade in closed if float(trade.get("pnl", 0)) > 0]
    losses = [trade for trade in closed if float(trade.get("pnl", 0)) < 0]
    net = sum(float(trade.get("pnl", 0)) for trade in closed)
    win_rate = 100.0 * len(wins) / len(closed) if closed else 0.0

    open_rows: List[Dict[str, Any]] = []
    floating_total = 0.0
    open_risk_total = 0.0
    long_count = 0
    short_count = 0
    now = time.time()

    for symbol, bot in bots.items():
        position = bot.position
        if position is None:
            continue
        current = float(current_prices.get(symbol, position.entry))
        floating = (
            (current - position.entry) * position.size
            if position.direction == "LONG"
            else (position.entry - current) * position.size
        )
        initial_risk = abs(position.entry - position.initial_sl) * position.size
        current_r = floating / initial_risk if initial_risk > 0 else 0.0
        floating_total += floating
        open_risk_total += initial_risk
        long_count += int(position.direction == "LONG")
        short_count += int(position.direction == "SHORT")
        open_rows.append({
            "symbol": symbol,
            "direction": position.direction,
            "entry": position.entry,
            "current": current,
            "sl": position.sl,
            "tp": position.tp,
            "strategy": position.strategy,
            "held": format_duration(now - position.opened_at),
            "pnl": floating,
            "r": current_r,
            "be": position.be_moved,
        })

    lines = [
        "📊 Adaptive Bot v13 Stats",
        "",
        f"Mode: {'PAPER' if paper else 'LIVE'}",
        "",
        f"OPEN POSITIONS ({len(open_rows)})",
        "――――――――――――――――",
    ]
    if open_rows:
        for row in open_rows:
            icon = "🟢" if row["direction"] == "LONG" else "🔴"
            lines += [
                f"{icon} {row['symbol'].split('/')[0]} {row['direction']}",
                f"Entry : {row['entry']:,.6f}",
                f"Now   : {row['current']:,.6f}",
                f"PnL   : ${row['pnl']:+.2f} ({row['r']:+.2f}R)",
                f"SL    : {row['sl']:,.6f}{' (BE)' if row['be'] else ''}",
                f"TP    : {row['tp']:,.6f}",
                f"Strategy: {row['strategy']}",
                f"Held  : {row['held']}",
                "",
            ]
    else:
        lines += ["No open positions", ""]

    lines += [
        "EXPOSURE",
        "――――――――――――――――",
        f"Long / Short : {long_count} / {short_count}",
        f"Margin Used  : ${len(open_rows) * margin_usdt:.2f}",
        f"Floating PnL : ${floating_total:+.2f}",
        f"Initial Risk : ${open_risk_total:.2f}",
        "",
        "OVERALL",
        "――――――――――――――――",
        f"Trades   : {len(closed)}  ({len(wins)}W / {len(losses)}L)",
        f"Win rate : {win_rate:.0f}%",
        f"TP hit   : {sum(t.get('reason') == 'TP' for t in closed)}/{len(closed)}",
        f"SL hit   : {sum(t.get('reason') == 'SL' for t in closed)}/{len(closed)}",
        f"BE exit  : {sum(t.get('reason') == 'BE' for t in closed)}/{len(closed)}",
        f"Structure exit: {sum(t.get('reason') == 'STRUCTURE_EXIT' for t in closed)}/{len(closed)}",
        f"Net PnL  : ${net:+.2f}",
    ]

    if closed:
        lines += ["", "BY SYMBOL", "――――――――――――――――"]
        for symbol in sorted({str(t.get("symbol", "")) for t in closed}):
            rows = [t for t in closed if t.get("symbol") == symbol]
            row_wins = sum(float(t.get("pnl", 0)) > 0 for t in rows)
            row_net = sum(float(t.get("pnl", 0)) for t in rows)
            lines.append(
                f"{symbol.split('/')[0]:<6} {len(rows):>3} trades "
                f"{100 * row_wins / len(rows):>3.0f}%WR ${row_net:+.2f}"
            )
        lines += ["", "LAST 5 TRADES", "――――――――――――――――"]
        for index, trade in enumerate(reversed(closed[-5:]), 1):
            pnl = float(trade.get("pnl", 0))
            icon = "✅" if pnl > 0 else ("❌" if pnl < 0 else "➖")
            lines.append(
                f"{index}. {icon} {str(trade.get('symbol', '')).split('/')[0]} "
                f"{trade.get('direction', '')} ${pnl:+.2f} — {trade.get('reason', '')}"
            )
    return "\n".join(lines)


def create_chart(candles: List[Any], payload: Dict[str, Any], path: str) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except Exception as error:
        logger.error("Chart dependency unavailable: %s", error)
        return False

    rows = list(candles[-80:])
    if len(rows) < 20:
        logger.error("Not enough candles for mandatory chart")
        return False

    figure, (price_ax, volume_ax) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True,
        gridspec_kw={"height_ratios": [4, 1]},
    )
    for index, candle in enumerate(rows):
        open_price = candle_field(candle, "open", 1)
        high = candle_field(candle, "high", 2)
        low = candle_field(candle, "low", 3)
        close = candle_field(candle, "close", 4)
        volume = candle_field(candle, "volume", 5)
        color = "#26a69a" if close >= open_price else "#ef5350"
        price_ax.vlines(index, low, high, color=color, linewidth=1)
        price_ax.add_patch(Rectangle(
            (index - 0.3, min(open_price, close)),
            0.6,
            max(abs(close - open_price), max(close, 1) * 1e-6),
            facecolor=color,
            edgecolor=color,
        ))
        volume_ax.bar(index, volume, width=0.7, color=color)

    for key, label in (("entry", "ENTRY"), ("sl", "SL"), ("tp", "TP 2R")):
        value = float(payload.get(key, 0))
        price_ax.axhline(value, linestyle="--", linewidth=1.3, label=f"{label} {value:,.4f}")

    price_ax.scatter(
        len(rows) - 1,
        float(payload.get("entry", 0)),
        marker="^" if payload.get("direction") == "LONG" else "v",
        s=80,
        zorder=5,
    )
    price_ax.legend(loc="best")
    price_ax.grid(alpha=0.2)
    volume_ax.grid(alpha=0.15)
    price_ax.set_title(
        f"{payload.get('symbol')} {payload.get('direction')} | 15m | structure_trend"
    )
    price_ax.set_ylabel("Price")
    volume_ax.set_ylabel("Volume")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return os.path.exists(path) and os.path.getsize(path) > 0


async def main() -> None:
    paper = env_bool("PAPER_TRADING", True) or os.getenv("TRADING_MODE", "").lower() == "paper"
    symbols = [symbol.strip() for symbol in os.getenv(
        "SYMBOLS",
        "BTC/USDT:USDT,ETH/USDT:USDT,SOL/USDT:USDT,XRP/USDT:USDT",
    ).split(",") if symbol.strip()]
    leverage = int(os.getenv("LEVERAGE", "20"))
    margin_usdt = float(os.getenv("ADAPTIVE_MARGIN_USDT", "20"))
    interval_seconds = int(os.getenv("INTERVAL_SECONDS", "60"))
    max_positions = int(os.getenv("MAX_POSITIONS", "2"))

    token = first_env("TELEGRAM_BOT_TOKEN", "TELEGRAM_TOKEN", "TG_BOT_TOKEN")
    chat_id = first_env("TELEGRAM_CHAT_ID", "TG_CHAT_ID", "CHAT_ID")
    telegram_enabled = bool(token and chat_id)
    telegram_queue: asyncio.Queue = asyncio.Queue()
    update_offset = 0

    state_dir = os.getenv("BOT_STATE_DIR", "/tmp/adaptive_v13")
    ledger_file = os.getenv("TRADE_LEDGER_FILE", os.path.join(state_dir, "trade_ledger_v13.json"))
    trades: List[Dict[str, Any]] = load_json(ledger_file, [])
    latest_candles: Dict[str, List[Any]] = {}
    current_prices: Dict[str, float] = {}

    def telegram_api(method: str, fields: Dict[str, Any], photo_path: str = "") -> Dict[str, Any]:
        url = f"https://api.telegram.org/bot{token}/{method}"
        if not photo_path:
            request = urllib.request.Request(
                url,
                data=urllib.parse.urlencode(fields).encode(),
                method="POST",
            )
        else:
            boundary = "----Adaptive" + uuid.uuid4().hex
            parts: List[bytes] = []
            for key, value in fields.items():
                parts.extend([
                    f"--{boundary}\r\n".encode(),
                    f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode(),
                    str(value).encode(),
                    b"\r\n",
                ])
            with open(photo_path, "rb") as file:
                image = file.read()
            parts.extend([
                f"--{boundary}\r\n".encode(),
                b'Content-Disposition: form-data; name="photo"; filename="chart.png"\r\n',
                b"Content-Type: image/png\r\n\r\n",
                image,
                b"\r\n",
                f"--{boundary}--\r\n".encode(),
            ])
            request = urllib.request.Request(
                url,
                data=b"".join(parts),
                method="POST",
                headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            )
        with urllib.request.urlopen(request, timeout=20) as response:
            result = json.loads(response.read().decode())
        if not result.get("ok"):
            raise RuntimeError(result.get("description", f"Telegram {method} failed"))
        return result

    async def telegram_worker() -> None:
        while True:
            item = await telegram_queue.get()
            try:
                if item["kind"] == "photo":
                    await asyncio.to_thread(
                        telegram_api,
                        "sendPhoto",
                        {"chat_id": chat_id, "caption": item["caption"]},
                        item["path"],
                    )
                    try:
                        os.remove(item["path"])
                    except OSError:
                        pass
                else:
                    await asyncio.to_thread(
                        telegram_api,
                        "sendMessage",
                        {
                            "chat_id": chat_id,
                            "text": item["text"],
                            "disable_web_page_preview": "true",
                        },
                    )
            except asyncio.CancelledError:
                raise
            except Exception as error:
                logger.warning("Telegram send failed: %s", error)
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
        if order_type.startswith("OPEN_") and telegram_enabled:
            chart_path = f"/tmp/v13_{int(time.time())}_{uuid.uuid4().hex[:6]}.png"
            if not create_chart(
                latest_candles.get(str(payload.get("symbol")), []),
                payload,
                chart_path,
            ):
                raise RuntimeError("Mandatory Telegram entry chart could not be created")
            telegram_queue.put_nowait({
                "kind": "photo",
                "path": chart_path,
                "caption": format_trade(order_type, payload, paper),
            })
        elif telegram_enabled:
            telegram_queue.put_nowait({
                "kind": "text",
                "text": format_trade(order_type, payload, paper),
            })

        if paper:
            logger.info("[PAPER] %s %s", order_type, payload)
            return {"paper": True}
        if live_adapter is None:
            raise RuntimeError("Live adapter is not initialized")
        return live_adapter.execute(order_type, payload)

    bots = {
        symbol: TradingBot(
            symbol=symbol,
            margin_usdt=margin_usdt,
            leverage=leverage,
            paper=paper,
            state_file=os.path.join(
                state_dir,
                symbol.replace("/", "_").replace(":", "_") + ".json",
            ),
            execution_callback=executor,
        )
        for symbol in symbols
    }

    async def poll_commands() -> None:
        nonlocal update_offset
        if not telegram_enabled:
            return
        try:
            result = await asyncio.to_thread(
                telegram_api,
                "getUpdates",
                {"timeout": 0, "offset": update_offset},
            )
            for update in result.get("result", []):
                update_offset = max(update_offset, int(update.get("update_id", 0)) + 1)
                message = update.get("message") or {}
                if str((message.get("chat") or {}).get("id", "")) != str(chat_id):
                    continue
                text = str(message.get("text", "")).strip().lower()
                if text.startswith("/stats") or text.startswith("/restats"):
                    telegram_queue.put_nowait({
                        "kind": "text",
                        "text": render_stats(
                            trades,
                            bots,
                            current_prices,
                            paper,
                            margin_usdt,
                        ),
                    })
        except Exception as error:
            logger.warning("Telegram polling failed: %s", error)

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for stop_signal in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(stop_signal, stop_event.set)
        except NotImplementedError:
            pass

    telegram_task = asyncio.create_task(telegram_worker()) if telegram_enabled else None
    last_closed_timestamp: Dict[str, Any] = {}
    disabled_symbols = set()

    logger.info("=" * 64)
    logger.info("Adaptive Bot v13 | build=%s | mode=%s", BUILD_ID, "PAPER" if paper else "LIVE")
    logger.info(
        "telegram=%s chart=MANDATORY /stats=V13 max_positions=%s",
        "CONNECTED" if telegram_enabled else "DISABLED",
        max_positions,
    )
    logger.info("=" * 64)

    if telegram_enabled:
        telegram_queue.put_nowait({
            "kind": "text",
            "text": (
                f"🤖 Adaptive Bot v13 started\n"
                f"Mode: {'PAPER' if paper else 'LIVE'}\n"
                f"Logic: 4H Trend → 1H Quality → 15M Structure + EMA8/13\n"
                f"Margin: ${margin_usdt:.2f} | Leverage: {leverage}x\n"
                f"SL: Swing + ATR | TP: 2R | BE: 1R\n"
                f"Symbols: {', '.join(symbols)}\n"
                f"Command: /stats"
            ),
        })

    try:
        while not stop_event.is_set():
            await poll_commands()
            entries_allowed = fx_entry_window_open(datetime.now(timezone.utc))

            for symbol in symbols:
                if symbol in disabled_symbols:
                    continue
                try:
                    raw_15m = await connector.fetch_ohlcv(symbol, "15m", 300)
                    raw_1h = await connector.fetch_ohlcv(symbol, "1h", 200)
                    raw_4h = await connector.fetch_ohlcv(symbol, "4h", 200)
                    if len(raw_15m) < 82 or len(raw_1h) < 82 or len(raw_4h) < 82:
                        logger.info("[%s] WAIT candle warmup", symbol)
                        continue

                    candles_15m = list(raw_15m[:-1])
                    candles_1h = list(raw_1h[:-1])
                    candles_4h = list(raw_4h[:-1])
                    latest_candles[symbol] = candles_15m
                    closed_timestamp = candle_timestamp(candles_15m[-1])
                    if closed_timestamp == last_closed_timestamp.get(symbol):
                        continue
                    last_closed_timestamp[symbol] = closed_timestamp

                    i15 = compute(candles_15m)
                    i1 = compute(candles_1h)
                    i4 = compute(candles_4h)
                    if not i15 or not i1 or not i4:
                        logger.info("[%s] WAIT indicator warmup", symbol)
                        continue

                    price = float(i15["close"])
                    current_prices[symbol] = price
                    bot = bots[symbol]

                    if not bot.position_open:
                        if not entries_allowed:
                            logger.info("[%s] SLEEP_MODE — no new position", symbol)
                            continue
                        open_count = sum(int(item.position_open) for item in bots.values())
                        if open_count >= max_positions:
                            logger.info("[%s] WAIT max positions %s/%s", symbol, open_count, max_positions)
                            continue

                    event = bot.on_bar(i15, i1, i4, price)
                    if event:
                        trades.append({**event, "timestamp": time.time(), "version": "v13"})
                        save_json(ledger_file, trades[-2000:])
                    logger.info("[%s] %s", symbol, event if event else bot.last_signal)

                except ccxt.BadSymbol as error:
                    disabled_symbols.add(symbol)
                    logger.error("[%s] unsupported symbol: %s", symbol, error)
                except (ccxt.NetworkError, asyncio.TimeoutError) as error:
                    logger.warning("[%s] transient market-data error: %s", symbol, error)
                except Exception as error:
                    logger.exception("[%s] tick failed", symbol)
                    if telegram_enabled:
                        telegram_queue.put_nowait({
                            "kind": "text",
                            "text": (
                                f"❌ Adaptive v13 error\n"
                                f"Symbol: {symbol}\n"
                                f"{type(error).__name__}: {error}"
                            ),
                        })

            try:
                await asyncio.wait_for(stop_event.wait(), timeout=interval_seconds)
            except asyncio.TimeoutError:
                pass
    finally:
        if telegram_enabled:
            telegram_queue.put_nowait({"kind": "text", "text": "⏹ Adaptive Bot v13 stopped"})
            try:
                await asyncio.wait_for(telegram_queue.join(), timeout=8)
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
