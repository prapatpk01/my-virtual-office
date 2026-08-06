"""Adaptive Momentum v2 runner: 15M execution, Telegram charts and stats."""
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

import ccxt

from trading.connectors.binance_conn import BinanceConnector
from trading.adaptive_trading_bot import TradingBot, TP1_R, TP_R
from trading.indicator_engine import compute, ema

BUILD_ID = "adaptive-momentum-v2-2026-08-05"
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger("adaptive_momentum_v2")


def env_bool(key: str, default: bool = False) -> bool:
    value = os.getenv(key)
    return default if value is None else value.lower() in ("1", "true", "yes", "on")


def first_env(*keys: str) -> str:
    for key in keys:
        value = os.getenv(key, "").strip()
        if value:
            return value
    return ""


def fx_open(now: datetime) -> bool:
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
    return ny.hour >= 14


def load_json(path: str, default):
    try:
        return json.load(open(path, encoding="utf-8"))
    except Exception:
        return default


def save_json(path: str, value) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temp = path + ".tmp"
    with open(temp, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False)
    os.replace(temp, path)


def field(candle, name: str, index: int) -> float:
    value = getattr(candle, name, None)
    if value is None and isinstance(candle, dict):
        value = candle.get(name)
    if value is None and isinstance(candle, (list, tuple)) and len(candle) > index:
        value = candle[index]
    return float(value or 0.0)


def timestamp(candle):
    value = getattr(candle, "timestamp", None)
    if value is None and isinstance(candle, dict):
        value = candle.get("timestamp")
    if value is None and isinstance(candle, (list, tuple)) and candle:
        value = candle[0]
    return value


def duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes = seconds // 60
    return f"{days}d {hours}h {minutes}m" if days else f"{hours}h {minutes}m" if hours else f"{minutes}m"


def display_payload(value):
    """Round floats for logs only; trading calculations keep full precision."""
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, dict):
        return {key: display_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [display_payload(item) for item in value]
    if isinstance(value, tuple):
        return tuple(display_payload(item) for item in value)
    return value


def trade_text(order_type: str, payload: dict, paper: bool) -> str:
    mode = "PAPER" if paper else "LIVE"
    direction = str(payload.get("direction", ""))
    symbol = str(payload.get("symbol", ""))
    if order_type.startswith("OPEN_"):
        return (
            f"{'🟢' if direction == 'LONG' else '🔴'} MOMENTUM V2 {direction} — {symbol}\n"
            f"Mode: {mode} | TF: 15M\n"
            f"Trend: EMA20 {'>' if direction == 'LONG' else '<'} EMA50 ✅\n"
            f"Entry: EMA8 {'>' if direction == 'LONG' else '<'} EMA13 ✅\n"
            f"Trigger: {payload.get('trigger', 'MACD histogram flip')} ✅\n"
            f"MACD: {float(payload.get('macd', 0)):+.4f} / Signal {float(payload.get('macd_signal', 0)):+.4f}\n"
            f"ADX: {float(payload.get('adx', 0)):.1f} Rising ✅ | CHOP: {float(payload.get('chop', 0)):.1f}\n"
            f"Location: {float(payload.get('distance_ema13_atr', 0)):.2f} ATR from EMA13\n\n"
            f"Entry: {float(payload.get('entry', 0)):,.4f}\n"
            f"SL: {float(payload.get('sl', 0)):,.4f} ({float(payload.get('sl_pct', 0)):.2f}%)\n"
            f"TP1: {float(payload.get('tp1', 0)):,.4f} ({TP1_R:g}R · close 50% · SL→BE)\n"
            f"TP2: {float(payload.get('tp2', payload.get('tp', 0))):,.4f} ({TP_R:g}R)\n"
            f"Size: {float(payload.get('size', 0)):,.4f} | Risk: ${float(payload.get('risk_usdt', 0)):.2f}"
        )
    pnl = float(payload.get("pnl", 0))
    return (
        f"{'✅' if pnl >= 0 else '❌'} [{mode}] {payload.get('reason', 'CLOSE')} {direction} {symbol}\n"
        f"Price: {float(payload.get('price', 0)):,.4f}\n"
        f"PnL: ${pnl:+.2f} ({float(payload.get('r_multiple', 0)):+.2f}R)"
    )


def stats_text(trades, bots, prices, paper: bool, margin: float) -> str:
    closed = [trade for trade in trades if trade.get("event") == "CLOSE"]
    partials = [trade for trade in trades if trade.get("event") == "PARTIAL"]
    wins = [trade for trade in closed if float(trade.get("pnl", 0)) > 0]
    net = sum(float(trade.get("pnl", 0)) for trade in closed + partials)
    lines = [
        "📊 Adaptive Momentum v2 Stats", "",
        f"Mode: {'PAPER' if paper else 'LIVE'}",
        "Logic: EMA20/50 → EMA8/13 → MACD Hist Flip → ADX/CHOP → Location", "",
        f"OPEN POSITIONS ({sum(int(bot.position_open) for bot in bots.values())})",
        "――――――――――――――――",
    ]
    count = 0
    floating = 0.0
    for symbol, bot in bots.items():
        position = bot.position
        if not position:
            continue
        count += 1
        current = float(prices.get(symbol, position.entry))
        pnl = ((current - position.entry) * position.size if position.direction == "LONG"
               else (position.entry - current) * position.size)
        floating += pnl
        lines += [
            f"{'🟢' if position.direction == 'LONG' else '🔴'} {symbol.split('/')[0]} {position.direction}",
            f"Entry : {position.entry:,.4f}", f"Now   : {current:,.4f}",
            f"PnL   : ${pnl:+.2f}",
            f"SL    : {position.sl:,.4f}{' (BE)' if position.be_moved else ''}",
            f"TP1   : {position.tp1:,.4f} {'✅' if position.tp1_hit else ''}",
            f"TP2   : {position.tp2:,.4f}", f"Trigger: {position.trigger}",
            f"Held  : {duration(time.time() - position.opened_at)}", "",
        ]
    if not count:
        lines += ["No open positions", ""]
    win_rate = 100 * len(wins) / len(closed) if closed else 0
    lines += [
        "EXPOSURE", "――――――――――――――――",
        f"Margin Used  : ${count * margin:.2f}", f"Floating PnL : ${floating:+.2f}", "",
        "OVERALL", "――――――――――――――――",
        f"Trades   : {len(closed)} ({len(wins)}W / {len(closed) - len(wins)}L)",
        f"Win rate : {win_rate:.0f}%", f"TP1 hit  : {len(partials)}",
        f"TP2 hit  : {sum(t.get('reason') == 'TP2' for t in closed)}",
        f"SL hit   : {sum(t.get('reason') == 'SL' for t in closed)}",
        f"BE exit  : {sum(t.get('reason') == 'BE' for t in closed)}",
        f"Momentum exit: {sum(t.get('reason') == 'MOMENTUM_FLIP' for t in closed)}",
        f"Net PnL  : ${net:+.2f}",
    ]
    if closed:
        lines += ["", "LAST 5 TRADES", "――――――――――――――――"]
        for index, trade in enumerate(reversed(closed[-5:]), 1):
            pnl = float(trade.get("pnl", 0))
            lines.append(f"{index}. {'✅' if pnl > 0 else '❌'} {str(trade.get('symbol', '')).split('/')[0]} {trade.get('direction', '')} ${pnl:+.2f} — {trade.get('reason', '')}")
    return "\n".join(lines)


def chart(candles, payload: dict, path: str) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except Exception as error:
        logger.error("Chart unavailable: %s", error)
        return False
    rows = list(candles[-100:])
    if len(rows) < 40:
        return False
    opens = [field(row, "open", 1) for row in rows]
    highs = [field(row, "high", 2) for row in rows]
    lows = [field(row, "low", 3) for row in rows]
    closes = [field(row, "close", 4) for row in rows]
    volumes = [field(row, "volume", 5) for row in rows]
    e8, e13, e20, e50 = ema(closes, 8), ema(closes, 13), ema(closes, 20), ema(closes, 50)
    fig, (axis, volume_axis) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [4, 1]})
    for index, (open_price, high, low, close, volume) in enumerate(zip(opens, highs, lows, closes, volumes)):
        candle_color = "#26a69a" if close >= open_price else "#ef5350"
        axis.vlines(index, low, high, color=candle_color, linewidth=1)
        axis.add_patch(Rectangle((index - 0.3, min(open_price, close)), 0.6, max(abs(close - open_price), max(close, 1) * 1e-6), facecolor=candle_color, edgecolor=candle_color))
        volume_axis.bar(index, volume, width=0.7, color=candle_color)
    axis.plot(e8, label="EMA8", linewidth=1.0)
    axis.plot(e13, label="EMA13", linewidth=1.0)
    axis.plot(e20, label="EMA20", linewidth=1.2)
    axis.plot(e50, label="EMA50", linewidth=1.2)
    for key, label in (("entry", "ENTRY"), ("sl", "SL"), ("tp1", "TP1"), ("tp2", "TP2")):
        value = float(payload.get(key, 0))
        if value:
            axis.axhline(value, linestyle="--", linewidth=1.1, label=f"{label} {value:,.4f}")
    entry = float(payload.get("entry", 0))
    axis.scatter(len(rows) - 1, entry, marker="^" if payload.get("direction") == "LONG" else "v", s=120, zorder=6, label="ENTRY POINT")
    axis.legend(loc="best", fontsize=7, ncol=2)
    axis.grid(alpha=0.2)
    volume_axis.grid(alpha=0.15)
    axis.set_title(f"{payload.get('symbol')} {payload.get('direction')} | Momentum v2 | 15M | {payload.get('trigger')}")
    axis.set_ylabel("Price")
    volume_axis.set_ylabel("Volume")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return os.path.exists(path) and os.path.getsize(path) > 0


async def main() -> None:
    paper = env_bool("PAPER_TRADING", True) or os.getenv("TRADING_MODE", "").lower() == "paper"
    symbols = [symbol.strip() for symbol in os.getenv("SYMBOLS", "XAU/USDT:USDT,SOL/USDT:USDT,XAG/USDT:USDT,XRP/USDT:USDT,HYPE/USDT:USDT").split(",") if symbol.strip()]
    leverage = int(os.getenv("LEVERAGE", "20"))
    margin = float(os.getenv("ADAPTIVE_MARGIN_USDT", "20"))
    interval = int(os.getenv("INTERVAL_SECONDS", "60"))
    max_positions = int(os.getenv("MAX_POSITIONS", "2"))
    risk_usdt = float(os.getenv("MOM_RISK_USDT", "5.0"))
    token = first_env("TELEGRAM_BOT_TOKEN", "TELEGRAM_TOKEN", "TG_BOT_TOKEN")
    chat_id = first_env("TELEGRAM_CHAT_ID", "TG_CHAT_ID", "CHAT_ID")
    telegram_enabled = bool(token and chat_id)
    queue: asyncio.Queue = asyncio.Queue()
    update_offset = 0
    state_dir = os.getenv("BOT_STATE_DIR", "/tmp/adaptive_momentum_v2")
    ledger_path = os.getenv("TRADE_LEDGER_FILE", os.path.join(state_dir, "trade_ledger_momentum_v2.json"))
    trades = load_json(ledger_path, [])
    latest_candles, prices = {}, {}

    def telegram_api(method: str, fields: dict, photo: str = ""):
        url = f"https://api.telegram.org/bot{token}/{method}"
        if not photo:
            request = urllib.request.Request(url, data=urllib.parse.urlencode(fields).encode(), method="POST")
        else:
            boundary = "----MomentumV2" + uuid.uuid4().hex
            parts = []
            for key, value in fields.items():
                parts += [f"--{boundary}\r\n".encode(), f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode(), str(value).encode(), b"\r\n"]
            image = open(photo, "rb").read()
            parts += [f"--{boundary}\r\n".encode(), b'Content-Disposition: form-data; name="photo"; filename="momentum_v2.png"\r\n', b"Content-Type: image/png\r\n\r\n", image, b"\r\n", f"--{boundary}--\r\n".encode()]
            request = urllib.request.Request(url, data=b"".join(parts), method="POST", headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        with urllib.request.urlopen(request, timeout=20) as response:
            result = json.loads(response.read().decode())
        if not result.get("ok"):
            raise RuntimeError(result.get("description", "Telegram failed"))
        return result

    async def telegram_worker():
        while True:
            item = await queue.get()
            try:
                if item["kind"] == "photo":
                    await asyncio.to_thread(telegram_api, "sendPhoto", {"chat_id": chat_id, "caption": item["caption"]}, item["path"])
                    if os.path.exists(item["path"]):
                        os.remove(item["path"])
                else:
                    await asyncio.to_thread(telegram_api, "sendMessage", {"chat_id": chat_id, "text": item["text"], "disable_web_page_preview": "true"})
            except asyncio.CancelledError:
                raise
            except Exception as error:
                logger.warning("Telegram failed: %s", error)
            finally:
                queue.task_done()

    connector = BinanceConnector(
        api_key="" if paper else os.getenv("EXCHANGE_API_KEY", ""),
        api_secret="" if paper else os.getenv("EXCHANGE_API_SECRET", ""),
        paper=True,
        exchange_id=os.getenv("EXCHANGE", "okx"),
        passphrase="" if paper else os.getenv("EXCHANGE_PASSPHRASE", ""),
        leverage=leverage,
    )
    live = None
    if not paper:
        from trading.connectors.okx_adapter import OKXAdapter
        live = OKXAdapter(api_key=os.getenv("EXCHANGE_API_KEY", ""), api_secret=os.getenv("EXCHANGE_API_SECRET", ""), api_passphrase=os.getenv("EXCHANGE_PASSPHRASE", ""), paper=False, leverage=leverage)

    def execute(order_type: str, payload: dict):
        if order_type.startswith("OPEN_"):
            payload["tp2"] = payload.get("tp2", payload.get("tp"))
            if telegram_enabled:
                image_path = f"/tmp/momentum_v2_{int(time.time())}_{uuid.uuid4().hex[:6]}.png"
                if not chart(latest_candles.get(str(payload.get("symbol")), []), payload, image_path):
                    raise RuntimeError("Mandatory Momentum v2 chart failed")
                queue.put_nowait({"kind": "photo", "path": image_path, "caption": trade_text(order_type, payload, paper)})
        elif telegram_enabled:
            queue.put_nowait({"kind": "text", "text": trade_text(order_type, payload, paper)})
        if paper:
            logger.info("[PAPER] %s %s", order_type, display_payload(payload))
            return {"paper": True}
        if live is None:
            raise RuntimeError("Live adapter unavailable")
        adapter_type = "CLOSE_FULL" if order_type.startswith("CLOSE_") and order_type != "CLOSE_PARTIAL" else order_type
        result = live.execute(adapter_type, payload)
        if adapter_type in {"CLOSE_FULL", "CLOSE_PARTIAL"} and result is None:
            raise RuntimeError(f"Exchange close failed for {payload.get('symbol')}")
        return result

    bots = {
        symbol: TradingBot(symbol, margin, leverage, paper, os.path.join(state_dir, symbol.replace("/", "_").replace(":", "_") + ".json"), execute, risk_usdt)
        for symbol in symbols
    }

    async def poll_commands():
        nonlocal update_offset
        if not telegram_enabled:
            return
        try:
            result = await asyncio.to_thread(telegram_api, "getUpdates", {"timeout": 0, "offset": update_offset})
            for update in result.get("result", []):
                update_offset = max(update_offset, int(update.get("update_id", 0)) + 1)
                message = update.get("message") or {}
                if str((message.get("chat") or {}).get("id", "")) != str(chat_id):
                    continue
                if str(message.get("text", "")).lower().startswith(("/stats", "/restats")):
                    queue.put_nowait({"kind": "text", "text": stats_text(trades, bots, prices, paper, margin)})
        except Exception as error:
            logger.warning("Telegram polling: %s", error)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signum, stop.set)
        except NotImplementedError:
            pass
    telegram_task = asyncio.create_task(telegram_worker()) if telegram_enabled else None
    last_bar, disabled = {}, set()
    logger.info("Adaptive Momentum v2 | build=%s | mode=%s | tf=15m | telegram=%s", BUILD_ID, "PAPER" if paper else "LIVE", "CONNECTED" if telegram_enabled else "DISABLED")
    if telegram_enabled:
        queue.put_nowait({"kind": "text", "text": f"🤖 Adaptive Momentum v2 started\nMode: {'PAPER' if paper else 'LIVE'}\nTF: 15M\nLogic: EMA20/50 → EMA8/13 → MACD Histogram Flip → ADX Rising + CHOP → Location\nCooldown: {os.getenv('MOM_COOLDOWN_BARS', '3')} bars after momentum exit\nTP1: {TP1_R:g}R close 50% + BE | TP2: {TP_R:g}R\nMargin: ${margin:.2f} | Risk: ${risk_usdt:.2f}/trade\nCommand: /stats"})

    try:
        while not stop.is_set():
            await poll_commands()
            entries_allowed = fx_open(datetime.now(timezone.utc))
            for symbol in symbols:
                if symbol in disabled:
                    continue
                try:
                    raw_15m = await connector.fetch_ohlcv(symbol, "15m", 320)
                    if len(raw_15m) < 100:
                        continue
                    closed_15m = list(raw_15m[:-1])
                    latest_candles[symbol] = closed_15m
                    bar_timestamp = timestamp(closed_15m[-1])
                    bot = bots[symbol]
                    live_price = field(raw_15m[-1], "close", 4) or float(prices.get(symbol, 0))
                    prices[symbol] = live_price
                    if bot.position_open and live_price:
                        event = bot.check_price(live_price)
                        if event:
                            trades.append({**event, "timestamp": time.time(), "version": "momentum-v2"})
                            save_json(ledger_path, trades[-2000:])
                    if bar_timestamp == last_bar.get(symbol):
                        continue
                    last_bar[symbol] = bar_timestamp
                    indicators = compute(closed_15m)
                    if not indicators:
                        continue
                    if not bot.position_open:
                        if not entries_allowed:
                            logger.info("[%s] SLEEP_MODE", symbol)
                            continue
                        if sum(int(item.position_open) for item in bots.values()) >= max_positions:
                            logger.info("[%s] WAIT max positions", symbol)
                            continue
                    event = bot.on_bar(indicators, price=live_price)
                    if event:
                        trades.append({**event, "timestamp": time.time(), "version": "momentum-v2"})
                        save_json(ledger_path, trades[-2000:])
                    logger.info("[%s] %s", symbol, display_payload(event) if event else bot.last_signal)
                except ccxt.BadSymbol as error:
                    disabled.add(symbol)
                    logger.error("[%s] unsupported: %s", symbol, error)
                except (ccxt.NetworkError, asyncio.TimeoutError) as error:
                    logger.warning("[%s] network: %s", symbol, error)
                except Exception as error:
                    logger.exception("[%s] tick failed", symbol)
                    if telegram_enabled:
                        queue.put_nowait({"kind": "text", "text": f"❌ Adaptive Momentum v2 error\nSymbol: {symbol}\n{type(error).__name__}: {error}"})
            try:
                await asyncio.wait_for(stop.wait(), timeout=interval)
            except asyncio.TimeoutError:
                pass
    finally:
        if telegram_enabled:
            queue.put_nowait({"kind": "text", "text": "⏹ Adaptive Momentum v2 stopped"})
            try:
                await asyncio.wait_for(queue.join(), timeout=5)
            except asyncio.TimeoutError:
                pass
        if telegram_task:
            telegram_task.cancel()
            try:
                await telegram_task
            except asyncio.CancelledError:
                pass
        exchange = getattr(connector, "_exchange", None)
        if exchange is not None:
            try:
                await exchange.close()
                logger.info("Closed market-data exchange session")
            except Exception as error:
                logger.warning("Exchange close failed: %s", error)


if __name__ == "__main__":
    asyncio.run(main())
