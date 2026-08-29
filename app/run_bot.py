"""Adaptive SMC MTF V7.0 runner.

Closed-candle decision pipeline:
  4H  TSS-style direction filter
  15M Market Structure
  5M  AMD (Accumulation -> Manipulation -> Distribution)
  1M  IFVG precision execution

The loop polls every INTERVAL_SECONDS, but a new entry decision is evaluated only
once per newly closed 1M candle. Higher-timeframe candle sets are cached and only
refreshed when their own time bucket advances.
"""
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
from trading.adaptive_trading_bot import BE_LOCK_R, TP1_R, TP2_R, TradingBot
from trading.indicator_engine import compute, ema

BUILD_ID = "adaptive-smc-v7.0-2026-08-29"
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger("adaptive_smc_v7")


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
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
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
    if isinstance(value, float):
        return round(value, 5)
    if isinstance(value, dict):
        return {key: display_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [display_payload(item) for item in value]
    return value


def trade_text(order_type: str, payload: dict, paper: bool) -> str:
    mode = "PAPER" if paper else "LIVE"
    direction = str(payload.get("direction", ""))
    symbol = str(payload.get("symbol", ""))
    style = str(payload.get("style", "LEGACY"))

    if order_type.startswith("OPEN_"):
        return "\n".join([
            f"{'🟢' if direction == 'LONG' else '🔴'} ADAPTIVE SMC V7 {direction} — {symbol}",
            f"Mode: {mode} | Execution: M1 CLOSED",
            f"4H TSS: {payload.get('tss_bias','-')} ({float(payload.get('tss_score',0)):.0f})",
            f"M15 Structure: {payload.get('structure','-')}",
            f"M5 AMD: {payload.get('amd_phase','-')}",
            f"M1 IFVG: {float(payload.get('ifvg_low',0)):,.4f} – {float(payload.get('ifvg_high',0)):,.4f}",
            f"Trigger: {payload.get('trigger', '-')}",
            "",
            f"Entry: {float(payload.get('entry',0)):,.4f}",
            f"SL: {float(payload.get('sl',0)):,.4f} ({float(payload.get('sl_pct',0)):.2f}%)",
            f"TP1: {float(payload.get('tp1',0)):,.4f} ({TP1_R:g}R · close 50% · SL→BE+{BE_LOCK_R:g}R)",
            f"TP2: {float(payload.get('tp2',0)):,.4f} ({TP2_R:g}R · close remaining)",
            "Runner after TP1: M15/M5 invalidation can exit early",
            f"Size: {float(payload.get('size',0)):,.6g} | Risk: ${float(payload.get('risk_usdt',0)):.2f}",
        ])

    pnl = float(payload.get("pnl", 0.0))
    return (
        f"{'✅' if pnl >= 0 else '❌'} [{mode}] {style} {payload.get('reason','CLOSE')} — {symbol} {direction}\n"
        f"Price: {float(payload.get('price',0)):,.4f}\n"
        f"PnL: ${pnl:+.2f} ({float(payload.get('r_multiple',0)):+.2f}R)"
    )


def stats_text(trades, bots, prices, paper: bool, margin: float, stats_started_at: float) -> str:
    closed = [t for t in trades if t.get("event") == "CLOSE"]
    partials = [t for t in trades if t.get("event") == "PARTIAL"]
    wins = [t for t in closed if float(t.get("pnl", 0)) > 0]
    net = sum(float(t.get("pnl", 0)) for t in closed + partials)
    started = datetime.fromtimestamp(stats_started_at, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        "📊 Adaptive SMC MTF V7 Stats", "",
        f"Mode: {'PAPER' if paper else 'LIVE'}",
        "Pipeline: 4H TSS → M15 Structure → M5 AMD → M1 IFVG",
        "Signals: CLOSED candles only · M1 execution cadence",
        "4H: EMA20/50 + HMA16 trend-tunnel approximation",
        "M15: HH/HL · LH/LL · BOS/CHOCH",
        "M5: Accumulation → liquidity sweep → Distribution",
        "M1: inverted FVG + fresh retest/rejection",
        f"Risk: TP1 {TP1_R:g}R 50% → BE+{BE_LOCK_R:g}R · TP2 {TP2_R:g}R closes remaining",
        "Early runner exit: M15 CHOCH or opposite M5 distribution after TP1",
        f"Stats since: {started}", "",
        f"OPEN POSITIONS ({sum(int(bot.position_open) for bot in bots.values())})",
        "――――――――――――――――",
    ]

    count = 0
    floating = 0.0
    for symbol, bot in bots.items():
        p = bot.position
        if not p:
            continue
        count += 1
        current = float(prices.get(symbol, p.entry))
        pnl = (current - p.entry) * p.size if p.direction == "LONG" else (p.entry - current) * p.size
        floating += pnl
        lines += [
            f"{'🟢' if p.direction == 'LONG' else '🔴'} {symbol.split('/')[0]} {p.style} {p.direction}",
            f"Entry : {p.entry:,.4f}",
            f"Now   : {current:,.4f}",
            f"PnL   : ${pnl:+.2f}",
            f"SL    : {p.sl:,.4f}{' (LOCK)' if p.be_moved else ''}",
            f"TP1   : {p.tp1:,.4f} {'✅' if p.tp1_hit else ''}",
            f"TP2   : {p.tp2:,.4f} {'✅' if p.tp2_hit else ''}",
            f"4H/M15/M5: {p.tss_bias or '-'} / {p.structure or '-'} / {p.amd_phase or '-'}",
            f"IFVG  : {p.ifvg_low:,.4f} – {p.ifvg_high:,.4f}",
            f"Held  : {duration(time.time() - p.opened_at)}", "",
        ]

    if not count:
        lines += ["No open positions", ""]

    wr = 100 * len(wins) / len(closed) if closed else 0.0
    lines += [
        "EXPOSURE", "――――――――――――――――",
        f"Margin Target : ${count * margin:.2f}",
        f"Floating PnL  : ${floating:+.2f}", "",
        "OVERALL", "――――――――――――――――",
        f"Trades   : {len(closed)} ({len(wins)}W / {len(closed)-len(wins)}L)",
        f"Win rate : {wr:.0f}%",
        f"TP1 hit  : {sum(t.get('reason') == 'TP1' for t in partials)}",
        f"TP2 hit  : {sum(t.get('reason') == 'TP2' for t in closed)}",
        f"SL hit   : {sum(t.get('reason') == 'SL' for t in closed)}",
        f"Locked SL: {sum(t.get('reason') == 'LOCKED_SL' for t in closed)}",
        f"SMC runner exits: {sum(t.get('reason') == 'RUNNER_STRUCTURE_EXIT' for t in closed)}",
        f"Exchange reconciles: {sum(str(t.get('reason','')).startswith('EXCHANGE_') for t in closed)}",
        f"Net PnL  : ${net:+.2f}", "",
        "BY STRATEGY", "――――――――――――――――",
    ]

    current_closed = [t for t in closed if t.get("style") == "SMC_MTF_V1"]
    current_partials = [t for t in partials if t.get("style") == "SMC_MTF_V1"]
    current_wins = sum(float(t.get("pnl", 0)) > 0 for t in current_closed)
    current_wr = 100 * current_wins / len(current_closed) if current_closed else 0.0
    current_pnl = sum(float(t.get("pnl", 0)) for t in current_closed + current_partials)
    lines.append(f"SMC_MTF_V1 {len(current_closed):>2} closes · {current_wr:.0f}%WR · ${current_pnl:+.2f}")

    old = [t for t in closed + partials if t.get("style") != "SMC_MTF_V1"]
    if old:
        old_closed = [t for t in closed if t.get("style") != "SMC_MTF_V1"]
        old_pnl = sum(float(t.get("pnl", 0)) for t in old)
        lines += ["", f"PRESERVED/RECOVERED: {len(old_closed)} closes · ${old_pnl:+.2f}"]

    if closed:
        lines += ["", "LAST 5 CLOSES", "――――――――――――――――"]
        for idx, trade in enumerate(reversed(closed[-5:]), 1):
            pnl = float(trade.get("pnl", 0))
            lines.append(
                f"{idx}. {'✅' if pnl > 0 else '❌'} {str(trade.get('symbol','')).split('/')[0]} "
                f"{trade.get('style','')} {trade.get('direction','')} ${pnl:+.2f} — {trade.get('reason','')}"
            )
    return "\n".join(lines)


def chart(candles, payload: dict, path: str) -> bool:
    """M1 execution chart for Telegram. EMA20 is visual context only."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except Exception as error:
        logger.error("Chart unavailable: %s", error)
        return False

    rows = list(candles[-120:])
    if len(rows) < 50:
        return False
    opens = [field(r, "open", 1) for r in rows]
    highs = [field(r, "high", 2) for r in rows]
    lows = [field(r, "low", 3) for r in rows]
    closes = [field(r, "close", 4) for r in rows]
    volumes = [field(r, "volume", 5) for r in rows]
    e20 = ema(closes, 20)

    fig, (axis, volume_axis) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [4, 1]}
    )
    for idx, (o, h, l, c, v) in enumerate(zip(opens, highs, lows, closes, volumes)):
        color = "#26a69a" if c >= o else "#ef5350"
        axis.vlines(idx, l, h, color=color, linewidth=1)
        axis.add_patch(Rectangle(
            (idx - 0.3, min(o, c)), 0.6,
            max(abs(c - o), max(c, 1) * 1e-6),
            facecolor=color, edgecolor=color,
        ))
        volume_axis.bar(idx, v, width=0.7, color=color)

    axis.plot(e20, label="M1 EMA20 (visual)", linewidth=1.2)
    levels = (
        ("entry", "ENTRY"), ("sl", "SL"), ("tp1", "TP1"), ("tp2", "TP2"),
        ("ifvg_low", "IFVG LOW"), ("ifvg_high", "IFVG HIGH"),
        ("manipulation_low", "M5 MANIP LOW"), ("manipulation_high", "M5 MANIP HIGH"),
    )
    for key, label in levels:
        value = float(payload.get(key, 0) or 0)
        if value:
            axis.axhline(value, linestyle="--", linewidth=1.0, label=f"{label} {value:,.4f}")
    entry = float(payload.get("entry", 0))
    axis.scatter(
        len(rows) - 1, entry,
        marker="^" if payload.get("direction") == "LONG" else "v",
        s=120, zorder=6, label="ENTRY",
    )
    axis.legend(loc="best", fontsize=7)
    axis.grid(alpha=0.2)
    volume_axis.grid(alpha=0.15)
    axis.set_title(
        f"{payload.get('symbol')} {payload.get('direction')} | Adaptive SMC V7 | "
        f"4H {payload.get('tss_bias')} · M15 {payload.get('structure')} · M5 {payload.get('amd_phase')}"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return os.path.exists(path) and os.path.getsize(path) > 0


async def main() -> None:
    paper = env_bool("PAPER_TRADING", True) or os.getenv("TRADING_MODE", "").lower() == "paper"
    symbols = [s.strip() for s in os.getenv(
        "SYMBOLS", "XAU/USDT:USDT,SOL/USDT:USDT,XAG/USDT:USDT,XRP/USDT:USDT,HYPE/USDT:USDT,BTC/USDT:USDT"
    ).split(",") if s.strip()]
    leverage = int(os.getenv("LEVERAGE", "20"))
    margin = float(os.getenv("ADAPTIVE_MARGIN_USDT", "20"))
    interval = max(10, int(os.getenv("INTERVAL_SECONDS", "30")))
    max_positions = int(os.getenv("MAX_POSITIONS", "2"))
    risk_usdt = float(os.getenv("MOM_RISK_USDT", "5.0"))
    token = first_env("TELEGRAM_BOT_TOKEN", "TELEGRAM_TOKEN", "TG_BOT_TOKEN")
    chat_id = first_env("TELEGRAM_CHAT_ID", "TG_CHAT_ID", "CHAT_ID")
    telegram_enabled = bool(token and chat_id)
    queue: asyncio.Queue = asyncio.Queue()
    update_offset = 0

    # Keep the same default state directory as V6.1 so a persistent mounted
    # volume can preserve any position state through the strategy upgrade.
    state_dir = os.getenv("BOT_STATE_DIR", "/tmp/adaptive_momentum_v4_1")
    ledger_path = os.getenv("TRADE_LEDGER_FILE", os.path.join(state_dir, "trade_ledger_smc_v7.json"))
    stats_meta_path = os.path.join(state_dir, "stats_meta_smc_v7.json")
    trades = load_json(ledger_path, [])
    stats_meta = load_json(stats_meta_path, {"started_at": time.time()})
    stats_started_at = float(stats_meta.get("started_at", time.time()))
    latest_candles, prices = {}, {}

    def telegram_api(method: str, fields: dict, photo: str = ""):
        url = f"https://api.telegram.org/bot{token}/{method}"
        if not photo:
            request = urllib.request.Request(
                url, data=urllib.parse.urlencode(fields).encode(), method="POST"
            )
        else:
            boundary = "----AdaptiveSMCV7" + uuid.uuid4().hex
            parts = []
            for key, value in fields.items():
                parts += [
                    f"--{boundary}\r\n".encode(),
                    f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode(),
                    str(value).encode(), b"\r\n",
                ]
            with open(photo, "rb") as handle:
                image = handle.read()
            parts += [
                f"--{boundary}\r\n".encode(),
                b'Content-Disposition: form-data; name="photo"; filename="adaptive_smc_v7.png"\r\n',
                b"Content-Type: image/png\r\n\r\n", image, b"\r\n",
                f"--{boundary}--\r\n".encode(),
            ]
            request = urllib.request.Request(
                url, data=b"".join(parts), method="POST",
                headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            )
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
                    await asyncio.to_thread(
                        telegram_api, "sendPhoto",
                        {"chat_id": chat_id, "caption": item["caption"]}, item["path"],
                    )
                    if os.path.exists(item["path"]):
                        os.remove(item["path"])
                else:
                    await asyncio.to_thread(
                        telegram_api, "sendMessage",
                        {"chat_id": chat_id, "text": item["text"], "disable_web_page_preview": "true"},
                    )
            except asyncio.CancelledError:
                raise
            except Exception as error:
                logger.warning("Telegram failed: %s", error)
            finally:
                queue.task_done()

    # Public market-data connector. paper=True keeps this path read-only even in
    # live mode; authenticated orders go exclusively through OKXAdapter below.
    connector = BinanceConnector(
        api_key="", api_secret="", paper=True,
        exchange_id=os.getenv("EXCHANGE", "okx"), passphrase="", leverage=leverage,
    )

    live = None
    if not paper:
        from trading.connectors.okx_adapter import OKXAdapter
        live = OKXAdapter(
            api_key=os.getenv("EXCHANGE_API_KEY", ""),
            api_secret=os.getenv("EXCHANGE_API_SECRET", ""),
            api_passphrase=os.getenv("EXCHANGE_PASSPHRASE", ""),
            paper=False, leverage=leverage,
        )

    def execute(order_type: str, payload: dict):
        # SL amendments are operational updates, not user-facing trade events.
        if order_type == "AMEND_SL":
            if paper:
                logger.info("[PAPER] AMEND_SL %s", display_payload(payload))
                return {"paper": True, "_amended": True}
            if live is None:
                raise RuntimeError("Live adapter unavailable")
            return live.execute("AMEND_SL", payload)

        if paper:
            result = {"paper": True}
            logger.info("[PAPER] %s %s", order_type, display_payload(payload))
        else:
            if live is None:
                raise RuntimeError("Live adapter unavailable")
            adapter_type = (
                "CLOSE_FULL"
                if order_type.startswith("CLOSE_") and order_type != "CLOSE_PARTIAL"
                else order_type
            )
            result = live.execute(adapter_type, payload)
            if adapter_type in {"CLOSE_FULL", "CLOSE_PARTIAL"} and result is None:
                raise RuntimeError(f"Exchange close failed for {payload.get('symbol')}")

        # Notify only after paper acceptance / live exchange execution succeeds.
        if telegram_enabled:
            if order_type.startswith("OPEN_"):
                image_path = f"/tmp/adaptive_smc_v7_{int(time.time())}_{uuid.uuid4().hex[:6]}.png"
                if chart(latest_candles.get(str(payload.get("symbol")), []), payload, image_path):
                    queue.put_nowait({
                        "kind": "photo", "path": image_path,
                        "caption": trade_text(order_type, payload, paper),
                    })
                else:
                    queue.put_nowait({"kind": "text", "text": trade_text(order_type, payload, paper)})
            else:
                queue.put_nowait({"kind": "text", "text": trade_text(order_type, payload, paper)})
        return result

    bots = {
        symbol: TradingBot(
            symbol, margin, leverage, paper,
            os.path.join(state_dir, symbol.replace("/", "_").replace(":", "_") + ".json"),
            execute, risk_usdt,
        )
        for symbol in symbols
    }

    def record_event(event: dict | None) -> None:
        if not event:
            return
        trades.append({**event, "timestamp": time.time(), "version": "adaptive-smc-v7"})
        save_json(ledger_path, trades[-2000:])

    async def poll_commands():
        nonlocal update_offset, stats_started_at
        if not telegram_enabled:
            return
        try:
            result = await asyncio.to_thread(
                telegram_api, "getUpdates", {"timeout": 0, "offset": update_offset}
            )
            for update in result.get("result", []):
                update_offset = max(update_offset, int(update.get("update_id", 0)) + 1)
                message = update.get("message") or {}
                if str((message.get("chat") or {}).get("id", "")) != str(chat_id):
                    continue
                command = str(message.get("text", "")).strip().lower().split()[0] if message.get("text") else ""
                if command == "/stats":
                    queue.put_nowait({
                        "kind": "text",
                        "text": stats_text(trades, bots, prices, paper, margin, stats_started_at),
                    })
                elif command == "/restats":
                    trades.clear()
                    save_json(ledger_path, [])
                    stats_started_at = time.time()
                    save_json(stats_meta_path, {"started_at": stats_started_at})
                    queue.put_nowait({
                        "kind": "text",
                        "text": "♻️ Adaptive SMC V7 stats reset\nTrades: 0\nWin rate: 0%\nNet PnL: $0.00\nOpen positions were NOT closed.",
                    })
        except Exception as error:
            logger.warning("Telegram polling: %s", error)

    blocked_external: set[str] = set()

    async def reconcile_live_position(symbol: str, bot: TradingBot, live_price: float = 0.0) -> bool:
        """Reconcile local state against OKX; returns True if this cycle should stop."""
        if paper or live is None:
            return False
        try:
            pos = await asyncio.to_thread(live.fetch_open_position, symbol)
        except Exception as error:
            logger.warning("[%s] live reconcile failed: %s", symbol, error)
            return False

        if bot.position_open:
            if pos is None:
                event = bot.reconcile_exchange_closed(live_price, "EXCHANGE_CLOSED")
                record_event(event)
                blocked_external.discard(symbol)
                logger.warning("[%s] local position reconciled flat from OKX", symbol)
                return True
            # Recover SL algo id after restart when state had the position but
            # not the exchange order id.
            if not bot.position.sl_algo_id:
                try:
                    side = bot.position.direction.lower()
                    attached = await asyncio.to_thread(live.fetch_attached_sl_tp, symbol, side)
                    if attached and attached.get("algo_id"):
                        bot.position.sl_algo_id = str(attached["algo_id"])
                        bot.save_state()
                except Exception:
                    pass
            return False

        if pos is None:
            blocked_external.discard(symbol)
            return False

        # Exchange has a position but local state is absent (common after a
        # Railway restart with ephemeral /tmp). Adopt only when a real exchange
        # stop can be recovered; never fabricate risk on a live position.
        side = str(pos.get("side") or "").lower()
        if side not in {"long", "short"}:
            blocked_external.add(symbol)
            return True
        try:
            attached = await asyncio.to_thread(live.fetch_attached_sl_tp, symbol, side)
        except Exception:
            attached = None
        if not attached or not attached.get("sl"):
            if symbol not in blocked_external:
                logger.error(
                    "[%s] live %s exists but attached SL cannot be recovered — blocking new entries for this symbol",
                    symbol, side,
                )
            blocked_external.add(symbol)
            return True

        contracts = abs(float(pos.get("contracts") or 0.0))
        contract_size = abs(float(pos.get("contractSize") or 1.0))
        size = contracts * contract_size
        entry = float(pos.get("entryPrice") or pos.get("entry_price") or 0.0)
        if entry <= 0 or size <= 0:
            blocked_external.add(symbol)
            return True
        bot.adopt_exchange_position(
            side.upper(), entry, size, float(attached["sl"]),
            float(attached.get("tp") or 0.0), str(attached.get("algo_id") or ""),
        )
        blocked_external.discard(symbol)
        logger.warning("[%s] adopted live OKX %s after restart: entry=%.4f size=%.6g", symbol, side, entry, size)
        return False

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signum, stop.set)
        except NotImplementedError:
            pass

    telegram_task = asyncio.create_task(telegram_worker()) if telegram_enabled else None
    logger.info(
        "Adaptive SMC V7 | build=%s | mode=%s | pipeline=4H>T15>T5>T1 | poll=%ss | telegram=%s",
        BUILD_ID, "PAPER" if paper else "LIVE", interval,
        "CONNECTED" if telegram_enabled else "DISABLED",
    )

    if telegram_enabled:
        queue.put_nowait({"kind": "text", "text": (
            f"🤖 Adaptive SMC MTF V7 started\nMode: {'PAPER' if paper else 'LIVE'}\n"
            "Pipeline: 4H TSS-style → M15 Market Structure → M5 AMD → M1 IFVG\n"
            "Signals use CLOSED candles only; execution is evaluated once per closed M1 candle.\n"
            "No 1H hard gate. 4H chooses direction; M15 validates structure; M5 finds liquidity manipulation; M1 times entry.\n"
            f"TP1 {TP1_R:g}R close 50% → BE+{BE_LOCK_R:g}R | TP2 {TP2_R:g}R closes remaining\n"
            "After TP1, M15 CHOCH / opposite M5 distribution can exit the remainder early.\n"
            f"Margin cap: ${margin:.2f} | Risk target: ${risk_usdt:.2f}/trade | Max positions: {max_positions}\n"
            "Commands: /stats · /restats"
        )})

    # First live reconciliation before any scanner is allowed to open a trade.
    if not paper:
        for symbol, bot in bots.items():
            await reconcile_live_position(symbol, bot, 0.0)

    last_1m_bar: dict[str, object] = {}
    disabled: set[str] = set()
    tf_cache: dict[tuple[str, str], list] = {}
    tf_bucket: dict[tuple[str, str], int] = {}

    async def get_closed_cached(symbol: str, tf: str, limit: int, bucket_seconds: int):
        key = (symbol, tf)
        bucket = int(time.time() // bucket_seconds)
        cached = tf_cache.get(key)
        if cached is not None and tf_bucket.get(key) == bucket:
            return cached
        raw = await connector.fetch_ohlcv(symbol, tf, limit)
        if len(raw) < 3:
            return cached or []
        closed = list(raw[:-1])
        if not closed:
            return cached or []
        old_last = timestamp(cached[-1]) if cached else None
        new_last = timestamp(closed[-1])
        tf_cache[key] = closed
        # If the exchange has not published the just-closed candle yet, leave
        # the old bucket marker so the next poll retries instead of staying stale.
        if cached is None or new_last != old_last:
            tf_bucket[key] = bucket
        return closed

    try:
        while not stop.is_set():
            await poll_commands()
            now_utc = datetime.now(timezone.utc)

            for symbol in symbols:
                if symbol in disabled:
                    continue
                try:
                    raw_1m = await connector.fetch_ohlcv(symbol, "1m", 220)
                    if len(raw_1m) < 80:
                        logger.debug("[%s] M1 warmup %d/80", symbol, len(raw_1m))
                        continue
                    closed_1m = list(raw_1m[:-1])
                    latest_candles[symbol] = closed_1m
                    bar_ts = timestamp(closed_1m[-1])
                    live_price = field(raw_1m[-1], "close", 4) or field(closed_1m[-1], "close", 4)
                    prices[symbol] = live_price
                    bot = bots[symbol]

                    # Exchange reconciliation precedes local price management so
                    # an OKX-side SL/TP cannot leave a ghost local position.
                    reconciled = await reconcile_live_position(symbol, bot, live_price)
                    if reconciled and not bot.position_open:
                        continue

                    if bot.position_open and live_price:
                        event = bot.check_price(live_price)
                        if event:
                            record_event(event)

                    # No duplicate decision on the same closed M1 candle.
                    if bar_ts == last_1m_bar.get(symbol):
                        continue
                    last_1m_bar[symbol] = bar_ts

                    closed_5m = await get_closed_cached(symbol, "5m", 140, 5 * 60)
                    closed_15m = await get_closed_cached(symbol, "15m", 140, 15 * 60)
                    closed_4h = await get_closed_cached(symbol, "4h", 90, 4 * 60 * 60)
                    indicators = compute(closed_1m, closed_5m, closed_15m, closed_4h)
                    if not indicators:
                        logger.info(
                            "[%s] WAIT MTF warmup | M1=%d M5=%d M15=%d H4=%d",
                            symbol, len(closed_1m), len(closed_5m), len(closed_15m), len(closed_4h),
                        )
                        continue

                    if not bot.position_open:
                        base = symbol.split("/")[0].upper()
                        if base in {"XAU", "XAG"} and not fx_open(now_utc):
                            logger.info("[%s] SLEEP_MODE FX closed", symbol)
                            continue
                        if symbol in blocked_external:
                            logger.warning("[%s] WAIT external live position not safely adopted", symbol)
                            continue
                        if sum(int(item.position_open) for item in bots.values()) >= max_positions:
                            logger.info("[%s] WAIT max positions", symbol)
                            continue

                    event = bot.on_bar(indicators, price=live_price)
                    if event:
                        record_event(event)
                    logger.info("[%s] %s", symbol, display_payload(event) if event else bot.last_signal)

                except ccxt.BadSymbol as error:
                    disabled.add(symbol)
                    logger.error("[%s] unsupported: %s", symbol, error)
                except (ccxt.NetworkError, asyncio.TimeoutError) as error:
                    logger.warning("[%s] network: %s", symbol, error)
                except Exception as error:
                    logger.exception("[%s] tick failed", symbol)
                    if telegram_enabled:
                        queue.put_nowait({
                            "kind": "text",
                            "text": f"❌ Adaptive SMC V7 error\nSymbol: {symbol}\n{type(error).__name__}: {error}",
                        })

            try:
                await asyncio.wait_for(stop.wait(), timeout=interval)
            except asyncio.TimeoutError:
                pass
    finally:
        if telegram_enabled:
            queue.put_nowait({"kind": "text", "text": "⏹ Adaptive SMC MTF V7 stopped"})
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
        if live is not None:
            try:
                await live.close()
            except Exception:
                pass


if __name__ == "__main__":
    asyncio.run(main())
