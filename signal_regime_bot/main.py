"""
Signal Regime Bias Strategy — live bot entry point.

Loop: every `poll_interval_sec`, for each symbol —
  1. fetch 30m/1h/4h closed bars (skip symbol if any TF has < min_bars)
  2. if a position is open: check SL/TP1/TP2 against the live ticker every
     tick; run the 3-bar-confirm health monitor once per newly-closed 30m bar
  3. else: once per newly-closed 30m bar, evaluate SignalEngine and open a
     position if regime/bias/entry all clear their thresholds and risk
     manager allows a new entry (no cooldown, no daily limit breach)

Every branch that mutates state is wrapped so one symbol's exception can
never kill the loop or leave an order half-placed with silent failure.
"""
from __future__ import annotations

import asyncio
import logging
import signal
import time

import pandas as pd

from config import load_config
from exchange_client import ExchangeClient
from data_engine import DataEngine
from entry_engine import SignalEngine, LONG, SHORT
from risk_manager import RiskManager
from position_manager import PositionManager
from telegram_notifier import TelegramNotifier
from chart_engine import build_entry_chart

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("main")


class Bot:
    def __init__(self):
        self.cfg = load_config()
        self.client = ExchangeClient(
            api_key=self.cfg.okx_api_key, api_secret=self.cfg.okx_secret,
            passphrase=self.cfg.okx_passphrase, paper=self.cfg.paper,
            leverage=self.cfg.leverage, margin_mode=self.cfg.margin_mode,
        )
        self.data = DataEngine(self.cfg, self.client)
        self.signal_engine = SignalEngine(self.cfg)
        self.risk = RiskManager(self.cfg)
        self.positions = PositionManager(self.cfg, self.client, self.risk)
        self.telegram = TelegramNotifier(self.cfg.telegram_bot_token, self.cfg.telegram_chat_id)

        self._last_entry_bar: dict[str, pd.Timestamp] = {}
        self._daily_alert_sent = False
        self._cooldown_alert_sent = False
        self._running = False

    async def start(self):
        problems = self.cfg.validate_live()
        if problems:
            raise RuntimeError("Cannot start live: " + "; ".join(problems))

        if not self.cfg.paper:
            hedge_ok = await self.client.ensure_hedge_mode()
            if not hedge_ok:
                raise RuntimeError("Could not confirm OKX hedge mode — refusing to trade live.")

        balance = await self.client.fetch_balance_usdt()
        logger.info("=== Signal Regime Bias Bot starting [%s] symbols=%s lev=%dx balance=%.2f ===",
                   "PAPER" if self.cfg.paper else "LIVE", self.cfg.symbols, self.cfg.leverage, balance)
        if not self.telegram.enabled:
            logger.warning("Telegram not configured — TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID missing")

        self._running = True

    async def stop(self):
        self._running = False
        await self.client.close()
        logger.info("Bot stopped cleanly.")

    async def run_forever(self):
        while self._running:
            for symbol in self.cfg.symbols:
                try:
                    await self._process_symbol(symbol)
                except Exception as e:
                    logger.error("[%s] unhandled error: %s", symbol, e, exc_info=True)
                    await self.telegram.error(symbol, str(e))
            await self._check_global_alerts()
            await asyncio.sleep(self.cfg.poll_interval_sec)

    # ── Per-symbol processing ────────────────────────────────────────────────

    async def _process_symbol(self, symbol: str):
        self.data.new_tick()
        frames = await self.data.fetch_all(symbol)
        if not self.data.has_min_bars(frames):
            logger.debug("[%s] insufficient bars (<%d) on one or more TFs — skip",
                        symbol, self.cfg.min_bars)
            return

        df_30m = frames[self.cfg.tf_entry]
        df_1h = frames[self.cfg.tf_bias]
        df_4h = frames[self.cfg.tf_regime]

        if self.positions.has_position(symbol):
            await self._manage_open_position(symbol, df_30m, df_1h, df_4h)
        else:
            await self._look_for_entry(symbol, df_30m, df_1h, df_4h)

    async def _manage_open_position(self, symbol: str, df_30m, df_1h, df_4h):
        pos = self.positions.get(symbol)
        try:
            ticker = await self.client.fetch_ticker(symbol)
            price = float(ticker["last"])
        except Exception as e:
            logger.warning("[%s] ticker fetch failed: %s", symbol, e)
            return

        event = await self.positions.check_exits_live(symbol, price)
        if event:
            await self._handle_event(event)
            return   # position fully or partially closed this tick — health check waits for next

        # Health monitor — once per newly-closed 30m bar only.
        bar_ts = df_30m.index[-1] if len(df_30m) else None
        if bar_ts is not None and pos.last_health_bar_ts != bar_ts:
            regime = self.signal_engine.regime_engine.analyze(df_4h)
            bias = self.signal_engine.bias_engine.analyze(df_1h)
            hevent = await self.positions.process_closed_bar_health(symbol, df_30m, regime, bias)
            if hevent:
                await self._handle_event(hevent)

    async def _look_for_entry(self, symbol: str, df_30m, df_1h, df_4h):
        bar_ts = df_30m.index[-1] if len(df_30m) else None
        if bar_ts is None or self._last_entry_bar.get(symbol) == bar_ts:
            return   # already evaluated this closed bar
        self._last_entry_bar[symbol] = bar_ts

        now = time.time()
        balance = await self.client.fetch_balance_usdt()
        can_open, reason = self.risk.can_open_new(balance, now, self.positions.open_position_count())
        if not can_open:
            logger.debug("[%s] entry blocked: %s", symbol, reason)
            return

        sig = self.signal_engine.evaluate(df_30m, df_1h, df_4h)
        if sig.direction not in (LONG, SHORT):
            logger.debug("[%s] no signal: %s", symbol, sig.reason)
            return

        risk_pct = self.cfg.risk_per_trade * sig.regime.size_multiplier
        chart_path = build_entry_chart(
            symbol, df_30m, sig.direction, sig.price,
            *self._preview_sl_tp(sig, df_30m),
        )

        pos = await self.positions.open_position(
            symbol, sig.direction, sig.price, df_30m, sig.regime, sig.bias, sig.entry_score)
        if pos is None:
            return

        await self.telegram.entry_signal(
            symbol, sig.direction, pos.entry_price, pos.stop_loss, pos.tp1, pos.tp2,
            sig.regime, sig.bias, sig.entry_score, risk_pct, self.cfg.leverage, chart_path)
        await self.telegram.order_opened(
            symbol, sig.direction, pos.entry_price, pos.amount, pos.stop_loss, pos.tp1, pos.tp2)

    def _preview_sl_tp(self, sig, df_30m) -> tuple[float, float, float]:
        """Compute SL/TP1/TP2 for the chart BEFORE the order is placed (same math as open_position)."""
        import indicators as ind
        from position_manager import calc_stop_loss, calc_take_profits
        c = self.cfg
        atr_val = float(ind.atr(df_30m, c.sl_atr_period).iloc[-1])
        swing_high, swing_low = ind.recent_swing_levels(
            df_30m["high"], df_30m["low"], c.swing_lookback_left, c.swing_lookback_right)
        side = "long" if sig.direction == LONG else "short"
        sl = calc_stop_loss(side, sig.price, atr_val, c.sl_atr_mult, swing_high, swing_low,
                           c.sl_min_pct, c.sl_max_pct)
        tp1, tp2 = calc_take_profits(side, sig.price, sl, c.tp1_r, c.tp2_r)
        return sl, tp1, tp2

    async def _handle_event(self, event: dict):
        ev = event.get("event")
        symbol = event.get("symbol", "")
        if ev == "TP1_HIT":
            await self.telegram.tp1_hit(symbol, event["price"], event["pnl"], event["new_sl"])
        elif ev == "TP2_HIT":
            await self.telegram.tp2_hit(symbol, event["price"], event["pnl"])
        elif ev == "SL_HIT":
            await self.telegram.sl_hit(symbol, event["price"], event["pnl"], at_breakeven=False)
        elif ev == "BE_HIT":
            await self.telegram.sl_hit(symbol, event["price"], event["pnl"], at_breakeven=True)
        elif ev == "HEALTH_CLOSE":
            await self.telegram.health_close(symbol, event["price"], event["pnl"],
                                             event.get("health_score", 0.0), event.get("weak_count", 0))
        elif ev == "ERROR":
            await self.telegram.error(symbol, event.get("detail", "unknown error"))
        # HEALTH_OK / HEALTH_WEAK (not yet confirmed) are logged only, not alerted —
        # avoids spamming Telegram every 30m bar while a position rides normally.

    async def _check_global_alerts(self):
        now = time.time()
        balance = await self.client.fetch_balance_usdt()
        allowed, reason = self.risk.check_daily_limits(balance, now)
        if not allowed and not self._daily_alert_sent:
            self._daily_alert_sent = True
            pnl_pct = self.risk.state.day_realized_pnl / max(self.risk.state.day_start_balance, 1e-9)
            if "loss" in reason.lower():
                await self.telegram.daily_loss_limit(pnl_pct)
            else:
                await self.telegram.daily_profit_lock(pnl_pct)
        elif allowed:
            self._daily_alert_sent = False

        in_cd = self.risk.is_in_cooldown(now)
        if in_cd and not self._cooldown_alert_sent:
            self._cooldown_alert_sent = True
            await self.telegram.cooldown_activated(
                int(self.risk.cooldown_remaining_sec(now) / 60), self.risk.state.loss_streak)
        elif not in_cd:
            self._cooldown_alert_sent = False


async def _main():
    bot = Bot()
    await bot.start()

    stop_event = asyncio.Event()
    loop = asyncio.get_event_loop()
    for sig_name in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig_name, stop_event.set)
        except (NotImplementedError, RuntimeError):
            pass

    run_task = asyncio.create_task(bot.run_forever())
    await stop_event.wait()
    logger.info("Shutdown signal received...")
    run_task.cancel()
    try:
        await run_task
    except asyncio.CancelledError:
        pass
    await bot.stop()


if __name__ == "__main__":
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        pass
