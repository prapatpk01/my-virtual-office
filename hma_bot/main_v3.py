"""HMA Expert MTF V3 runtime.

Reuses the production HMA bot's reconciliation, management, Telegram and stats,
but upgrades entry execution to 4H -> 1H -> 15M location -> 5M trigger.
"""
from __future__ import annotations

import asyncio
import time

import pandas as pd

import main as base
import strategy_v3 as S


class Bot(base.Bot):
    def __init__(self):
        super().__init__()
        self.strat = S.PrecisionTrendStructureV3(self.cfg.strategy_config())

    async def start(self):
        problems = self.cfg.validate_live()
        if problems:
            raise RuntimeError("Cannot start: " + "; ".join(problems))
        if not self.cfg.paper:
            if not await self.client.ensure_hedge_mode():
                raise RuntimeError("Could not confirm OKX hedge mode.")
        balance = await self.client.fetch_balance_usdt()
        base.logger.info(
            "=== HMA EXPERT MTF V3 [%s] symbols=%s margin=$%.2f leverage=x%d max_pos=%d balance=%.2f ===",
            "PAPER" if self.cfg.paper else "LIVE", self.cfg.symbols,
            self.cfg.margin_per_position_usd, self.cfg.leverage,
            self.cfg.max_positions, balance,
        )
        await self._reconcile_startup()
        self._running = True
        if self.tg.enabled:
            asyncio.create_task(self._command_loop())
            await self.tg.send_text(
                f"🤖 *HMA Expert MTF V3 started* [{'PAPER' if self.cfg.paper else 'LIVE'}]\n"
                f"Symbols: `{', '.join(self.cfg.symbols)}`\n"
                f"Balance: `{balance:.2f}` USDT | Margin `${self.cfg.margin_per_position_usd:.2f}`/position "
                f"| Leverage `x{self.cfg.leverage}` | Max `{self.cfg.max_positions}` positions\n"
                f"4H Direction → 1H Q+DMI → 15M Location/Structure → 5M Execution\n"
                f"SL: 15M structure + ATR buffer | T1 +0.6%→lock +0.3% | T2 +1.0%→lock +0.7% | TP +1.5%"
            )

    async def _entry_frames(self, symbol: str):
        df5, df15, df1h, df4h = await asyncio.gather(
            self._frame(symbol, "5m", 5, 360),
            self._frame(symbol, "15m", 15, 320),
            self._frame(symbol, "1h", 60, 240),
            self._frame(symbol, "4h", 240, 220),
        )
        return df5, df15, df1h, df4h

    def _set_view_v3(self, symbol: str, df5, df15, df1h, df4h):
        try:
            t = self.strat.trend_state_4h(df4h)
            q = self.strat.quality_state_1h(df1h)
            dmi_ok = (q.plus_di > q.minus_di) if t.trend == S.Trend.BULL else (q.minus_di > q.plus_di)
            dmi = "DMI✓" if dmi_ok else "DMI×"
            if base._metal_halted(symbol, pd.Timestamp.now(tz="UTC")):
                why = "HALT"
            elif self.open_position_count() >= self.cfg.max_positions:
                why = f"MAX {self.cfg.max_positions}"
            elif t.trend == S.Trend.NEUTRAL:
                why = "WAIT 4H trend"
            elif q.q < self.cfg.min_trend_quality:
                why = f"WAIT Q<{self.cfg.min_trend_quality:.0f}"
            elif not dmi_ok:
                why = "WAIT 1H DMI align"
            else:
                why = "WAIT 15M location / 5M trigger"
            px = float(df5["close"].iloc[-1]) if len(df5) else 0.0
            self._view[symbol] = (
                f"4H={t.trend.value} HMA={'UP' if t.hma_state>0 else 'DOWN' if t.hma_state<0 else 'FLAT'} "
                f"| 1H Q={q.q:.0f} ADX={q.adx:.1f} CHOP={q.chop:.1f} {dmi} "
                f"| 5M px={px:.6g} | {why}"
            )
        except Exception as exc:
            self._view[symbol] = f"view error: {str(exc)[:80]}"

    async def _look_for_entry(self, symbol: str, st: dict):
        df5, df15, df1h, df4h = await self._entry_frames(symbol)
        if len(df5) < 70 or len(df15) < 90 or len(df1h) < 60 or len(df4h) < 70:
            self._view[symbol] = (
                f"warming up 5M={len(df5)} 15M={len(df15)} 1H={len(df1h)} 4H={len(df4h)}"
            )
            return

        self._set_view_v3(symbol, df5, df15, df1h, df4h)

        bar_key = df5.index[-1].isoformat()
        if st.get("last_bar") == bar_key:
            return
        st["last_bar"] = bar_key
        self._save_state()

        if base._metal_halted(symbol, pd.Timestamp.now(tz="UTC")):
            return
        if self.open_position_count() >= self.cfg.max_positions:
            return
        if time.time() < self._cooldown_until.get(symbol, 0):
            return

        sig = self.strat.generate_entry(df4h, df1h, df15, df5, has_open_position=False)
        if sig is None:
            return

        direction = "long" if sig.side == S.Side.LONG else "short"
        ticker = await self.client.fetch_ticker(symbol)
        fill_ref = float(ticker["last"])

        sl_dist = abs(float(sig.entry_price) - float(sig.stop_loss))
        tp_dist = abs(float(sig.take_profit) - float(sig.entry_price))
        sl = fill_ref - sl_dist if direction == "long" else fill_ref + sl_dist
        tp = fill_ref + tp_dist if direction == "long" else fill_ref - tp_dist

        balance = await self.client.fetch_balance_usdt()
        required_margin = float(self.cfg.margin_per_position_usd)
        notional = required_margin * float(self.cfg.leverage)
        if balance < required_margin:
            self._view[symbol] = f"insufficient balance ${balance:.2f} for ${required_margin:.2f} margin"
            return
        amount = notional / fill_ref if fill_ref > 0 else 0.0
        if amount <= 0 or amount * fill_ref < 5:
            return

        side = "buy" if direction == "long" else "sell"
        try:
            order = await self.client.create_order(
                symbol, side, amount, pos_side=direction, tp_price=tp, sl_price=sl
            )
        except Exception as exc:
            base.logger.error("[%s] order failed: %s", symbol, exc)
            await self.tg.send_text(f"❌ `{base._sym(symbol)}` entry order failed: {str(exc)[:150]}")
            return

        fill = order.avg_price or fill_ref
        sl = fill - sl_dist if direction == "long" else fill + sl_dist
        tp = fill + tp_dist if direction == "long" else fill - tp_dist
        st["pos"] = {
            "side": direction,
            "entry": fill,
            "sl": sl,
            "initial_sl": sl,
            "tp": tp,
            "risk": abs(fill - sl),
            "amount": order.amount or amount,
            "margin_usd": required_margin,
            "leverage": self.cfg.leverage,
            "notional_usd": notional,
            "opened_ms": int(time.time() * 1000),
            "exit_bar": None,
            "best_price": fill,
            "lock_stage": 0,
            "setup": sig.setup.value,
            "q_1h": sig.q_1h,
            "room_pct": sig.room_pct,
            "trigger": sig.trigger,
        }
        self._save_state()

        sl_pct = abs(fill - sl) / fill * 100 if fill else 0.0
        base.logger.info(
            "[%s] OPEN %s @ %.6g sl=%.6g tp=%.6g 4H=%s Q=%.0f setup=%s trigger=%s room=%.2f%%",
            symbol, direction.upper(), fill, sl, tp, sig.trend_4h.value,
            sig.q_1h, sig.setup.value, sig.trigger, sig.room_pct * 100,
        )
        caption = (
            f"🟢 *{base._sym(symbol)} {direction.upper()}* @ `{fill:.6g}`\n"
            f"4H `{sig.trend_4h.value}` | 1H Q `{sig.q_1h:.0f}` + DMI aligned "
            f"(ADX {sig.adx_1h:.1f}, CHOP {sig.chop_1h:.1f})\n"
            f"15M `{sig.setup.value}` → 5M `{sig.trigger}` | Room `{sig.room_pct*100:.2f}%`\n"
            f"{sig.reason}\n"
            f"Structure SL `{sl:.6g}` (−{sl_pct:.2f}%) | Final TP `{tp:.6g}`\n"
            f"T1 `+0.6%` → lock `+0.3%` | T2 `+1.0%` → lock `+0.7%`\n"
            f"Margin `${required_margin:.2f}` × `x{self.cfg.leverage}` ≈ `${notional:.2f}` notional"
        )
        chart = self._build_chart(symbol, df15, direction, fill, sl, tp)
        if chart:
            await self.tg._send_photo(chart, caption)
        else:
            await self.tg.send_text(caption)


async def _main():
    bot = Bot()
    loop = asyncio.get_running_loop()
    for sig_name in ("SIGINT", "SIGTERM"):
        try:
            loop.add_signal_handler(getattr(base._signal, sig_name), lambda: asyncio.ensure_future(bot.stop()))
        except (NotImplementedError, AttributeError):
            pass
    await bot.start()
    try:
        await bot.run_forever()
    finally:
        await bot.stop()


if __name__ == "__main__":
    asyncio.run(_main())
