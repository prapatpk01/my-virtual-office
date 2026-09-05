"""EMA Hybrid A+B+C+D Quality V2.4 runtime wrapper.

A = EMA8/13 cross
B = precision pullback reclaim / strict micro BOS
C = Bollinger + MACD + KDJ trend-aligned reversal confirmation
D = MA5/MA20 trend pullback + candle/volume confirmation

Keeps the proven EMA Hybrid runtime/journal/Telegram lifecycle code in main_core.py,
plus XAU/XAG correlation protection and sync/async client compatibility.
"""
from __future__ import annotations

import asyncio
import inspect
import os
import signal
from datetime import datetime, timezone

import main_core as core
import strategy_d as setup_d

_LOG = core._LOG


# Extend the existing runtime attribution without rewriting main_core.py.
_BASE_SETUP_KEY = core._setup_key_from_trigger
_BASE_SETUP_LABEL = core._setup_label_from_trigger
_BASE_SETUP_TEXT = core._setup_label_from_text


def _setup_key_from_trigger(trigger: str) -> str:
    t = str(trigger or "").upper()
    if "MA5_MA20_" in t:
        return "D_MA5_MA20"
    if "BOLL_MACD_KDJ_" in t:
        return "C_BOLL_MACD_KDJ"
    return _BASE_SETUP_KEY(trigger)


def _setup_label_from_trigger(trigger: str) -> str:
    key = _setup_key_from_trigger(trigger)
    if key == "D_MA5_MA20":
        return "D · MA5/MA20 TREND PULLBACK"
    if key == "C_BOLL_MACD_KDJ":
        return "C · BOLL+MACD+KDJ REVERSAL"
    return _BASE_SETUP_LABEL(trigger)


def _setup_label_from_text(text: str) -> str | None:
    upper = str(text or "").upper()
    if "MA5_MA20_" in upper:
        return "D · MA5/MA20 TREND PULLBACK"
    if "BOLL_MACD_KDJ_" in upper:
        return "C · BOLL+MACD+KDJ REVERSAL"
    return _BASE_SETUP_TEXT(text)


core._setup_key_from_trigger = _setup_key_from_trigger
core._setup_label_from_trigger = _setup_label_from_trigger
core._setup_label_from_text = _setup_label_from_text


class Bot(core.Bot):
    METAL_CORR_GUARD = os.getenv("EMA_METAL_CORR_GUARD", "true").strip().lower() in {
        "1", "true", "yes", "on",
    }

    def __init__(self):
        super().__init__()
        self.strat = setup_d.EMAHybridProStrategy(self.cfg.strategy_config())
        self._install_client_await_compat()
        self.strat.correlation_guard = self._metal_correlation_blocked

    def _install_client_await_compat(self) -> None:
        """Wrap legacy sync helpers so awaited EMA lifecycle code is safe."""
        quantize = getattr(self.client, "quantize_amount", None)
        if callable(quantize) and not inspect.iscoroutinefunction(quantize):
            async def _quantize_amount(*args, __fn=quantize, **kwargs):
                result = __fn(*args, **kwargs)
                if result is None:
                    _LOG.warning("[EMA COMPAT] quantize_amount returned None; using zero-sized result")
                    return 0.0, 0.0
                return result
            self.client.quantize_amount = _quantize_amount
            _LOG.info("[EMA COMPAT] wrapped sync quantize_amount as awaitable")

        move_sl = getattr(self.client, "move_sl_to_breakeven", None)
        if callable(move_sl) and not inspect.iscoroutinefunction(move_sl):
            async def _move_sl_to_breakeven(*args, __fn=move_sl, **kwargs):
                result = __fn(*args, **kwargs)
                return bool(result) if result is not None else False
            self.client.move_sl_to_breakeven = _move_sl_to_breakeven
            _LOG.info("[EMA COMPAT] wrapped sync move_sl_to_breakeven as awaitable")

    async def _entry_frames(self, symbol: str):
        frames = await super()._entry_frames(symbol)
        for frame in frames:
            try:
                frame.attrs["symbol"] = symbol
            except Exception:
                pass
        return frames

    def _metal_correlation_blocked(self, symbol: str, side) -> bool:
        """Do not add XAU and XAG in the same direction at the same time."""
        if not self.METAL_CORR_GUARD:
            return False

        root = str(symbol).split("/")[0].upper()
        if root not in {"XAU", "XAG"}:
            return False

        peer_root = "XAG" if root == "XAU" else "XAU"
        peer_symbol = next(
            (s for s in self.cfg.symbols if str(s).split("/")[0].upper() == peer_root),
            None,
        )
        if not peer_symbol:
            return False

        peer_pos = (self.state.get(peer_symbol) or {}).get("pos") or {}
        peer_side = str(peer_pos.get("side") or "").lower()
        wanted = str(getattr(side, "value", side) or "").lower()
        blocked = peer_side in {"long", "short"} and peer_side == wanted

        if blocked:
            self._view[symbol] = (
                f"CORR BLOCK | {root} {wanted.upper()} blocked because "
                f"{peer_root} {peer_side.upper()} is already open"
            )
            _LOG.info(
                "[%s] correlation block: %s %s already open",
                symbol, peer_root, peer_side.upper(),
            )
        return blocked

    async def _build_stats_report(self) -> str:
        """Add Setup C/D attribution to the existing A/B advanced stats."""
        report = await super()._build_stats_report()
        if not self.cfg.paper:
            return report

        now = datetime.now(timezone.utc)
        month_start = int(datetime(now.year, now.month, 1, tzinfo=timezone.utc).timestamp() * 1000)
        month_rows = [r for r in self.ema_journal if int(r.get("close_ms") or 0) >= month_start]

        def metrics(block: list[dict]):
            wins = sum(1 for r in block if float(r.get("pnl") or 0.0) > 0)
            net = sum(float(r.get("pnl") or 0.0) for r in block)
            gross_win = sum(float(r.get("pnl") or 0.0) for r in block if float(r.get("pnl") or 0.0) > 0)
            gross_loss = abs(sum(float(r.get("pnl") or 0.0) for r in block if float(r.get("pnl") or 0.0) < 0))
            pf = gross_win / gross_loss if gross_loss > 1e-12 else (float("inf") if gross_win > 0 else 0.0)
            wr = wins / len(block) * 100.0 if block else 0.0
            pf_text = "∞" if pf == float("inf") else f"{pf:.2f}"
            return wr, pf_text, net

        specs = (
            (
                "C BOLL/MACD/KDJ",
                lambda r: str(r.get("setup") or "") == "C_BOLL_MACD_KDJ"
                or "BOLL_MACD_KDJ_" in str(r.get("trigger") or "").upper(),
            ),
            (
                "D MA5/MA20",
                lambda r: str(r.get("setup") or "") == "D_MA5_MA20"
                or "MA5_MA20_" in str(r.get("trigger") or "").upper(),
            ),
        )

        extra_lines = []
        for label, predicate in specs:
            block = [r for r in month_rows if predicate(r)]
            if not block:
                continue
            wr, pf_text, net = metrics(block)
            extra_lines.append(f"{label:16s} {len(block)} | {wr:.0f}% WR | PF {pf_text} | ${net:+.2f}")

        if not extra_lines:
            return report

        lines = report.splitlines()
        section = next((i for i, line in enumerate(lines) if line.startswith("BY SETUP —")), None)
        if section is None:
            return report

        insert_at = section + 2
        while insert_at < len(lines) and lines[insert_at] != "":
            if lines[insert_at].startswith("(no completed"):
                lines.pop(insert_at)
                continue
            insert_at += 1
        for line in extra_lines:
            lines.insert(insert_at, line)
            insert_at += 1
        return "\n".join(lines)

    async def start(self):
        problems = self.cfg.validate_live()
        if problems:
            raise RuntimeError("Cannot start: " + "; ".join(problems))
        if not self.cfg.paper and not await self.client.ensure_hedge_mode():
            raise RuntimeError("Could not confirm OKX hedge mode.")

        balance = await self.client.fetch_balance_usdt()
        _LOG.info(
            "=== EMA HYBRID A+B+C+D QUALITY V2.4 [%s] symbols=%s margin=$%.2f "
            "leverage=x%d max_pos=%d balance=%.2f ===",
            "PAPER" if self.cfg.paper else "LIVE",
            self.cfg.symbols,
            self.cfg.margin_per_position_usd,
            self.cfg.leverage,
            self.cfg.max_positions,
            balance,
        )

        await self._reconcile_startup()
        self._running = True

        if self.tg.enabled:
            asyncio.create_task(self._command_loop())
            mode = "PAPER" if self.cfg.paper else "LIVE"
            await self.tg.send_text(
                f"📈 *EMA Hybrid A+B+C+D Quality V2.4 — {mode}*\n"
                f"Symbols: `{', '.join(self.cfg.symbols)}`\n"
                f"Balance: `{balance:.2f}` USDT | Margin `${self.cfg.margin_per_position_usd:.2f}`/position "
                f"| Leverage `x{self.cfg.leverage}` | Max `{self.cfg.max_positions}` positions\n\n"
                f"15M Bias: LONG Close>SMA{self.strat.SMA_LEN} + RSI≥`{self.strat.BIAS_RSI_LONG_MIN:.0f}` "
                f"+ SMA slope UP | SHORT Close<SMA{self.strat.SMA_LEN} + RSI≤`{self.strat.BIAS_RSI_SHORT_MAX:.0f}` "
                "+ SMA slope DOWN\n"
                f"A: EMA{self.strat.EMA_FAST}/{self.strat.EMA_SLOW} fresh cross + direction candle + expanding spread\n"
                f"C: BOLL({self.strat.BOLL_LEN},{self.strat.BOLL_STD:g}) band re-entry + MACD({self.strat.MACD_FAST},{self.strat.MACD_SLOW},{self.strat.MACD_SIGNAL}) momentum + "
                f"KDJ({self.strat.KDJ_LEN},{self.strat.KDJ_SMOOTH_K},{self.strat.KDJ_SMOOTH_D}) OS≤`{self.strat.KDJ_OS:.0f}`/OB≥`{self.strat.KDJ_OB:.0f}` cross\n"
                f"D: MA{self.strat.D_MA_FAST}/MA{self.strat.D_MA_SLOW} recent cross → wait pullback → candle confirm + volume `≥{self.strat.D_VOLUME_RATIO_MIN:.2f}x` avg\n"
                f"D Guard: MA20 slope with bias + ADX `≥{self.strat.D_ADX_MIN:.0f}` + CHOP `≤{self.strat.D_CHOP_MAX:.0f}` + spread `≥{self.strat.D_MIN_SPREAD_ATR:.2f} ATR` + entry `≤{self.strat.D_MAX_ENTRY_ATR:.2f} ATR`\n"
                f"B Core: fresh pullback `≤{self.strat.B_FRESH_LOOKBACK}` bars | EMA13 zone `±{self.strat.PULLBACK_TOUCH_ATR:.2f} ATR` | max depth `{self.strat.B_MAX_PULLBACK_DEPTH_ATR:.2f} ATR`\n"
                f"B1 Reclaim: candle confirm + close beyond EMA13 `≥{self.strat.B1_RECLAIM_BUFFER_ATR:.2f} ATR` + spread not contracting + entry `≤{self.strat.B1_MAX_ENTRY_ATR:.2f} ATR` | "
                f"ADX `≥{self.strat.B1_ADX_MIN:.0f}` (if <`{self.strat.B1_ADX_FREEPASS:.0f}` must rise) | CHOP `≤{self.strat.B1_CHOP_MAX:.0f}`\n"
                f"B2 Micro BOS: break `≥{self.strat.MICRO_BOS_BREAK_ATR:.2f} ATR` + spread expanding + ADX `≥{self.strat.MICRO_BOS_ADX_MIN:.0f}` rising + CHOP `≤{self.strat.MICRO_BOS_CHOP_MAX:.0f}` + entry `≤{self.strat.B2_MAX_ENTRY_ATR:.2f} ATR`\n"
                "Priority: `A > C > D > B` when multiple setups fire on the same 5M close\n"
                f"SL Gate: `{self.strat.SL_MIN_PCT*100:.2f}%–{self.strat.SL_MAX_PCT*100:.2f}%` | Structure buffer `{self.strat.SL_BUFFER_ATR:.2f} ATR`\n"
                f"TP1: `+{self.TP1_R:.1f}R` → trim `{self.TP1_TRIM_PCT*100:.0f}%` → SL `BE+{self.TP1_LOCK_R:.2f}R`\n"
                f"TP2: next 5M liquidity/swing with room `≥{self.strat.TP2_MIN_RR:.1f}R`\n"
                f"XAU/XAG same-direction guard: `{'ON' if self.METAL_CORR_GUARD else 'OFF'}`\n"
                "Telegram: Entry + Setup A/B/C/D + TP1 + TP2/SL/TP1_LOCK alerts\n"
                "PAPER entries: `24/7` | LIVE entries: `24/5` | Open positions managed: `24/7`"
            )

        _LOG.info(
            "EMA Hybrid A+B+C+D Quality V2.4 active: A>C>D>B priority, "
            "MA5/20 pullback D, precision B, triple-confirm C, SL sanity, "
            "metal correlation guard and await-compat"
        )


async def _main():
    bot = Bot()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, lambda: asyncio.ensure_future(bot.stop()))
        except (NotImplementedError, AttributeError):
            pass
    await bot.start()
    try:
        await bot.run_forever()
    finally:
        await bot.stop()


if __name__ == "__main__":
    asyncio.run(_main())
