"""Precision Trend Structure V2 bot.

4H EMA20/50 + HMA16 selects trade side, 1H ADX+CHOP Q confirms market quality,
15M location/setup is armed before a micro-structure trigger. Uses fixed $20
margin, x20 isolated leverage, max 2 positions, staged profit locks, and
structure/HTF invalidation exits.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import signal as _signal
import sys
import time

import numpy as np
import pandas as pd

from config import Config
import strategy as S

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "signal_regime_bot"))
from exchange_client import ExchangeClient
from telegram_notifier import TelegramNotifier
try:
    from chart_engine import build_entry_chart
except Exception:
    build_entry_chart = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S", stream=sys.stdout, force=True)
logger = logging.getLogger("precision_structure")
_TF_MIN = 15

def _sym(symbol: str) -> str:
    return symbol.split("/")[0]

def _ohlcv_to_df(raw: list) -> pd.DataFrame:
    if not raw:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.set_index("ts").astype(float)

def _is_metal(symbol: str) -> bool:
    u = symbol.upper()
    return any(k in u for k in ("XAU", "XAG", "GOLD", "SILVER", "XPT", "XPD"))

def _metal_halted(symbol: str, ts: pd.Timestamp) -> bool:
    if not _is_metal(symbol):
        return False
    wd, hr = ts.weekday(), ts.hour
    return (wd == 4 and hr >= 21) or wd == 5 or (wd == 6 and hr < 21)

class Bot:
    def __init__(self):
        self.cfg = Config()
        self.strat = S.PrecisionTrendStructureV2(self.cfg.strategy_config())
        self.client = ExchangeClient(api_key=self.cfg.okx_api_key, api_secret=self.cfg.okx_secret, passphrase=self.cfg.okx_passphrase, paper=self.cfg.paper, leverage=self.cfg.leverage, margin_mode=self.cfg.margin_mode, fee_rate=self.cfg.fee_rate)
        self.tg = TelegramNotifier(self.cfg.telegram_token, self.cfg.telegram_chat_id)
        self._state_path = os.path.join(self.cfg.state_dir, "hma_state.json")
        self._journal_path = os.path.join(self.cfg.state_dir, "trade_journal.jsonl")
        self.state = self._load_state()
        self.journal = self._load_journal()
        self._journaled_close_ms = {(e["symbol"], int(e["close_ms"] // 60000)) for e in self.journal}
        self._view = {s: "starting…" for s in self.cfg.symbols}
        self._cooldown_until = {}
        self._tg_offset = 0
        self._running = False
        self._last_status_ts = 0.0

    def _load_state(self):
        try:
            with open(self._state_path) as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {}

    def _save_state(self):
        os.makedirs(self.cfg.state_dir, exist_ok=True)
        tmp = self._state_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.state, f)
        os.replace(tmp, self._state_path)

    def _sym_state(self, symbol):
        return self.state.setdefault(symbol, {"last_bar": None, "pos": None})

    def open_position_count(self):
        return sum(1 for st in self.state.values() if st.get("pos"))

    def _load_journal(self):
        out = []
        try:
            with open(self._journal_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        out.append(json.loads(line))
        except (FileNotFoundError, OSError):
            pass
        return out

    async def start(self):
        problems = self.cfg.validate_live()
        if problems:
            raise RuntimeError("Cannot start: " + "; ".join(problems))
        if not self.cfg.paper and not await self.client.ensure_hedge_mode():
            raise RuntimeError("Could not confirm OKX hedge mode.")
        balance = await self.client.fetch_balance_usdt()
        logger.info("=== HMA base runtime [%s] symbols=%s margin=$%.2f leverage=x%d max_pos=%d balance=%.2f ===", "PAPER" if self.cfg.paper else "LIVE", self.cfg.symbols, self.cfg.margin_per_position_usd, self.cfg.leverage, self.cfg.max_positions, balance)
        await self._reconcile_startup()
        self._running = True
        if self.tg.enabled:
            asyncio.create_task(self._command_loop())

    async def _reconcile_startup(self):
        for symbol in self.cfg.symbols:
            st = self._sym_state(symbol)
            pos = st.get("pos")
            for side in ("long", "short"):
                amt = await self.client.fetch_position_amount(symbol, side)
                tracked = pos is not None and pos.get("side") == side
                if amt > 0 and not tracked:
                    st["pos"] = {"side": side, "entry": 0.0, "sl": 0.0, "tp": 0.0, "risk": 0.0, "amount": amt, "opened_ms": int(time.time() * 1000), "adopted": True}
                if tracked and amt <= 0:
                    st["pos"] = None
        self._save_state()

    async def run_forever(self):
        while self._running:
            for symbol in self.cfg.symbols:
                try:
                    await self._process(symbol)
                except Exception as exc:
                    logger.error("[%s] unhandled: %s", symbol, exc, exc_info=True)
            self._maybe_status_log()
            await asyncio.sleep(self.cfg.poll_interval_sec)

    async def stop(self):
        self._running = False
        await self.client.close()

    async def _process(self, symbol):
        st = self._sym_state(symbol)
        if st.get("pos"):
            await self._manage(symbol, st)
        else:
            await self._look_for_entry(symbol, st)

    async def _frame(self, symbol, tf, minutes, limit=300):
        now_ms = int(time.time() * 1000)
        raw = await self.client.fetch_ohlcv(symbol, tf, limit=limit)
        df = _ohlcv_to_df(raw)
        if df.empty:
            return df
        close_ms = (df.index.as_unit("ns").asi8 // 1_000_000) + minutes * 60_000
        return df[close_ms <= now_ms]

    async def _frames(self, symbol):
        df15, df1h, df4h = await asyncio.gather(self._frame(symbol, "15m", 15, 320), self._frame(symbol, "1h", 60, 240), self._frame(symbol, "4h", 240, 220))
        return df15, df1h, df4h

    async def _look_for_entry(self, symbol, st):
        df15, df1h, df4h = await self._frames(symbol)
        if len(df15) < 80 or len(df1h) < 60 or len(df4h) < 60:
            return
        sig = self.strat.generate_entry(df4h, df1h, df15, has_open_position=False)
        if sig is None:
            return

    async def _manage(self, symbol, st):
        pos = st.get("pos") or {}
        side = pos.get("side")
        if side not in ("long", "short"):
            return
        amt = await self.client.fetch_position_amount(symbol, side)
        if amt <= 0:
            st["pos"] = None
            self._save_state()

    def _view_line(self, symbol):
        st = self.state.get(symbol) or {}
        pos = st.get("pos")
        if pos:
            return f"OPEN {str(pos.get('side','?')).upper()} @ {float(pos.get('entry') or 0):.6g}"
        return self._view.get(symbol, "flat")

    def _maybe_status_log(self):
        now = time.time()
        if now - self._last_status_ts < self.cfg.status_log_interval_sec:
            return
        self._last_status_ts = now
        for symbol in self.cfg.symbols:
            logger.info("%-16s %s", _sym(symbol), self._view_line(symbol))

    async def _command_loop(self):
        while self._running:
            try:
                updates = await self.tg.get_updates(self._tg_offset + 1)
                for update in updates:
                    self._tg_offset = max(self._tg_offset, int(update.get("update_id", 0)))
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("[TG] %s", exc)
                await asyncio.sleep(5)

async def _main():
    bot = Bot()
    loop = asyncio.get_running_loop()
    for sig_name in ("SIGINT", "SIGTERM"):
        try:
            loop.add_signal_handler(getattr(_signal, sig_name), lambda: asyncio.ensure_future(bot.stop()))
        except (NotImplementedError, AttributeError):
            pass
    await bot.start()
    try:
        await bot.run_forever()
    finally:
        await bot.stop()

if __name__ == "__main__":
    asyncio.run(_main())
