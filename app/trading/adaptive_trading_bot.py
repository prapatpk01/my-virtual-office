"""Adaptive SMC Execution Bot V7.0.

New execution architecture:
  4H  TSS-style trend tunnel -> direction only
  15M market structure       -> HH/HL or LH/LL context, BOS/CHOCH
  5M  AMD                    -> accumulation, liquidity sweep/manipulation,
                                then distribution confirmation
  1M  IFVG                   -> precision execution trigger

Risk remains deterministic:
  TP1 1R closes 50% -> stop to BE+0.10R
  Remaining 50% becomes the runner toward TP2 2R
  Runner exits early on M15/M5 structural invalidation; TP2 closes all remaining.

The bot deliberately avoids adding 1H as another hard gate; the four layers above
already have distinct jobs and are easier to audit/backtest.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Callable, Dict, Optional
import json
import logging
import os
import sys
import time

TP1_R = float(os.getenv("SMC_TP1_R", "1.0"))
TP2_R = float(os.getenv("SMC_TP2_R", "2.0"))
BE_LOCK_R = float(os.getenv("SMC_BE_LOCK_R", "0.10"))
RISK_USDT = float(os.getenv("MOM_RISK_USDT", "5.0"))
MIN_SL_PCT = float(os.getenv("SMC_MIN_SL_PCT", "0.0025"))
MAX_SL_PCT = float(os.getenv("SMC_MAX_SL_PCT", "0.030"))
COOLDOWN_BARS = int(os.getenv("SMC_COOLDOWN_1M_BARS", "5"))
SUPPORTED_SCHEMAS = {"adaptive-smc-mtf-v1"}


@dataclass
class Position:
    direction: str
    entry: float
    sl: float
    initial_sl: float
    tp: float
    tp1: float
    tp2: float
    size: float
    initial_size: float
    strategy: str
    trigger: str
    opened_at: float
    tp1_hit: bool = False
    be_moved: bool = False
    style: str = "LEGACY"
    tp2_hit: bool = False
    runner_active: bool = False
    tss_bias: str = ""
    structure: str = ""
    amd_phase: str = ""
    ifvg_low: float = 0.0
    ifvg_high: float = 0.0
    manipulation_low: float = 0.0
    manipulation_high: float = 0.0
    sl_algo_id: str = ""


class TradingBot:
    def __init__(
        self,
        symbol: str,
        margin_usdt: float = 20.0,
        leverage: int = 20,
        paper: bool = True,
        state_file: str = "",
        execution_callback: Optional[Callable] = None,
        risk_usdt: float = RISK_USDT,
        **_kwargs,
    ):
        self.symbol = symbol
        self.margin_usdt = float(margin_usdt)
        self.leverage = int(leverage)
        self.paper = bool(paper)
        self.state_file = state_file
        self.execution_callback = execution_callback
        self.risk_usdt = float(risk_usdt)
        self.position: Optional[Position] = None
        self.cooldown_remaining = 0
        self.last_signal = "WARMUP"
        self.counts = {"scans": 0, "entries": 0, "cooldown": 0, "wait": 0}
        self._identity()
        self.load_state()

    @staticmethod
    def _identity() -> None:
        try:
            runner = sys.modules.get("run_bot") or sys.modules.get("__main__")
            if runner is not None and hasattr(runner, "logger"):
                runner.logger = logging.getLogger("adaptive_smc_v7")
            if runner is not None and hasattr(runner, "BUILD_ID"):
                runner.BUILD_ID = "adaptive-smc-v7.0-2026-08-29"
        except Exception:
            pass

    @property
    def position_open(self) -> bool:
        return self.position is not None

    def load_state(self) -> None:
        if not self.state_file or not os.path.exists(self.state_file):
            return
        try:
            with open(self.state_file, encoding="utf-8") as handle:
                raw = json.load(handle)
            position_raw = raw.get("position")
            if position_raw:
                allowed = {field.name for field in fields(Position)}
                clean = {key: value for key, value in position_raw.items() if key in allowed}
                self.position = Position(**clean)
            self.cooldown_remaining = int(raw.get("cooldown_remaining", 0))
        except Exception as exc:
            logging.getLogger("adaptive_smc_v7").warning("state load failed: %s", exc)
            self.position = None
            self.cooldown_remaining = 0

    def save_state(self) -> None:
        if not self.state_file:
            return
        os.makedirs(os.path.dirname(self.state_file) or ".", exist_ok=True)
        temp = self.state_file + ".tmp"
        with open(temp, "w", encoding="utf-8") as handle:
            json.dump({
                "position": asdict(self.position) if self.position else None,
                "cooldown_remaining": self.cooldown_remaining,
            }, handle)
        os.replace(temp, self.state_file)

    def _debug(self, i: Dict, reason: str) -> str:
        symbol = self.symbol.split("/")[0]
        if reason == "COOLDOWN":
            return f"SMC V7 · {symbol} · COOLDOWN {self.cooldown_remaining}x1M · WAIT"
        tss = str(i.get("tss_bias", "NEUTRAL"))
        structure = str(i.get("structure", "UNKNOWN"))
        amd = str(i.get("amd_phase", "WAIT"))
        l_ifvg = "Y" if (i.get("ifvg_long") or {}).get("valid") else "-"
        s_ifvg = "Y" if (i.get("ifvg_short") or {}).get("valid") else "-"
        return (
            f"SMC V7 · {symbol} · 4H={tss} · M15={structure} · M5={amd} · "
            f"M1 IFVG L={l_ifvg}/S={s_ifvg} · WAIT"
        )

    def _build(self, i: Dict, direction: str, trigger: str, live_price: float):
        entry = float(live_price or i.get("close", 0.0))
        if entry <= 0:
            return None
        raw_sl = float(i.get("sl", 0.0) or 0.0)
        if raw_sl <= 0:
            return None

        if direction == "LONG":
            if raw_sl >= entry:
                return None
            risk = entry - raw_sl
        else:
            if raw_sl <= entry:
                return None
            risk = raw_sl - entry

        min_risk = entry * MIN_SL_PCT
        if risk < min_risk:
            raw_sl = entry - min_risk if direction == "LONG" else entry + min_risk
            risk = min_risk

        sl_pct = risk / entry
        # Do not move an excessively wide structure stop inward: that would put
        # the SL back inside the manipulation. Reject the setup instead.
        if sl_pct > MAX_SL_PCT:
            self.last_signal = f"WAIT SMC stop too wide {sl_pct*100:.2f}% > {MAX_SL_PCT*100:.2f}%"
            return None

        if direction == "LONG":
            tp1, tp2 = entry + TP1_R * risk, entry + TP2_R * risk
        else:
            tp1, tp2 = entry - TP1_R * risk, entry - TP2_R * risk

        size = min(
            self.risk_usdt / max(risk, 1e-12),
            (self.margin_usdt * self.leverage) / max(entry, 1e-12),
        )
        if size <= 0:
            return None

        return {
            "direction": direction,
            "style": "SMC_MTF_V1",
            "strategy": "adaptive_smc_mtf_v1",
            "trigger": trigger,
            "entry": entry,
            "sl": raw_sl,
            "tp": tp2,
            "tp1": tp1,
            "tp2": tp2,
            "size": size,
            "risk_usdt": size * risk,
            "sl_pct": 100.0 * risk / entry,
            "tss_bias": str(i.get("tss_bias", "")),
            "tss_score": float(i.get("tss_score", 0.0)),
            "structure": str(i.get("structure", "")),
            "structure_bias": str(i.get("structure_bias", "")),
            "amd_phase": str(i.get("amd_phase", "")),
            "ifvg_low": float(i.get("ifvg_low", 0.0)),
            "ifvg_high": float(i.get("ifvg_high", 0.0)),
            "manipulation_low": float(i.get("manipulation_low", 0.0)),
            "manipulation_high": float(i.get("manipulation_high", 0.0)),
            "rsi1": float(i.get("rsi1", 50.0)),
            "atr1": float(i.get("atr1", 0.0)),
        }

    def _close(self, price: float, reason: str):
        p = self.position
        assert p
        pnl = (price - p.entry) * p.size if p.direction == "LONG" else (p.entry - price) * p.size
        initial_risk = abs(p.entry - p.initial_sl) * max(p.initial_size, 1e-12)
        r_multiple = pnl / initial_risk if initial_risk else 0.0
        payload = {
            "symbol": self.symbol,
            "direction": p.direction,
            "style": p.style,
            "price": price,
            "entry": p.entry,
            "sl": p.sl,
            "tp": p.tp2,
            "tp1": p.tp1,
            "tp2": p.tp2,
            "size": p.size,
            "strategy": p.strategy,
            "trigger": p.trigger,
            "reason": reason,
            "pnl": pnl,
            "r_multiple": r_multiple,
        }
        if self.execution_callback:
            self.execution_callback("CLOSE_" + p.direction, payload)
        self.position = None
        if reason in {"SL", "LOCKED_SL"}:
            self.cooldown_remaining = COOLDOWN_BARS
        self.save_state()
        self.last_signal = f"CLOSE {reason} pnl=${pnl:+.2f} r={r_multiple:+.2f}R"
        return {"event": "CLOSE", **payload}

    def _partial(self, price: float, qty: float, reason: str, r_multiple: float):
        p = self.position
        assert p
        requested_qty = min(max(float(qty), 0.0), p.size)
        payload = {
            "symbol": self.symbol,
            "direction": p.direction,
            "style": p.style,
            "price": float(price),
            "entry": p.entry,
            "size": requested_qty,
            "trigger": p.trigger,
            "reason": reason,
        }
        execution_result = None
        if self.execution_callback:
            execution_result = self.execution_callback("CLOSE_PARTIAL", payload)

        actual_qty = requested_qty
        actual_price = float(price)
        actual_pnl = None
        already_flat = False
        if isinstance(execution_result, dict):
            already_flat = bool(execution_result.get("_already_flat"))
            filled = float(execution_result.get("_exit_fill_sz") or 0.0)
            fill_px = float(execution_result.get("_exit_avg_px") or 0.0)
            if already_flat:
                actual_qty = p.size
            elif filled > 0:
                actual_qty = min(p.size, filled)
            if fill_px > 0:
                actual_price = fill_px
            if "_realized_pnl" in execution_result:
                actual_pnl = float(execution_result.get("_realized_pnl") or 0.0)

        pnl = actual_pnl
        if pnl is None:
            pnl = ((actual_price - p.entry) * actual_qty if p.direction == "LONG"
                   else (p.entry - actual_price) * actual_qty)
        initial_risk = abs(p.entry - p.initial_sl) * max(p.initial_size, 1e-12)
        realized_r = pnl / initial_risk if initial_risk else r_multiple
        return {
            "event": "PARTIAL",
            **payload,
            "price": actual_price,
            "size": actual_qty,
            "pnl": pnl,
            "r_multiple": realized_r,
            "already_flat": already_flat,
        }

    def _amend_exchange_sl(self, new_sl: float) -> None:
        p = self.position
        if not p or not p.sl_algo_id or not self.execution_callback:
            return
        try:
            self.execution_callback("AMEND_SL", {
                "symbol": self.symbol,
                "direction": p.direction,
                "sl_algo_id": p.sl_algo_id,
                "new_sl": float(new_sl),
            })
        except Exception as exc:
            logging.getLogger("adaptive_smc_v7").warning(
                "[%s] exchange SL amend failed; local stop remains %.4f: %s",
                self.symbol, new_sl, exc,
            )

    def check_price(self, price: float):
        p = self.position
        if not p:
            return None

        if (p.direction == "LONG" and price <= p.sl) or (p.direction == "SHORT" and price >= p.sl):
            return self._close(price, "LOCKED_SL" if p.be_moved else "SL")

        if not p.tp1_hit and (
            (p.direction == "LONG" and price >= p.tp1)
            or (p.direction == "SHORT" and price <= p.tp1)
        ):
            qty = min(p.initial_size * 0.50, p.size)
            event = self._partial(price, qty, "TP1", TP1_R)
            p.size = max(0.0, p.size - float(event.get("size", qty)))
            if event.get("already_flat") or p.size <= 1e-12:
                # Minimum-contract rounding can turn an intended partial into a
                # full exchange close. Keep local state truthful immediately.
                self.position = None
                self.save_state()
                event["event"] = "CLOSE"
                event["reason"] = "TP1_EXCHANGE_FULL"
                return event

            p.tp1_hit = True
            p.runner_active = True
            risk_unit = abs(p.entry - p.initial_sl)
            lock = BE_LOCK_R * risk_unit
            p.sl = p.entry + lock if p.direction == "LONG" else p.entry - lock
            p.be_moved = True
            self._amend_exchange_sl(p.sl)
            self.save_state()
            return event

        if p.tp1_hit and not p.tp2_hit and (
            (p.direction == "LONG" and price >= p.tp2)
            or (p.direction == "SHORT" and price <= p.tp2)
        ):
            if p.style == "SMC_MTF_V1":
                # SMC V7 aligns the local TP2 with the exchange-attached TP2:
                # close all remaining size at 2R; no conflicting partial order.
                p.tp2_hit = True
                return self._close(price, "TP2")

            # Preserve legacy V6 position semantics across a deployment.
            qty = min(p.initial_size * 0.25, p.size)
            event = self._partial(price, qty, "TP2_PARTIAL", TP2_R)
            p.size = max(0.0, p.size - float(event.get("size", qty)))
            if event.get("already_flat") or p.size <= 1e-12:
                self.position = None
                self.save_state()
                event["event"] = "CLOSE"
                event["reason"] = "TP2_EXCHANGE_FULL"
                return event
            p.tp2_hit = True
            p.runner_active = True
            risk_unit = abs(p.entry - p.initial_sl)
            p.sl = p.entry + risk_unit if p.direction == "LONG" else p.entry - risk_unit
            p.be_moved = True
            self._amend_exchange_sl(p.sl)
            self.save_state()
            return event

        return None

    def reconcile_flat(self, price: float, reason: str = "EXCHANGE_CLOSED"):
        return self.reconcile_exchange_closed(price, reason) if self.position else None

    def reconcile_exchange_closed(self, price: float, reason: str = "EXCHANGE_CLOSED"):
        """Clear a local position after OKX reports the symbol is already flat.

        This path deliberately does NOT send another close order. It prevents a
        ghost local position when an exchange-side SL/TP or minimum-lot rounding
        closed the trade between polling cycles.
        """
        p = self.position
        if not p:
            return None
        px = float(price or p.entry)
        pnl = (px - p.entry) * p.size if p.direction == "LONG" else (p.entry - px) * p.size
        initial_risk = abs(p.entry - p.initial_sl) * max(p.initial_size, 1e-12)
        r_multiple = pnl / initial_risk if initial_risk else 0.0
        payload = {
            "symbol": self.symbol, "direction": p.direction, "style": p.style,
            "price": px, "entry": p.entry, "sl": p.sl, "tp": p.tp2,
            "tp1": p.tp1, "tp2": p.tp2, "size": p.size,
            "strategy": p.strategy, "trigger": p.trigger, "reason": reason,
            "pnl": pnl, "r_multiple": r_multiple,
        }
        self.position = None
        self.save_state()
        self.last_signal = f"RECONCILED {reason} @ {px:.4f}"
        return {"event": "CLOSE", **payload}

    def adopt_exchange_position(self, direction: str, entry: float, size: float,
                                sl: float, tp2: float, sl_algo_id: str = "") -> None:
        """Adopt a live OKX position after a restart when local /tmp state is gone.

        Adoption is only called when the runner can recover a real exchange SL;
        it never invents a stop for a live position.
        """
        direction = direction.upper()
        risk = abs(float(entry) - float(sl))
        if direction not in {"LONG", "SHORT"} or entry <= 0 or size <= 0 or risk <= 0:
            raise ValueError("invalid exchange position for adoption")
        tp1 = entry + risk * TP1_R if direction == "LONG" else entry - risk * TP1_R
        target2 = float(tp2 or 0.0)
        if target2 <= 0:
            target2 = entry + risk * TP2_R if direction == "LONG" else entry - risk * TP2_R
        self.position = Position(
            direction=direction, entry=float(entry), sl=float(sl), initial_sl=float(sl),
            tp=target2, tp1=tp1, tp2=target2, size=float(size), initial_size=float(size),
            strategy="recovered_exchange", trigger="Recovered after restart",
            opened_at=time.time(), style="RECOVERED", sl_algo_id=str(sl_algo_id or ""),
        )
        self.save_state()
        self.last_signal = f"RECOVERED {direction} entry={entry:.4f} size={size:.6g}"

    def on_bar(self, i: Dict, _i1=None, _i4=None, price: float = 0.0):
        if not i:
            self.last_signal = "WAIT INDICATOR_WARMUP"
            return None
        if i.get("schema") not in SUPPORTED_SCHEMAS:
            raise RuntimeError(f"ADAPTIVE_SMC_SCHEMA_MISMATCH: {i.get('schema')}")

        live_price = float(price or i.get("close", 0.0))

        if self.position:
            event = self.check_price(live_price)
            if event:
                return event

            p = self.position
            if p.runner_active:
                if p.style == "SMC_MTF_V1":
                    invalidate = (
                        p.direction == "LONG" and bool(i.get("runner_exit_long"))
                    ) or (
                        p.direction == "SHORT" and bool(i.get("runner_exit_short"))
                    )
                    if invalidate:
                        return self._close(float(i.get("close", live_price)), "RUNNER_STRUCTURE_EXIT")
                else:
                    # Preserve the old V6 runner rule for positions that existed
                    # before the SMC upgrade: closed M15 failure of EMA20.
                    m15_close = float(i.get("m15_close", 0.0) or 0.0)
                    m15_ema20 = float(i.get("m15_ema20", 0.0) or 0.0)
                    if m15_close and m15_ema20:
                        legacy_fail = (p.direction == "LONG" and m15_close < m15_ema20) or (
                            p.direction == "SHORT" and m15_close > m15_ema20
                        )
                        if legacy_fail:
                            return self._close(m15_close, "RUNNER_EMA20_EXIT")

            self.last_signal = (
                f"MANAGE {p.style} {p.direction} | SL={p.sl:.4f} | "
                f"TP1={p.tp1:.4f} | TP2={p.tp2:.4f} | Runner={int(p.runner_active)}"
            )
            return None

        self.counts["scans"] += 1
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            self.counts["cooldown"] += 1
            self.save_state()
            self.last_signal = self._debug(i, "COOLDOWN")
            return None

        direction = "LONG" if i.get("long_signal") else "SHORT" if i.get("short_signal") else "NONE"
        if direction == "NONE":
            self.counts["wait"] += 1
            self.last_signal = self._debug(i, "WAIT")
            return None

        trigger = str(i.get("trigger") or "4H → M15 → M5 → M1 IFVG")
        payload = self._build(i, direction, trigger, live_price)
        if not payload:
            if not self.last_signal.startswith("WAIT SMC stop"):
                self.last_signal = f"SMC V7 · {self.symbol.split('/')[0]} · WAIT RISK BUILD"
            return None

        payload["symbol"] = self.symbol
        execution_result = None
        if self.execution_callback:
            execution_result = self.execution_callback("OPEN_" + direction, payload)

        # Use actual filled coin quantity when the live adapter reports it.
        actual_size = payload["size"]
        if isinstance(execution_result, dict):
            actual_size = float(execution_result.get("_filled_coins") or actual_size)

        self.position = Position(
            direction=direction,
            entry=payload["entry"],
            sl=payload["sl"],
            initial_sl=payload["sl"],
            tp=payload["tp2"],
            tp1=payload["tp1"],
            tp2=payload["tp2"],
            size=actual_size,
            initial_size=actual_size,
            strategy=payload["strategy"],
            trigger=payload["trigger"],
            opened_at=time.time(),
            style="SMC_MTF_V1",
            tss_bias=payload["tss_bias"],
            structure=payload["structure"],
            amd_phase=payload["amd_phase"],
            ifvg_low=payload["ifvg_low"],
            ifvg_high=payload["ifvg_high"],
            manipulation_low=payload["manipulation_low"],
            manipulation_high=payload["manipulation_high"],
            sl_algo_id=(str(execution_result.get("_sl_algo_id") or "")
                        if isinstance(execution_result, dict) else ""),
        )
        self.counts["entries"] += 1
        self.save_state()
        self.last_signal = f"ENTRY {direction} · {trigger}"
        return {"event": "OPEN", **payload}
