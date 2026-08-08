"""Trend Confirm V5 — simplified direction/quality layers.

Layer 1 (4H) chooses direction only:
  EMA20/50 25 + EMA20 slope 25 + price location 20 + structure 30.
Layer 2 (1H) decides trade quality only:
  ADX 25 + CHOP 20 + structure 20 + momentum 15 + room 20.
Layer 3 remains the existing closed-15M trigger router (EMA / WT / Structure).

Risk plan:
  Initial SL = 1.0% (1R)
  T1 = +1.0R, trim 40%, move runner SL to breakeven
  TP2 = +2.0R
Entry-owner exit is inherited unchanged:
  EMA entry -> EMA reverse cross; WT entry -> WT opposite cross;
  Structure entry -> structure invalidation/opposite CHOCH.
"""
from __future__ import annotations

from typing import Optional
import numpy as np

from .base import SignalType
from .trend_confirm_wt_fixed_strategy import TrendConfirmWTFixedStrategy


class TrendConfirmV5Strategy(TrendConfirmWTFixedStrategy):
    VERSION = "5.0"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.t1_trigger_pct = 0.010  # 1R because live initial SL is 1.0%
        self.t1_trim_pct = 0.40
        self.t1_lock_pct = 0.0       # runner SL -> breakeven at T1

    @staticmethod
    def _pivot_levels(candles: list, span: int = 2):
        highs, lows = [], []
        for i in range(span, len(candles) - span):
            h = float(candles[i].high)
            l = float(candles[i].low)
            if h >= max(float(c.high) for c in candles[i-span:i]) and h >= max(float(c.high) for c in candles[i+1:i+span+1]):
                highs.append((i, h))
            if l <= min(float(c.low) for c in candles[i-span:i]) and l <= min(float(c.low) for c in candles[i+1:i+span+1]):
                lows.append((i, l))
        return highs, lows

    def _structure_bias(self, candles: list) -> str:
        highs, lows = self._pivot_levels(candles[-40:] if len(candles) > 40 else candles)
        if len(highs) >= 2 and len(lows) >= 2:
            hh = highs[-1][1] > highs[-2][1]
            hl = lows[-1][1] > lows[-2][1]
            lh = highs[-1][1] < highs[-2][1]
            ll = lows[-1][1] < lows[-2][1]
            if hh and hl:
                return "bull"
            if lh and ll:
                return "bear"
        return "mixed"

    def _macro_trend_4h(self, candles_4h: list) -> dict:
        """4H direction only. No entry authority and no MACD/HMA stacking."""
        lb = max(2, int(self.ema_slope_lookback))
        need = max(55, self.quality_ema_slow + lb + 3)
        if not candles_4h or len(candles_4h) < need:
            return {"state": "WARMUP", "direction": None, "score": None,
                    "bull_votes": 0, "bear_votes": 0, "signals": {}, "bars": len(candles_4h or [])}

        closes = [float(c.close) for c in candles_4h]
        e20 = self.ema(closes, 20)
        e50 = self.ema(closes, 50)
        vals = (e20[-1], e50[-1], e20[-1-lb])
        if any(np.isnan(v) for v in vals):
            return {"state": "WARMUP", "direction": None, "score": None,
                    "bull_votes": 0, "bear_votes": 0, "signals": {}, "bars": len(candles_4h)}

        close = closes[-1]
        structure = self._structure_bias(candles_4h)
        bull = 0.0
        bear = 0.0
        signals = {}

        if e20[-1] > e50[-1]:
            bull += 25; signals["ema20_50"] = 1
        elif e20[-1] < e50[-1]:
            bear += 25; signals["ema20_50"] = -1
        else:
            signals["ema20_50"] = 0

        if e20[-1] > e20[-1-lb]:
            bull += 25; signals["ema20_slope"] = 1
        elif e20[-1] < e20[-1-lb]:
            bear += 25; signals["ema20_slope"] = -1
        else:
            signals["ema20_slope"] = 0

        if close > e20[-1]:
            bull += 20; signals["price_location"] = 1
        elif close < e20[-1]:
            bear += 20; signals["price_location"] = -1
        else:
            signals["price_location"] = 0

        if structure == "bull":
            bull += 30; signals["structure"] = 1
        elif structure == "bear":
            bear += 30; signals["structure"] = -1
        else:
            # Mixed structure is intentionally neutral, not a veto.
            signals["structure"] = 0

        if bull >= 55 and bull > bear:
            direction = "long"
            state = "STRONG_BULL" if bull >= 70 else "BULL"
            score = bull
        elif bear >= 55 and bear > bull:
            direction = "short"
            state = "STRONG_BEAR" if bear >= 70 else "BEAR"
            score = bear
        else:
            direction = None
            state = "NEUTRAL"
            score = max(bull, bear)

        return {
            "state": state, "direction": direction, "score": round(float(score), 1),
            "bull_score": round(bull, 1), "bear_score": round(bear, 1),
            "bull_votes": sum(v > 0 for v in signals.values()),
            "bear_votes": sum(v < 0 for v in signals.values()),
            "signals": signals, "structure": structure,
            "ema20": round(float(e20[-1]), 8), "ema50": round(float(e50[-1]), 8),
            "ema20_slope": round(float(e20[-1] - e20[-1-lb]), 8),
            "close": round(float(close), 8), "bars": len(candles_4h),
            "layer_role": "DIRECTION_ONLY",
        }

    def _context_1h(self, candles_1h: list, direction: str) -> Optional[dict]:
        """1H quality score only. Avoid all-indicators-must-agree hard gating."""
        if direction not in ("long", "short"):
            return None
        need = max(55, 2 * self.adx_period + 3, self.chop_period + 3)
        if not candles_1h or len(candles_1h) < need:
            return None

        closes = [float(c.close) for c in candles_1h]
        e20 = self.ema(closes, 20)
        e50 = self.ema(closes, 50)
        adx_arr, plus_di, minus_di = self.adx(candles_1h, self.adx_period)
        chop = self._choppiness(candles_1h, self.chop_period)
        macd_line, macd_sig, macd_hist = self.macd(closes, self.macd_fast, self.macd_slow, self.macd_signal)
        atr_arr = self.atr(candles_1h, self.atr_period)
        vals = (e20[-1], e50[-1], adx_arr[-1], macd_line[-1], macd_sig[-1], atr_arr[-1])
        if chop is None or any(np.isnan(v) for v in vals):
            return None

        close = closes[-1]
        adx_val = float(adx_arr[-1])
        chop_val = float(chop)
        atr = max(float(atr_arr[-1]), 1e-12)
        structure = self._structure_bias(candles_1h)

        # ADX 0..25
        if adx_val < 15: adx_score = 0.0
        elif adx_val < 20: adx_score = 10.0
        elif adx_val < 25: adx_score = 18.0
        else: adx_score = 25.0

        # CHOP 0..20
        if chop_val < 45: chop_score = 20.0
        elif chop_val < 50: chop_score = 16.0
        elif chop_val < 55: chop_score = 10.0
        elif chop_val < 60: chop_score = 5.0
        else: chop_score = 0.0

        structure_aligned = structure == ("bull" if direction == "long" else "bear")
        structure_opposite = structure == ("bear" if direction == "long" else "bull")
        structure_score = 20.0 if structure_aligned else 0.0 if structure_opposite else 8.0

        hist = float(macd_hist[-1]) if len(macd_hist) and np.isfinite(macd_hist[-1]) else float(macd_line[-1] - macd_sig[-1])
        momentum_aligned = hist > 0 if direction == "long" else hist < 0
        momentum_opposite = hist < 0 if direction == "long" else hist > 0
        momentum_score = 15.0 if momentum_aligned else 0.0 if momentum_opposite else 5.0

        # Room to nearest confirmed opposing 1H pivot. No nearby opposing pivot = full room.
        highs, lows = self._pivot_levels(candles_1h[-50:] if len(candles_1h) > 50 else candles_1h)
        if direction == "long":
            opposing = [p for _i, p in highs if p > close]
            nearest = min(opposing) if opposing else None
        else:
            opposing = [p for _i, p in lows if p < close]
            nearest = max(opposing) if opposing else None
        risk_distance = max(close * 0.01, 0.5 * atr)
        room_r = ((abs(nearest - close) / risk_distance) if nearest is not None else 2.0)
        if room_r >= 1.5: room_score = 20.0
        elif room_r >= 1.2: room_score = 15.0
        elif room_r >= 1.0: room_score = 8.0
        else: room_score = 0.0

        score = adx_score + chop_score + structure_score + momentum_score + room_score
        hard_block = bool(
            chop_val >= 62.0
            or room_r < 1.0
            or (structure_opposite and momentum_opposite)
        )
        if score >= 70: label = "STRONG"
        elif score >= 55: label = "NORMAL"
        elif score >= 45: label = "WEAK"
        else: label = "BLOCK"
        ready = bool(score >= 55.0 and not hard_block)

        # Compatibility fields consumed by existing runner/viewlog.
        bias = direction if not structure_opposite else ("short" if direction == "long" else "long")
        return {
            "ready": ready, "score": round(score, 1), "label": label,
            "bias": bias, "bias_aligned": not structure_opposite,
            "bull_votes": int(e20[-1] > e50[-1]) + int(close > e20[-1]) + int(hist > 0),
            "bear_votes": int(e20[-1] < e50[-1]) + int(close < e20[-1]) + int(hist < 0),
            "votes": 0,
            "adx": round(adx_val, 1), "chop": round(chop_val, 1),
            "adx_ok": adx_val >= 15.0, "chop_ok": chop_val < 62.0,
            "ema20": round(float(e20[-1]), 8), "ema50": round(float(e50[-1]), 8),
            "structure": structure, "momentum_aligned": momentum_aligned,
            "room_r": round(float(room_r), 2), "nearest_opposing": nearest,
            "components": {
                "adx": adx_score, "chop": chop_score, "structure": structure_score,
                "momentum": momentum_score, "room": room_score,
            },
            "hard_block": hard_block,
            "layer_role": "QUALITY_ONLY",
        }

    async def analyze(self, candles: list, current_price: float, mtf_candles: dict = None):
        signal = await super().analyze(candles, current_price, mtf_candles)
        meta = signal.metadata if isinstance(getattr(signal, "metadata", None), dict) else {}
        meta["trend_confirm_version"] = self.VERSION
        meta["risk_plan"] = "1R_SL__T1_1R_TRIM40_BE__TP2_2R"
        meta["t1_r"] = 1.0
        meta["t1_trim_pct"] = 40.0
        meta["t1_lock"] = "BREAKEVEN"
        meta["tp2_r"] = 2.0

        if signal.type != SignalType.HOLD:
            entry = float(signal.price or current_price)
            direction = "long" if signal.type == SignalType.BUY else "short"
            risk = entry * 0.01
            sl = entry - risk if direction == "long" else entry + risk
            tp = entry + 2.0 * risk if direction == "long" else entry - 2.0 * risk
            self._entry_price = entry
            self._entry_sl = float(sl)
            meta.update({
                "stop_loss": round(sl, 8), "take_profit": round(tp, 8),
                "rr_ratio": 2.0, "sl_pct": 1.0, "tp_pct": 2.0,
                "t1_trigger_pct": 1.0, "t1_lock_pct": 0.0,
                "partial_tp_enabled": True, "partial_tp_pct": 40.0,
                "tp1_close_pct": 0.40,
            })
        signal.metadata = meta
        return signal

    def tick_open_position(self, current_price: float, position_key: Optional[str] = None):
        update = super().tick_open_position(current_price, position_key)
        if update is None:
            return None
        reason = str(getattr(update, "reason", "") or "")
        if getattr(update, "action", "") == "partial_tp":
            update.reason = "+1.0R T1 reached — take profit 40%, move runner SL to breakeven; remaining 60% targets 2.0R or entry-owner exit"
        elif getattr(update, "action", "") == "hold":
            trigger = self._active_entry_trigger or self.ENTRY_LEGACY
            owner_wait = {
                self.ENTRY_EMA: "EMA8/13 reverse cross",
                self.ENTRY_WT: "opposite WT1/WT2 cross",
                self.ENTRY_STRUCTURE: "structure invalidation/opposite CHOCH",
            }.get(trigger, "owner exit")
            update.reason = f"Holding {str(self._open_position).upper()} [{trigger}] — SL 1.0R / T1 1.0R trim40->BE / TP2 2.0R; waiting {owner_wait}"
        return update
