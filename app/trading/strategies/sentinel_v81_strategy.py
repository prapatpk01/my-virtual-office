"""Sentinel V8.1 — Quality Price Action Core.

Refines V8 after PAPER showed that responsiveness alone produced low-quality
re-entries and, on tight-stop markets such as XAU, transaction costs consumed a
large fraction of one nominal R.

Design:
- 15M bias remains fast, but EMA20 slope is now the anchor.  Direction requires
  slope plus at least one confirmation from price location or RSI momentum.
- 5M keeps the three V8 price-action triggers (pullback reclaim, micro breakout,
  sweep reclaim) but demands a quality close and sensible location.
- Post-cooldown strict mode is real: while RiskManager exposes an entry-threshold
  bonus, only A-grade 3/3 bias + aligned 5M slope/volume setups may enter.
- Fee-aware edge floor: a structure stop narrower than 0.40% of entry is skipped
  instead of pretending a tiny gross R is economical with market-order fees.
- One successful entry per closed 15M bar; after a hard SL, require three closed
  5M bars before the symbol can re-enter. Other exits keep a one-bar cooldown.
- SL/TP management remains TP1 +1R close 50% -> runner SL +0.15R, TP2 +2R.
"""
from __future__ import annotations

import numpy as np

from .base import SignalType
from .sentinel_v8_strategy import SentinelV8Strategy


class SentinelV81Strategy(SentinelV8Strategy):
    VERSION = "8.1"

    # Bias hysteresis / quality.
    BIAS_SLOPE_MIN_ATR = 0.05
    BIAS_PRICE_BUFFER_ATR = 0.03
    RSI_LONG_CONFIRM = 52.0
    RSI_SHORT_CONFIRM = 48.0

    # Market gate: moderate, not restrictive.
    ADX_FLOOR = 12.0
    CHOP_CEILING = 64.0
    ATR_ACTIVITY_FLOOR = 0.65

    # Entry freshness / economics.
    MAX_TRIGGER_CHASE_ATR = 0.30
    MIN_ECONOMIC_RISK_PCT = 0.0040  # 0.40%; keeps ~0.10% round-trip fee from dominating R

    # Structure stop remains local but avoids noise-sized stops.
    SL_BUFFER_ATR = 0.18
    MIN_SL_ATR = 0.90
    MAX_SL_ATR = 1.80

    HARD_SL_COOLDOWN_5M_BARS = 3
    NORMAL_COOLDOWN_5M_BARS = 1

    def __init__(self, symbol: str, **kwargs):
        super().__init__(symbol, **kwargs)
        self.name = f"SentinelV8.1({symbol})"
        self._last_entry_15m_ts: int | None = None
        self._bias_strength = 0
        self._last_close_was_hard_sl = False
        self.EXIT_COOLDOWN_5M_BARS = self.NORMAL_COOLDOWN_5M_BARS

    # ------------------------------------------------------------------
    # 15M direction: slope anchored, then price/RSI confirmation.
    # ------------------------------------------------------------------
    def _bias_15m(self, candles: list) -> dict:
        if len(candles) < self.MIN_15M_BARS:
            self._bias_strength = 0
            return {"ready": False, "direction": None, "reason": "15M warmup"}

        closes = [float(c.close) for c in candles]
        ema20 = self.ema(closes, 20)
        atr = self.atr(candles, 14)
        rsi = self.rsi(closes, 14)
        if not self._finite(ema20[-1], ema20[-4], atr[-1], rsi[-1]):
            self._bias_strength = 0
            return {"ready": False, "direction": None, "reason": "15M indicators unavailable"}

        close = float(closes[-1])
        e20 = float(ema20[-1])
        atr_now = max(float(atr[-1]), 1e-12)
        slope_atr = (e20 - float(ema20[-4])) / atr_now
        rsi_now = float(rsi[-1])
        buffer = self.BIAS_PRICE_BUFFER_ATR * atr_now

        slope_long = slope_atr >= self.BIAS_SLOPE_MIN_ATR
        slope_short = slope_atr <= -self.BIAS_SLOPE_MIN_ATR
        price_long = close >= e20 + buffer
        price_short = close <= e20 - buffer
        rsi_long = rsi_now >= self.RSI_LONG_CONFIRM
        rsi_short = rsi_now <= self.RSI_SHORT_CONFIRM

        if slope_long and (price_long or rsi_long):
            direction = "long"
            strength = 1 + int(price_long) + int(rsi_long)
        elif slope_short and (price_short or rsi_short):
            direction = "short"
            strength = 1 + int(price_short) + int(rsi_short)
        else:
            direction = None
            strength = 0

        self._bias_strength = strength
        return {
            "ready": direction is not None,
            "direction": direction,
            "strength": strength,
            "close": round(close, 8),
            "ema20": round(e20, 8),
            "ema20_slope_atr": round(float(slope_atr), 3),
            "rsi": round(rsi_now, 2),
            "slope_anchor": bool(slope_long or slope_short),
            "price_confirm": price_long if direction == "long" else price_short if direction == "short" else False,
            "rsi_confirm": rsi_long if direction == "long" else rsi_short if direction == "short" else False,
            "reason": f"15M {direction.upper()} slope-anchor bias {strength}/3" if direction else "15M bias neutral / slope not confirmed",
        }

    # ------------------------------------------------------------------
    # 5M setup quality overlay on top of V8's three PA triggers.
    # ------------------------------------------------------------------
    def _snapshot_5m(self, candles: list, direction: str | None, current_price: float) -> dict:
        out = super()._snapshot_5m(candles, direction, current_price)
        if not out.get("ready") or direction not in {"long", "short"}:
            return out

        # V8 may already have blocked the setup (market gate, chase, SL width).
        trigger = out.get("trigger")
        if not trigger:
            return out

        closes = [float(c.close) for c in candles]
        ema20 = self.ema(closes, 20)
        atr5 = max(float(out.get("atr5") or 0.0), 1e-12)
        bar = candles[-1]
        high = float(bar.high)
        low = float(bar.low)
        close = float(bar.close)
        open_ = float(bar.open)
        rng = max(high - low, 1e-12)
        body_atr = abs(close - open_) / atr5
        close_pos = (close - low) / rng  # 1.0=close at high, 0.0=close at low
        ema_slope_atr = (float(ema20[-1]) - float(ema20[-4])) / atr5 if self._finite(ema20[-1], ema20[-4]) else 0.0
        dist_ema_atr = abs(close - float(ema20[-1])) / atr5

        vols = [float(c.volume or 0.0) for c in candles[-21:-1]]
        med_vol = float(np.median(vols)) if vols else 0.0
        curr_vol = float(bar.volume or 0.0)
        vol_ratio = curr_vol / med_vol if med_vol > 0 else 1.0

        long = direction == "long"
        close_quality = close_pos >= 0.62 if long else close_pos <= 0.38
        slope5_aligned = ema_slope_atr > 0.0 if long else ema_slope_atr < 0.0

        if trigger == "MICRO_BREAKOUT":
            min_body = 0.25
            max_dist = 1.25
            volume_required = True
            slope_required = True
        elif trigger == "PULLBACK_RECLAIM":
            min_body = 0.18
            max_dist = 1.00
            volume_required = False
            slope_required = False
        else:  # SWEEP_RECLAIM
            min_body = 0.12
            max_dist = 1.20
            volume_required = False
            slope_required = False

        blocks = list(out.get("blocks", []))
        if not close_quality:
            blocks.append("WEAK_CLOSE")
        if body_atr < min_body:
            blocks.append("WEAK_BODY")
        if dist_ema_atr > max_dist:
            blocks.append("EXTENDED_FROM_EMA20")
        if volume_required and vol_ratio < 0.80:
            blocks.append("BREAKOUT_VOLUME")
        if slope_required and not slope5_aligned:
            blocks.append("5M_SLOPE")

        # Natural structure risk must be large enough relative to trading cost.
        # We SKIP tiny-risk setups instead of widening the stop outside the setup
        # just to manufacture a larger R denominator.
        raw_sl_atr = float(out.get("raw_sl_atr") or 0.0)
        raw_risk_pct = (raw_sl_atr * atr5 / max(float(out.get("entry") or current_price), 1e-12)) if raw_sl_atr > 0 else 0.0
        if raw_risk_pct < self.MIN_ECONOMIC_RISK_PCT:
            blocks.append("FEE_EDGE_TOO_TIGHT")

        strict_mode = float(getattr(self, "_entry_threshold_bonus", 0.0) or 0.0) > 0.0
        if strict_mode:
            if self._bias_strength < 3:
                blocks.append("STRICT_BIAS_3OF3")
            if not slope5_aligned:
                blocks.append("STRICT_5M_SLOPE")
            if vol_ratio < 1.0:
                blocks.append("STRICT_VOLUME")

        out.update({
            "body_atr": round(float(body_atr), 2),
            "close_pos": round(float(close_pos), 2),
            "ema20_slope_atr": round(float(ema_slope_atr), 3),
            "dist_ema_atr": round(float(dist_ema_atr), 2),
            "volume_ratio": round(float(vol_ratio), 2),
            "raw_risk_pct": round(float(raw_risk_pct * 100.0), 3),
            "min_economic_risk_pct": round(self.MIN_ECONOMIC_RISK_PCT * 100.0, 3),
            "strict_mode": bool(strict_mode),
            "quality_close": bool(close_quality),
            "slope5_aligned": bool(slope5_aligned),
        })

        if blocks:
            out["trigger_candidate"] = trigger
            out["trigger"] = None
            out["blocks"] = list(dict.fromkeys(blocks))
            out["reason"] = "5M trigger rejected by V8.1 quality/edge rules"
        return out

    # ------------------------------------------------------------------
    # Lifecycle: stop re-entry churn and sync actual fills.
    # ------------------------------------------------------------------
    async def analyze(self, candles: list, current_price: float, mtf_candles: dict = None):
        signal = await super().analyze(candles, current_price, mtf_candles=mtf_candles)
        meta = signal.metadata or {}
        meta["strategy"] = "SENTINEL_V8_1"
        meta["version"] = self.VERSION
        meta["architecture"] = "15M_SLOPE_ANCHORED_BIAS__5M_QUALITY_MULTI_PA"
        meta["fee_edge_floor_pct"] = self.MIN_ECONOMIC_RISK_PCT * 100.0
        signal.metadata = meta

        sig_value = getattr(getattr(signal, "type", None), "value", "hold")
        if sig_value == "hold":
            return signal

        # A successful fill sets _last_entry_15m_ts via attach_existing_position.
        # If the same 15M context already produced a filled trade, do not churn
        # another entry from a different 5M trigger inside that same 15M bar.
        current_15m_ts = int(self._bar_ts(self._latest_15m[-1])) if self._latest_15m else None
        if current_15m_ts is not None and self._last_entry_15m_ts == current_15m_ts:
            self.cancel_pending_entry("one filled entry per 15M bar")
            meta["setup_5m"] = dict(meta.get("setup_5m") or {})
            meta["setup_5m"]["blocks"] = ["ONE_ENTRY_PER_15M"]
            return self._hold(float(current_price), "one successful Sentinel entry already used this 15M bar", meta)

        return signal

    def attach_existing_position(self, direction: str, entry_price: float,
                                 stop_loss: float | None = None,
                                 take_profit: float | None = None) -> None:
        super().attach_existing_position(direction, entry_price, stop_loss, take_profit)
        if self._latest_15m:
            self._last_entry_15m_ts = int(self._bar_ts(self._latest_15m[-1]))
        self.EXIT_COOLDOWN_5M_BARS = self.NORMAL_COOLDOWN_5M_BARS
        self._last_close_was_hard_sl = False

    def record_closed_trade(self, exit_price: float, reason: str, duration_min: float = 0.0) -> None:
        r = str(reason or "").lower()
        hard_sl = "stop_loss" in r or "hard_sl" in r or "stop loss" in r
        self._last_close_was_hard_sl = hard_sl
        self.EXIT_COOLDOWN_5M_BARS = (
            self.HARD_SL_COOLDOWN_5M_BARS if hard_sl else self.NORMAL_COOLDOWN_5M_BARS
        )
        super().record_closed_trade(exit_price, reason, duration_min)
