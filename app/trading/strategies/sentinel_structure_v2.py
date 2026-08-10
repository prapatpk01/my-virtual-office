"""Sentinel V2 architecture overlay.

Adds a second, continuation-style entry engine to the existing Sentinel S/R
reversal engine without deleting the proven S/R path.

Architecture
------------
ENGINE A (existing): 1H hybrid S/R -> dynamic proximity -> 15M rejection /
reclaim / displacement -> MCDX confirmation.

ENGINE B (this overlay): Sentinel X v2.3-style confirmed impulse -> Fib
38.2-61.8 pullback location -> 15M rejection / reclaim / displacement -> MCDX
relative dominance -> 1H S/R risk map.

Probabilistic Target Engine is a SOFT factor only. It never hard-blocks an
otherwise valid entry. It contributes confidence / warning metadata and a
projected runner reference when the opposing 1H S/R side is open.

Source translation notes
------------------------
- Fib zone follows Sentinel X v2.3: valid confirmed impulse, leg >= 2 ATR,
  38.2/50/61.8 levels, with +/-0.25 ATR tolerance around the zone.
- Forecast uses the same factor families and weights as the Pine target engine
  (direction, energy, fast impulse, structure, HTF alignment, room). The Python
  implementation is intentionally a closed-bar approximation because the bot
  has 15M/1H/4H data rather than TradingView's chart-state series.
- MCDX remains the production Sentinel MCDX context: L/S >=45, dominance >10,
  flow >52 for long / <48 for short.
- Fib engine SL uses the nearest structural 1H S/R anchor plus 0.25 ATR1H.
- TP2 remains opposing 1H S/R when available. If unavailable, the existing
  OPEN_SKY/OPEN_FLOOR runner remains active; forecast target is only a soft
  reference.
"""
from __future__ import annotations

from typing import Optional
import numpy as np

from .base import Signal, SignalType


FIB_MIN_LEG_ATR = 2.0
FIB_TOLERANCE_ATR = 0.25
FIB_SL_BUFFER_ATR1H = 0.25
FORECAST_VISIBLE_CONF = 58.0
FORECAST_STRONG_CONF = 65.0
FORECAST_OPPOSITE_WARNING = 70.0


def install_sentinel_structure_v2(strategy_cls) -> None:
    """Install Sentinel V2 Fib-pullback + forecast overlay once."""
    if getattr(strategy_cls, "_sentinel_structure_v2_installed", False):
        return

    original_analyze = strategy_cls.analyze
    strategy_cls._sentinel_structure_v2_installed = True
    strategy_cls.VERSION = "2.0"
    strategy_cls.fib_min_leg_atr = FIB_MIN_LEG_ATR
    strategy_cls.fib_tolerance_atr = FIB_TOLERANCE_ATR
    strategy_cls.fib_sl_buffer_atr_1h = FIB_SL_BUFFER_ATR1H

    @staticmethod
    def _pivot_points(candles: list, span: int = 4) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
        highs: list[tuple[int, float]] = []
        lows: list[tuple[int, float]] = []
        if len(candles) < span * 2 + 3:
            return highs, lows
        for i in range(span, len(candles) - span):
            hi = float(candles[i].high)
            lo = float(candles[i].low)
            if hi >= max(float(c.high) for c in candles[i-span:i]) and hi >= max(float(c.high) for c in candles[i+1:i+span+1]):
                highs.append((i, hi))
            if lo <= min(float(c.low) for c in candles[i-span:i]) and lo <= min(float(c.low) for c in candles[i+1:i+span+1]):
                lows.append((i, lo))
        return highs, lows

    def _htf_alignment_count(self, mtf: dict) -> tuple[int, int]:
        """Return bull/bear alignment count using bot-available completed 1H/4H."""
        bull = bear = 0
        for tf in ("1h", "4h"):
            cs = list((mtf or {}).get(tf) or [])
            if len(cs) < 55:
                continue
            closes = [float(c.close) for c in cs]
            e20 = self.ema(closes, 20)
            e50 = self.ema(closes, 50)
            if not (np.isfinite(e20[-1]) and np.isfinite(e50[-1])):
                continue
            if closes[-1] > e20[-1] > e50[-1]:
                bull += 1
            if closes[-1] < e20[-1] < e50[-1]:
                bear += 1
        return bull, bear

    def _sentinel_x_snapshot(self, candles: list, mtf: dict, sr: dict) -> dict:
        """Closed-bar approximation of Sentinel X v2.3 SME/target factors."""
        if len(candles) < 80:
            return {"ready": False}

        closes = np.asarray([float(c.close) for c in candles], dtype=float)
        opens = np.asarray([float(c.open) for c in candles], dtype=float)
        highs = np.asarray([float(c.high) for c in candles], dtype=float)
        lows = np.asarray([float(c.low) for c in candles], dtype=float)
        vols = np.asarray([max(0.0, float(c.volume)) for c in candles], dtype=float)
        atr_arr = self.atr(candles, 14)
        atr = float(atr_arr[-1]) if len(atr_arr) and np.isfinite(atr_arr[-1]) else 0.0
        if atr <= 0:
            return {"ready": False}

        den = max(atr, 1e-12)
        e20 = self.ema(list(closes), 20)
        e50 = self.ema(list(closes), 50)
        h16 = self.hma(list(closes), 16)
        rsi = self.rsi(list(closes), 14)
        _, _, hist = self.macd(list(closes), 12, 26, 9)
        adx_arr, pdi, mdi = self.adx(candles, 14)

        def f(v, default=0.0):
            try:
                x = float(v)
                return x if np.isfinite(x) else float(default)
            except Exception:
                return float(default)

        hma16_sl = (f(h16[-1])-f(h16[-4]))/den
        ema20_sl = (f(e20[-1])-f(e20[-5]))/den
        ema50_sl = (f(e50[-1])-f(e50[-9]))/den
        sep = (f(e20[-1])-f(e50[-1]))/den
        trend_raw = self._clamp(
            hma16_sl*10.0 + ema20_sl*23.0 + ema50_sl*31.0 + sep*16.0 + (closes[-1]-f(e20[-1], closes[-1]))/den*8.0,
            -100.0, 100.0,
        )
        trend_dir = float(trend_raw)

        rng = max(highs[-1]-lows[-1], 1e-12)
        body_eff = abs(closes[-1]-opens[-1])/rng
        clv = (closes[-1]-lows[-1])/rng
        vol_ma = float(np.mean(vols[-20:])) if len(vols) >= 20 else max(vols[-1], 1.0)
        rvol = vols[-1]/max(vol_ma, 1e-12)

        dip = f(pdi[-1])
        dim = f(mdi[-1])
        di_den = max(dip+dim, 1.0)
        di_bias = (dip-dim)/di_den*100.0
        pa_pressure = self._clamp((1.0 if closes[-1] > opens[-1] else -1.0)*body_eff*60.0 + (clv-0.5)*65.0, -100.0, 100.0)
        flow_pressure = self._clamp((1.0 if closes[-1] > opens[-1] else -1.0)*((rvol-0.75)*48.0 + body_eff*42.0), -100.0, 100.0)

        direction = self._clamp(trend_dir*0.36 + di_bias*0.24 + pa_pressure*0.20 + flow_pressure*0.20, -100.0, 100.0)

        roc1_n = (closes[-1]-closes[-2])/den*35.0
        roc3_n = (closes[-1]-closes[-4])/den*18.0
        prev_roc1 = (closes[-3]-closes[-4])/max(f(atr_arr[-3], atr), 1e-12)*35.0 if len(closes) >= 4 else 0.0
        price_accel = (roc1_n-prev_roc1)*1.40
        rsi_vel = (f(rsi[-1], 50.0)-f(rsi[-3], 50.0))*2.20
        hist_accel = (f(hist[-1])-f(hist[-3]))/den*120.0
        pdi_prev = f(pdi[-3])
        mdi_prev = f(mdi[-3])
        prev_di_den = max(pdi_prev+mdi_prev, 1.0)
        prev_di_bias = (pdi_prev-mdi_prev)/prev_di_den*100.0
        di_accel = (di_bias-prev_di_bias)*0.95
        displacement = self._clamp((closes[-1]-opens[-1])/den*42.0, -100.0, 100.0)
        fast_impulse = self._clamp(
            roc1_n*0.18 + roc3_n*0.12 + price_accel*0.18 + rsi_vel*0.12 + hist_accel*0.18 + di_accel*0.10 + displacement*0.12,
            -100.0, 100.0,
        )

        adx = f(adx_arr[-1])
        trend_coherence = min(100.0, abs(sep)*18.0 + abs(ema20_sl)*22.0 + adx*1.15)
        move_eff = min(100.0, body_eff*50.0 + max(0.0, rvol-0.7)*40.0 + (12.0 if rvol >= 1.20 else 0.0))
        energy = self._clamp(abs(direction)*0.34 + abs(fast_impulse)*0.24 + trend_coherence*0.20 + move_eff*0.22, 0.0, 100.0)

        structure = self._structure(candles)
        structure_state = 1 if structure == "BULL" else -1 if structure == "BEAR" else 0
        htf_bull, htf_bear = _htf_alignment_count(self, mtf)
        s1 = sr.get("s1")
        r1 = sr.get("r1")
        room_long = 9.0 if r1 is None else (float(r1)-float(closes[-1]))/den
        room_short = 9.0 if s1 is None else (float(closes[-1])-float(s1))/den

        bull_conf = self._clamp(
            50.0 + direction*0.20 + energy*0.16 + max(0.0, fast_impulse)*0.10
            + (8.0 if structure_state > 0 else 0.0)
            + (8.0 if htf_bull >= 2 else 0.0)
            - (12.0 if room_long < 0.8 else 0.0),
            5.0, 95.0,
        )
        bear_conf = self._clamp(
            50.0 + abs(min(direction, 0.0))*0.20 + energy*0.16 + abs(min(fast_impulse, 0.0))*0.10
            + (8.0 if structure_state < 0 else 0.0)
            + (8.0 if htf_bear >= 2 else 0.0)
            - (12.0 if room_short < 0.8 else 0.0),
            5.0, 95.0,
        )
        bull_target = float(r1) if r1 is not None else float(closes[-1] + atr*2.2)
        bear_target = float(s1) if s1 is not None else float(closes[-1] - atr*2.2)

        return {
            "ready": True,
            "atr_15m": atr,
            "trend_dir": round(trend_dir, 1),
            "direction": round(direction, 1),
            "energy": round(energy, 1),
            "fast_impulse": round(fast_impulse, 1),
            "structure": structure,
            "structure_state": structure_state,
            "htf_bull": htf_bull,
            "htf_bear": htf_bear,
            "room_long": round(room_long, 2),
            "room_short": round(room_short, 2),
            "bull_conf": round(bull_conf, 1),
            "bear_conf": round(bear_conf, 1),
            "bull_target": round(bull_target, 8),
            "bear_target": round(bear_target, 8),
            "forecast_side": "BULL" if bull_conf >= bear_conf else "BEAR",
            "forecast_conf": round(max(bull_conf, bear_conf), 1),
        }

    def _fib_pullback_context(self, candles: list, sxv: dict) -> dict:
        """Sentinel X v2.3 Fib 38.2/50/61.8 location from confirmed 15M impulse."""
        atr = float(sxv.get("atr_15m", 0.0) or 0.0)
        if len(candles) < 80 or atr <= 0:
            return {"ready": False, "reason": "Fib warmup/ATR"}

        highs, lows = _pivot_points(candles[-160:], 4)
        if not highs or not lows:
            return {"ready": False, "reason": "No confirmed Fib pivots"}
        _, last_ph = highs[-1]
        _, last_pl = lows[-1]
        if last_ph <= last_pl:
            return {"ready": False, "reason": "Invalid impulse leg"}

        structure_state = int(sxv.get("structure_state", 0) or 0)
        trend_dir = float(sxv.get("trend_dir", 0.0) or 0.0)
        close = float(candles[-1].close)

        # Simple BOS allowance mirrors Pine's ability to keep Fib alive when a
        # fresh break supports the leg even before the slower structure state.
        prev_high = highs[-2][1] if len(highs) >= 2 else last_ph
        prev_low = lows[-2][1] if len(lows) >= 2 else last_pl
        bos_up = close > float(prev_high)
        bos_dn = close < float(prev_low)
        valid_bull = (structure_state > 0 or bos_up) and trend_dir >= 0
        valid_bear = (structure_state < 0 or bos_dn) and trend_dir < 0

        leg = float(last_ph-last_pl)
        leg_atr = leg/atr
        active = leg_atr >= FIB_MIN_LEG_ATR and (valid_bull or valid_bear)
        if not active:
            return {
                "ready": False,
                "reason": f"Fib leg inactive ({leg_atr:.2f}ATR)",
                "leg_atr": round(leg_atr, 2),
            }

        if valid_bull:
            fib38 = last_ph-leg*0.382
            fib50 = last_ph-leg*0.500
            fib62 = last_ph-leg*0.618
            side = "BULL"
            zone_low = fib62-atr*FIB_TOLERANCE_ATR
            zone_high = fib38+atr*FIB_TOLERANCE_ATR
        else:
            fib38 = last_pl+leg*0.382
            fib50 = last_pl+leg*0.500
            fib62 = last_pl+leg*0.618
            side = "BEAR"
            zone_low = fib38-atr*FIB_TOLERANCE_ATR
            zone_high = fib62+atr*FIB_TOLERANCE_ATR

        bar = candles[-1]
        prev = candles[-2]
        op = float(bar.open)
        hi = float(bar.high)
        lo = float(bar.low)
        cl = float(bar.close)
        prev_cl = float(prev.close)
        prev_hi = float(prev.high)
        prev_lo = float(prev.low)
        body = abs(cl-op)
        touched = hi >= zone_low and lo <= zone_high
        prev_touched = prev_hi >= zone_low and prev_lo <= zone_high

        long_rejection = side == "BULL" and touched and cl > op and cl >= fib50
        long_reclaim = side == "BULL" and lo <= fib50 and cl > fib50 and cl > op
        long_displacement = side == "BULL" and prev_touched and cl > prev_hi and cl > op and body >= 0.20*atr
        short_rejection = side == "BEAR" and touched and cl < op and cl <= fib50
        short_reclaim = side == "BEAR" and hi >= fib50 and cl < fib50 and cl < op
        short_displacement = side == "BEAR" and prev_touched and cl < prev_lo and cl < op and body >= 0.20*atr

        long_trigger = bool(long_rejection or long_reclaim or long_displacement)
        short_trigger = bool(short_rejection or short_reclaim or short_displacement)
        long_name = "FIB_REJECTION" if long_rejection else "FIB_RECLAIM" if long_reclaim else "FIB_DISPLACEMENT" if long_displacement else ""
        short_name = "FIB_REJECTION" if short_rejection else "FIB_RECLAIM" if short_reclaim else "FIB_DISPLACEMENT" if short_displacement else ""

        return {
            "ready": True,
            "side": side,
            "leg_low": round(float(last_pl), 8),
            "leg_high": round(float(last_ph), 8),
            "leg_atr": round(leg_atr, 2),
            "fib38": round(float(fib38), 8),
            "fib50": round(float(fib50), 8),
            "fib62": round(float(fib62), 8),
            "zone_low": round(float(zone_low), 8),
            "zone_high": round(float(zone_high), 8),
            "in_zone": bool(zone_low <= cl <= zone_high or touched),
            "long_trigger": long_trigger,
            "short_trigger": short_trigger,
            "long_trigger_name": long_name,
            "short_trigger_name": short_name,
        }

    def _forecast_quality(self, direction: str, sxv: dict) -> tuple[str, float, float, Optional[float]]:
        bull = float(sxv.get("bull_conf", 50.0) or 50.0)
        bear = float(sxv.get("bear_conf", 50.0) or 50.0)
        if direction == "long":
            aligned, opposite = bull, bear
            target = sxv.get("bull_target")
        else:
            aligned, opposite = bear, bull
            target = sxv.get("bear_target")

        if aligned >= FORECAST_STRONG_CONF and aligned >= opposite:
            label = "STRONG_SUPPORT"
            adjustment = 0.10
        elif aligned >= FORECAST_VISIBLE_CONF and aligned >= opposite:
            label = "SUPPORT"
            adjustment = 0.05
        elif opposite >= FORECAST_OPPOSITE_WARNING and opposite > aligned:
            label = "STRONG_WARNING"
            adjustment = -0.10
        elif opposite >= FORECAST_VISIBLE_CONF and opposite > aligned:
            label = "WARNING"
            adjustment = -0.05
        else:
            label = "NEUTRAL"
            adjustment = 0.0
        return label, aligned, adjustment, float(target) if target is not None else None

    async def analyze(self, candles: list, current_price: float, mtf_candles: dict = None) -> Signal:
        # Engine A remains intact and gets first right of refusal.
        base_signal = await original_analyze(self, candles, current_price, mtf_candles)
        if base_signal.type != SignalType.HOLD:
            md = dict(base_signal.metadata or {})
            md["architecture"] = "SENTINEL_V2_DUAL_ENGINE"
            md["entry_engine"] = "SR_REVERSAL"
            base_signal.metadata = md
            return base_signal

        # Never synthesize a second entry while Sentinel is already managing one.
        if getattr(self, "_open_position", None) is not None or len(candles) < 120:
            return base_signal

        mtf = mtf_candles or {}
        close = float(candles[-1].close)
        sr = getattr(self, "_latest_sr", None) or self._sr_map_1h(mtf, close)
        if not sr.get("ready"):
            return base_signal
        mc = getattr(self, "_latest_mc", None) or self._mcdx_context(candles, mtf)
        sx = getattr(self, "_latest_sx", None) or self._sentinel_context(candles, mtf)

        sxv = _sentinel_x_snapshot(self, candles, mtf, sr)
        fib = _fib_pullback_context(self, candles, sxv) if sxv.get("ready") else {"ready": False, "reason": "SX forecast unavailable"}

        md = dict(base_signal.metadata or {})
        md.update({
            "architecture": "SENTINEL_V2_DUAL_ENGINE",
            "entry_engine_a": "SR_REVERSAL",
            "entry_engine_b": "FIB_PULLBACK",
            "fib": fib,
            "probabilistic_target": sxv,
        })

        if not fib.get("ready"):
            base_signal.metadata = md
            base_signal.reason = f"{base_signal.reason} | FIB2: {fib.get('reason', 'inactive')}"
            return base_signal

        long_score = float(mc.get("long_score", 0.0) or 0.0)
        short_score = float(mc.get("short_score", 0.0) or 0.0)
        flow = float(mc.get("smart_flow", 50.0) or 50.0)
        gap_long = long_score-short_score
        gap_short = short_score-long_score
        mcdx_long = bool(long_score >= self.min_context_score and gap_long > self.mcdx_dominance_gap and flow > self.long_flow_min)
        mcdx_short = bool(short_score >= self.min_context_score and gap_short > self.mcdx_dominance_gap and flow < self.short_flow_max)

        # Sentinel X is a veto only when both bias AND structure strongly oppose
        # the Fib direction. This avoids the old double-confirmation bottleneck.
        long_veto = bool(sx.get("bias") == "BEAR" and sx.get("structure") == "BEAR")
        short_veto = bool(sx.get("bias") == "BULL" and sx.get("structure") == "BULL")

        s1 = sr.get("s1")
        r1 = sr.get("r1")
        atr1h = float(sr.get("atr_1h", 0.0) or 0.0)
        min_rr = float(getattr(self, "min_rr", 1.50))

        # ------------------------------ LONG Fib engine
        if fib.get("side") == "BULL" and fib.get("long_trigger") and mcdx_long and not long_veto and s1 is not None and atr1h > 0:
            sl = float(s1)-FIB_SL_BUFFER_ATR1H*atr1h
            risk = close-sl
            if risk > 0:
                target = float(r1) if r1 is not None and float(r1) > close else None
                rr = ((target-close)/risk) if target is not None else None
                if rr is None or rr >= min_rr:
                    fq, fconf, fadj, ftarget = _forecast_quality(self, "long", sxv)
                    self._open_position = "long"
                    self._entry_price = close
                    self._entry_sl = sl
                    self._open_ended = target is None
                    md.update({
                        "entry_engine": "FIB_PULLBACK",
                        "entry_location": "15M_FIB_38_62",
                        "entry_trigger": fib.get("long_trigger_name"),
                        "stop_loss": round(sl, 8),
                        "stop_basis": "1H_S1_MINUS_0.25ATR1H",
                        "take_profit": target,
                        "tp_basis": "1H_R1" if target is not None else "OPEN_SKY_RUNNER",
                        "open_ended_tp": target is None,
                        "rr_ratio": round(rr, 2) if rr is not None else None,
                        "forecast_quality": fq,
                        "forecast_aligned_conf": round(fconf, 1),
                        "forecast_soft_target": round(ftarget, 8) if ftarget is not None else None,
                    })
                    conf = self._clamp(0.55 + long_score/250.0 + fadj, 0.0, 1.0)
                    return Signal(
                        SignalType.BUY, self.symbol, current_price, 0.0,
                        f"SENTINEL V2 LONG FIB38-62 {fib.get('long_trigger_name')} | leg={fib.get('leg_atr')}ATR | "
                        f"MCDX L={long_score:.1f} S={short_score:.1f} Δ={gap_long:.1f} flow={flow:.1f} | "
                        f"SL=S1-0.25ATR1H | TP2={'R1' if target is not None else 'RUNNER'} | Forecast {fq} {fconf:.0f}%",
                        confidence=conf,
                        metadata=md,
                    )

        # ------------------------------ SHORT Fib engine
        if fib.get("side") == "BEAR" and fib.get("short_trigger") and mcdx_short and not short_veto and r1 is not None and atr1h > 0:
            sl = float(r1)+FIB_SL_BUFFER_ATR1H*atr1h
            risk = sl-close
            if risk > 0:
                target = float(s1) if s1 is not None and float(s1) < close else None
                rr = ((close-target)/risk) if target is not None else None
                if rr is None or rr >= min_rr:
                    fq, fconf, fadj, ftarget = _forecast_quality(self, "short", sxv)
                    self._open_position = "short"
                    self._entry_price = close
                    self._entry_sl = sl
                    self._open_ended = target is None
                    md.update({
                        "entry_engine": "FIB_PULLBACK",
                        "entry_location": "15M_FIB_38_62",
                        "entry_trigger": fib.get("short_trigger_name"),
                        "stop_loss": round(sl, 8),
                        "stop_basis": "1H_R1_PLUS_0.25ATR1H",
                        "take_profit": target,
                        "tp_basis": "1H_S1" if target is not None else "OPEN_FLOOR_RUNNER",
                        "open_ended_tp": target is None,
                        "rr_ratio": round(rr, 2) if rr is not None else None,
                        "forecast_quality": fq,
                        "forecast_aligned_conf": round(fconf, 1),
                        "forecast_soft_target": round(ftarget, 8) if ftarget is not None else None,
                    })
                    conf = self._clamp(0.55 + short_score/250.0 + fadj, 0.0, 1.0)
                    return Signal(
                        SignalType.SELL, self.symbol, current_price, 0.0,
                        f"SENTINEL V2 SHORT FIB38-62 {fib.get('short_trigger_name')} | leg={fib.get('leg_atr')}ATR | "
                        f"MCDX S={short_score:.1f} L={long_score:.1f} Δ={gap_short:.1f} flow={flow:.1f} | "
                        f"SL=R1+0.25ATR1H | TP2={'S1' if target is not None else 'RUNNER'} | Forecast {fq} {fconf:.0f}%",
                        confidence=conf,
                        metadata=md,
                    )

        # Detailed HOLD trace so Railway shows exactly which Engine B gate waits.
        blockers = []
        if fib.get("side") == "BULL":
            if not fib.get("in_zone"):
                blockers.append("price outside Fib38-62")
            if not fib.get("long_trigger"):
                blockers.append("wait 15M bull trigger")
            if not mcdx_long:
                blockers.append(f"MCDX long wait L={long_score:.0f} S={short_score:.0f} Δ={gap_long:.0f} flow={flow:.1f}")
            if long_veto:
                blockers.append("SentinelX strong BEAR veto")
            if s1 is None:
                blockers.append("need 1H S1 for SL")
            fq, fconf, _, ftarget = _forecast_quality(self, "long", sxv)
        else:
            if not fib.get("in_zone"):
                blockers.append("price outside Fib38-62")
            if not fib.get("short_trigger"):
                blockers.append("wait 15M bear trigger")
            if not mcdx_short:
                blockers.append(f"MCDX short wait S={short_score:.0f} L={long_score:.0f} Δ={gap_short:.0f} flow={flow:.1f}")
            if short_veto:
                blockers.append("SentinelX strong BULL veto")
            if r1 is None:
                blockers.append("need 1H R1 for SL")
            fq, fconf, _, ftarget = _forecast_quality(self, "short", sxv)

        md.update({
            "fib_engine_blockers": blockers,
            "forecast_quality": fq,
            "forecast_aligned_conf": round(fconf, 1),
            "forecast_soft_target": round(ftarget, 8) if ftarget is not None else None,
        })
        base_signal.metadata = md
        zone_txt = f"{fib.get('zone_low')}..{fib.get('zone_high')}"
        base_signal.reason = (
            f"{base_signal.reason} | FIB2 {fib.get('side')} zone={zone_txt} leg={fib.get('leg_atr')}ATR | "
            f"{' ; '.join(blockers) if blockers else 'ready'} | Forecast {fq} {fconf:.0f}%"
        )
        return base_signal

    strategy_cls.analyze = analyze
    strategy_cls._sentinel_v2_pivot_points = _pivot_points
    strategy_cls._sentinel_v2_snapshot = _sentinel_x_snapshot
    strategy_cls._sentinel_v2_fib = _fib_pullback_context
