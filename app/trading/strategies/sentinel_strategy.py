"""Sentinel V9 Clean Core.

Single-purpose architecture:
15M = analysis/setup/score/location -> 5M = execution trigger.
No 1H/4H direction, no MCDX, no legacy overlays, no V10 engines.

15M evidence:
- EMA20/50 trend + EMA200 macro bonus
- EMA slopes + HMA16
- RSI14 vs SMA14
- MACD 12/26/9
- ADX/DMI + CHOP + ATR activity
- confirmed swing structure / BOS / CHoCH / liquidity sweep
- S/R + Fib 38.2/50/61.8 location

Setup families and minimum score:
PB 6.0 | LQ 6.0 | BO_RETEST 6.5 | BO_DIRECT 7.0 | REV 7.5

5M execution:
PULLBACK_RECLAIM | MICRO_BREAKOUT | SWEEP_RECLAIM
with ADX/CHOP/ATR gate, candle quality and anti-chase.

Risk:
structure SL + 0.18 ATR, bounded 0.90..1.80 ATR;
TP1 +1.0R closes 50%, runner stop +0.15R; TP2 1.5..2.5R, fallback 2R.

Only closed candles are used for decisions.
"""
from __future__ import annotations

import math
import time
from typing import Optional

import numpy as np

from .base import BaseStrategy, Signal, SignalType
from ..engines.position_manager import PositionUpdate


class SentinelStrategy(BaseStrategy):
    VERSION = "9.1-CLEAN"
    entry_tf = "15m"

    # Market gate
    ADX_MIN = 12.0
    CHOP_MAX = 64.0
    ATR_ACTIVITY_MIN = 0.65

    # Structure
    PIVOT_SPAN = 2
    BOS_BUFFER_ATR = 0.05
    SWEEP_BUFFER_ATR = 0.05

    # Setup scores
    SCORE_MIN = {
        "PB": 6.0,
        "LQ": 6.0,
        "BO_RETEST": 6.5,
        "BO_DIRECT": 7.0,
        "REV": 7.5,
    }

    # Execution
    MAX_CHASE_ATR = 0.30
    SL_BUFFER_ATR = 0.18
    MIN_SL_ATR = 0.90
    MAX_SL_ATR = 1.80
    MIN_ECONOMIC_RISK_PCT = 0.0040

    # Position management
    TP1_R = 1.0
    TP1_CLOSE_PCT = 0.50
    TP1_LOCK_R = 0.15
    TP2_MIN_R = 1.50
    TP2_MAX_R = 2.50
    TP2_FALLBACK_R = 2.00

    HARD_SL_COOLDOWN_5M = 3
    NORMAL_COOLDOWN_5M = 1

    SETUP_PRIORITY = ("PB", "LQ", "REV", "BO_RETEST", "BO_DIRECT")

    def __init__(self, symbol: str, params: Optional[dict] = None, **kwargs):
        super().__init__(symbol, params)
        self.name = f"SentinelV9({symbol})"
        self._entry_threshold_bonus = 0.0
        self._open_position: Optional[str] = None
        self._entry_price: Optional[float] = None
        self._entry_sl: Optional[float] = None
        self._entry_tp: Optional[float] = None
        self._initial_risk: Optional[float] = None
        self._tp1_done = False
        self._pending_entry = False
        self._last_entry_15m_ts: Optional[int] = None
        self._last_exit_bar_ts: Optional[int] = None
        self._last_hard_sl_bar_ts: Optional[int] = None
        self._latest_analysis: dict = {}

    # ---------------------------- generic helpers ----------------------------
    @staticmethod
    def _finite(*xs) -> bool:
        return all(np.isfinite(float(x)) for x in xs)

    @staticmethod
    def _clamp(x, lo, hi):
        return max(lo, min(hi, x))

    @staticmethod
    def _bar_ts(c):
        return int(getattr(c, "timestamp", 0) or 0)

    @staticmethod
    def _confirmed_pivots(candles, span=2):
        highs, lows = [], []
        for i in range(span, len(candles) - span):
            h = float(candles[i].high); l = float(candles[i].low)
            hw = [float(x.high) for x in candles[i-span:i+span+1]]
            lw = [float(x.low) for x in candles[i-span:i+span+1]]
            if h >= max(hw): highs.append((i, h))
            if l <= min(lw): lows.append((i, l))
        return highs, lows

    @staticmethod
    def _nearest_above(levels, price):
        xs = [float(x) for x in levels if float(x) > price]
        return min(xs) if xs else None

    @staticmethod
    def _nearest_below(levels, price):
        xs = [float(x) for x in levels if float(x) < price]
        return max(xs) if xs else None

    @staticmethod
    def _chop(candles, period=14):
        if len(candles) < period + 1: return float("nan")
        trs = []
        for i in range(len(candles)-period, len(candles)):
            c = candles[i]; p = candles[i-1]
            trs.append(max(float(c.high)-float(c.low), abs(float(c.high)-float(p.close)), abs(float(c.low)-float(p.close))))
        hi = max(float(c.high) for c in candles[-period:])
        lo = min(float(c.low) for c in candles[-period:])
        if hi <= lo: return 100.0
        return 100.0 * math.log10(sum(trs) / (hi-lo)) / math.log10(period)

    # ---------------------------- 15M analysis ----------------------------
    def _analyze_15m(self, candles):
        if len(candles) < 70:
            return {"ready": False, "reason": "15M_WARMUP"}

        closes = [float(c.close) for c in candles]
        opens = [float(c.open) for c in candles]
        highs = [float(c.high) for c in candles]
        lows = [float(c.low) for c in candles]
        vols = [float(c.volume or 0) for c in candles]

        e20 = self.ema(closes, 20); e50 = self.ema(closes, 50)
        e200 = self.ema(closes, 200) if len(closes) >= 205 else np.full(len(closes), np.nan)
        hma16 = self.hma(closes, 16)
        atr = self.atr(candles, 14)
        rsi = self.rsi(closes, 14); rsi_sma = self.sma(list(rsi), 14)
        macd, macd_sig, hist = self.macd(closes, 12, 26, 9)
        adx, dip, dim = self.adx(candles, 14)
        chop = self._chop(candles, 14)

        vals = [e20[-1], e50[-1], hma16[-1], atr[-1], rsi[-1], rsi_sma[-1], macd[-1], macd_sig[-1], hist[-1], adx[-1], dip[-1], dim[-1], chop]
        if not self._finite(*vals): return {"ready": False, "reason": "15M_INDICATOR_NA"}

        c = closes[-1]; o = opens[-1]; h = highs[-1]; l = lows[-1]
        a = max(float(atr[-1]), 1e-12)
        ema20 = float(e20[-1]); ema50 = float(e50[-1]); ema200 = float(e200[-1]) if np.isfinite(e200[-1]) else None
        r = float(rsi[-1]); rs = float(rsi_sma[-1]); m = float(macd[-1]); ms = float(macd_sig[-1]); mh = float(hist[-1])
        adxv = float(adx[-1]); plus = float(dip[-1]); minus = float(dim[-1]); ch = float(chop)
        slope20 = (ema20 - float(e20[-4])) / a
        slope50 = (ema50 - float(e50[-6])) / a
        hma_slope = (float(hma16[-1]) - float(hma16[-3])) / a
        rng = max(h-l, 1e-12); body_atr = abs(c-o)/a; close_pos = (c-l)/rng

        ph, pl = self._confirmed_pivots(candles, self.PIVOT_SPAN)
        last_ph = ph[-1] if ph else None; prev_ph = ph[-2] if len(ph) > 1 else None
        last_pl = pl[-1] if pl else None; prev_pl = pl[-2] if len(pl) > 1 else None
        hh = bool(last_ph and prev_ph and last_ph[1] > prev_ph[1])
        lh = bool(last_ph and prev_ph and last_ph[1] < prev_ph[1])
        hl = bool(last_pl and prev_pl and last_pl[1] > prev_pl[1])
        ll = bool(last_pl and prev_pl and last_pl[1] < prev_pl[1])
        structure = 2 if hh and hl else -2 if lh and ll else 1 if (hh or hl) else -1 if (lh or ll) else 0

        ph_level = float(last_ph[1]) if last_ph else None
        pl_level = float(last_pl[1]) if last_pl else None
        prev_c = closes[-2]
        bos_up = bool(ph_level is not None and c > ph_level + self.BOS_BUFFER_ATR*a and prev_c <= ph_level and body_atr >= .30)
        bos_dn = bool(pl_level is not None and c < pl_level - self.BOS_BUFFER_ATR*a and prev_c >= pl_level and body_atr >= .30)
        choch_up = structure <= 0 and bos_up
        choch_dn = structure >= 0 and bos_dn
        sweep_low = bool(pl_level is not None and l < pl_level-self.SWEEP_BUFFER_ATR*a and c > pl_level and c > o and close_pos >= .58)
        sweep_high = bool(ph_level is not None and h > ph_level+self.SWEEP_BUFFER_ATR*a and c < ph_level and c < o and close_pos <= .42)

        resist = self._nearest_above([x[1] for x in ph[-10:]], c)
        support = self._nearest_below([x[1] for x in pl[-10:]], c)
        room_long = 5.0 if resist is None else max(0.0, (resist-c)/a)
        room_short = 5.0 if support is None else max(0.0, (c-support)/a)
        near_res = resist is not None and abs(resist-c) <= .50*a
        near_sup = support is not None and abs(c-support) <= .50*a

        # Fib value from latest completed impulse leg.
        fib_long = fib_short = False
        fib_levels = {}
        if last_ph:
            lows_before = [x for x in pl if x[0] < last_ph[0]]
            if lows_before and last_ph[1] > lows_before[-1][1]:
                lo = lows_before[-1][1]; hi = last_ph[1]; d = hi-lo
                fib_levels["long"] = [hi-d*.382, hi-d*.50, hi-d*.618]
                fib_long = any(abs(c-x) <= .22*a for x in fib_levels["long"])
        if last_pl:
            highs_before = [x for x in ph if x[0] < last_pl[0]]
            if highs_before and highs_before[-1][1] > last_pl[1]:
                hi = highs_before[-1][1]; lo = last_pl[1]; d = hi-lo
                fib_levels["short"] = [lo+d*.382, lo+d*.50, lo+d*.618]
                fib_short = any(abs(c-x) <= .22*a for x in fib_levels["short"])

        bull = ema20 > ema50 and slope20 >= -.02 and c >= ema50
        bear = ema20 < ema50 and slope20 <= .02 and c <= ema50
        macro_bull = ema200 is not None and ema50 > ema200
        macro_bear = ema200 is not None and ema50 < ema200

        # Evidence score: 5 independent 0..2 blocks.
        trend_l = (2 if bull and macro_bull else 1 if bull else 0) + (1 if slope20 > .05 else 0)
        trend_s = (2 if bear and macro_bear else 1 if bear else 0) + (1 if slope20 < -.05 else 0)
        trend_l = min(2.0, trend_l); trend_s = min(2.0, trend_s)

        quality_l = (1 if adxv >= 18 and plus > minus else 0) + (1 if ch < 55 and adxv >= self.ADX_MIN else 0)
        quality_s = (1 if adxv >= 18 and minus > plus else 0) + (1 if ch < 55 and adxv >= self.ADX_MIN else 0)
        structure_l = min(2.0, (1 if hh or hl else 0) + (1 if bos_up or choch_up or sweep_low else 0))
        structure_s = min(2.0, (1 if lh or ll else 0) + (1 if bos_dn or choch_dn or sweep_high else 0))
        location_l = min(2.0, (1 if near_sup or fib_long or abs(c-ema20) <= .28*a else 0) + (1 if room_long >= 1.2 else 0))
        location_s = min(2.0, (1 if near_res or fib_short or abs(c-ema20) <= .28*a else 0) + (1 if room_short >= 1.2 else 0))
        momentum_l = min(2.0, (1 if r > rs else 0) + (1 if m > ms or mh > 0 else 0))
        momentum_s = min(2.0, (1 if r < rs else 0) + (1 if m < ms or mh < 0 else 0))
        score_l = trend_l + quality_l + structure_l + location_l + momentum_l
        score_s = trend_s + quality_s + structure_s + location_s + momentum_s

        # Setup detection. Exact setup takes precedence over generic bias.
        setups = []
        if bull and (abs(c-ema20) <= .40*a or fib_long or near_sup) and not near_res and (r >= rs or hma_slope >= 0):
            setups.append(("PB", score_l))
        if sweep_low and (r >= rs or hma_slope >= 0): setups.append(("LQ", score_l + .25))
        if bos_up and room_long >= 1.2 and body_atr >= .35 and not near_res: setups.append(("BO_DIRECT", score_l + .25))
        if ph_level is not None and c > ph_level and not bos_up and room_long >= 1.0:
            # A breakout that occurred recently but is no longer the first bar is a retest candidate.
            setups.append(("BO_RETEST", score_l))
        if choch_up and (sweep_low or near_sup or fib_long): setups.append(("REV", score_l + .50))
        if bear and (abs(c-ema20) <= .40*a or fib_short or near_res) and not near_sup and (r <= rs or hma_slope <= 0):
            setups.append(("PB", score_s))
        if sweep_high and (r <= rs or hma_slope <= 0): setups.append(("LQ", score_s + .25))
        if bos_dn and room_short >= 1.2 and body_atr >= .35 and not near_sup: setups.append(("BO_DIRECT", score_s + .25))
        if pl_level is not None and c < pl_level and not bos_dn and room_short >= 1.0: setups.append(("BO_RETEST", score_s))
        if choch_dn and (sweep_high or near_res or fib_short): setups.append(("REV", score_s + .50))

        qualified = []
        for setup, sc in setups:
            if sc >= self.SCORE_MIN[setup]:
                direction = "long" if (setup in ("LQ","REV") and sweep_low) or score_l >= score_s else "short"
                if setup == "PB": direction = "long" if bull else "short"
                if setup in ("BO_DIRECT","BO_RETEST"): direction = "long" if c >= ema50 else "short"
                qualified.append((setup, direction, float(sc)))

        selected = None
        for name in self.SETUP_PRIORITY:
            candidates = [x for x in qualified if x[0] == name]
            if candidates:
                selected = max(candidates, key=lambda x: x[2]); break

        regime = (1 if adxv >= self.ADX_MIN else 0) + (1 if ch < self.CHOP_MAX else 0) + (1 if len(vols) >= 21 and vols[-1] >= .65*np.median(vols[-21:-1]) else 0)
        direction = selected[1] if selected else ("long" if score_l-score_s >= 1.5 and score_l >= 6.5 else "short" if score_s-score_l >= 1.5 and score_s >= 6.5 else None)

        out = {
            "ready": True, "direction": direction, "setup": selected[0] if selected else None,
            "score": round(float(selected[2]),2) if selected else 0.0,
            "long_score": round(float(score_l),2), "short_score": round(float(score_s),2),
            "adx": round(adxv,2), "chop": round(ch,2), "atr": a, "atr_activity": regime,
            "rsi": round(r,2), "rsi_sma": round(rs,2), "macd_hist": round(mh,6),
            "ema20": ema20, "ema50": ema50, "ema20_slope_atr": round(slope20,3),
            "hma16_slope_atr": round(hma_slope,3), "structure": structure,
            "bos_up": bos_up, "bos_dn": bos_dn, "choch_up": choch_up, "choch_dn": choch_dn,
            "sweep_low": sweep_low, "sweep_high": sweep_high,
            "support": support, "resistance": resist, "room_long_atr": room_long, "room_short_atr": room_short,
            "fib_long": fib_long, "fib_short": fib_short,
            "reason": f"{selected[0]} {selected[1].upper()} score={selected[2]:.2f}" if selected else "NO_QUALIFIED_SETUP",
        }
        self._latest_analysis = out
        return out

    # ---------------------------- 5M execution ----------------------------
    def _execution_5m(self, candles, direction, setup, current_price):
        if len(candles) < 30 or direction not in ("long", "short"):
            return {"ready": False, "reason": "5M_WARMUP"}
        closes = [float(c.close) for c in candles]; highs=[float(c.high) for c in candles]; lows=[float(c.low) for c in candles]; opens=[float(c.open) for c in candles]
        vols=[float(c.volume or 0) for c in candles]
        e20=self.ema(closes,20); atr=self.atr(candles,14); adx,dp,dm=self.adx(candles,14); ch=self._chop(candles,14)
        if not self._finite(e20[-1],e20[-4],atr[-1],adx[-1],dp[-1],dm[-1],ch): return {"ready":False,"reason":"5M_INDICATOR_NA"}
        a=max(float(atr[-1]),1e-12); c=closes[-1]; o=opens[-1]; h=highs[-1]; l=lows[-1]
        rng=max(h-l,1e-12); body=abs(c-o)/a; pos=(c-l)/rng; slope=(float(e20[-1])-float(e20[-4]))/a
        prev_hi=max(highs[-5:-1]); prev_lo=min(lows[-5:-1])
        sweep_l=l < prev_lo-.03*a and c > prev_lo and c > o
        sweep_s=h > prev_hi+.03*a and c < prev_hi and c < o
        reclaim_l=c>float(e20[-1]) and c>o and pos>=.55
        reclaim_s=c<float(e20[-1]) and c<o and pos<=.45
        breakout_l=c>prev_hi and c>o and body>=.25
        breakout_s=c<prev_lo and c<o and body>=.25
        if direction=="long":
            if sweep_l: trigger="SWEEP_RECLAIM"
            elif breakout_l: trigger="MICRO_BREAKOUT"
            elif reclaim_l and abs(c-float(e20[-1]))<=1.0*a: trigger="PULLBACK_RECLAIM"
            else: trigger=None
        else:
            if sweep_s: trigger="SWEEP_RECLAIM"
            elif breakout_s: trigger="MICRO_BREAKOUT"
            elif reclaim_s and abs(c-float(e20[-1]))<=1.0*a: trigger="PULLBACK_RECLAIM"
            else: trigger=None
        median_vol=np.median(vols[-21:-1]) if len(vols)>=21 else 0
        vol_ratio=(vols[-1]/median_vol) if median_vol>0 else 1.0
        blocks=[]
        if adx[-1] < self.ADX_MIN: blocks.append("ADX")
        if ch >= self.CHOP_MAX: blocks.append("CHOP")
        if trigger is None: blocks.append("NO_5M_TRIGGER")
        if abs(c-float(e20[-1]))/a > self.MAX_CHASE_ATR: blocks.append("ANTI_CHASE")
        if trigger=="MICRO_BREAKOUT" and vol_ratio < .65: blocks.append("BREAKOUT_VOLUME")
        if trigger=="MICRO_BREAKOUT" and (slope <= 0 if direction=="long" else slope >= 0): blocks.append("5M_SLOPE")
        if direction=="long" and pos < .55: blocks.append("WEAK_CLOSE")
        if direction=="short" and pos > .45: blocks.append("WEAK_CLOSE")
        return {"ready":not blocks,"trigger":trigger,"blocks":blocks,"atr":a,"ema20":float(e20[-1]),"slope":slope,"body_atr":body,"volume_ratio":vol_ratio,"bar_ts":self._bar_ts(candles[-1]),"reason":"5M_EXECUTION_READY" if not blocks else ",".join(blocks)}

    # ---------------------------- risk / targets ----------------------------
    def _make_levels(self, candles5, direction, setup, entry, analysis):
        a=max(float(self.atr(candles5,14)[-1]),1e-12)
        lookback=5 if setup in ("PB","LQ","REV") else 3
        lo=min(float(c.low) for c in candles5[-lookback:]); hi=max(float(c.high) for c in candles5[-lookback:])
        if direction=="long":
            raw_sl=lo-self.SL_BUFFER_ATR*a
            risk=entry-raw_sl
            sl_atr=risk/a
            if sl_atr < self.MIN_SL_ATR: raw_sl=entry-self.MIN_SL_ATR*a
            elif sl_atr > self.MAX_SL_ATR: raw_sl=entry-self.MAX_SL_ATR*a
            risk=entry-raw_sl
            candidates=[x for x in (analysis.get("resistance"),) if x is not None and x>entry]
            tp2=min(candidates) if candidates else entry+self.TP2_FALLBACK_R*risk
            if candidates and (tp2-entry)/risk < self.TP2_MIN_R: tp2=entry+self.TP2_MIN_R*risk
            tp2=min(tp2,entry+self.TP2_MAX_R*risk)
        else:
            raw_sl=hi+self.SL_BUFFER_ATR*a
            risk=raw_sl-entry
            sl_atr=risk/a
            if sl_atr < self.MIN_SL_ATR: raw_sl=entry+self.MIN_SL_ATR*a
            elif sl_atr > self.MAX_SL_ATR: raw_sl=entry+self.MAX_SL_ATR*a
            risk=raw_sl-entry
            candidates=[x for x in (analysis.get("support"),) if x is not None and x<entry]
            tp2=max(candidates) if candidates else entry-self.TP2_FALLBACK_R*risk
            if candidates and (entry-tp2)/risk < self.TP2_MIN_R: tp2=entry-self.TP2_MIN_R*risk
            tp2=max(tp2,entry-self.TP2_MAX_R*risk)
        if risk/entry < self.MIN_ECONOMIC_RISK_PCT: return None
        tp1=entry+self.TP1_R*risk if direction=="long" else entry-self.TP1_R*risk
        return {"sl":float(raw_sl),"tp1":float(tp1),"tp2":float(tp2),"risk":float(risk),"risk_atr":float(risk/a)}

    # ---------------------------- public strategy interface ----------------------------
    async def analyze(self, candles:list, current_price:float, mtf_candles:dict=None) -> Signal:
        a=self._analyze_15m(candles)
        if not a.get("ready"):
            return Signal(SignalType.HOLD,self.symbol,float(current_price),0,reason=a.get("reason","HOLD"),confidence=.0,metadata=a)
        if self._open_position is not None:
            return Signal(SignalType.HOLD,self.symbol,float(current_price),0,reason="POSITION_OPEN",confidence=0,metadata=a)
        if a.get("direction") not in ("long","short") or not a.get("setup"):
            return Signal(SignalType.HOLD,self.symbol,float(current_price),0,reason=a.get("reason","NO_SETUP"),confidence=0,metadata=a)
        c5=(mtf_candles or {}).get("5m") or (mtf_candles or {}).get("5M") or []
        ex=self._execution_5m(c5,a["direction"],a["setup"],current_price)
        if not ex.get("ready"):
            md={**a,"execution":ex}
            return Signal(SignalType.HOLD,self.symbol,float(current_price),0,reason=f"WAIT_5M:{ex.get('reason')}",confidence=min(1,a["score"]/10),metadata=md)
        levels=self._make_levels(c5,a["direction"],a["setup"],float(current_price),a)
        if not levels:
            return Signal(SignalType.HOLD,self.symbol,float(current_price),0,reason="FEE_EDGE_TOO_TIGHT",confidence=0,metadata={**a,"execution":ex})
        side=SignalType.BUY if a["direction"]=="long" else SignalType.SELL
        confidence=self._clamp((a["score"]/10)*.75 + min(1,a["atr_activity"]/3)*.15 + (.10 if ex["trigger"] else 0),0,1)
        md={**a,"execution":ex,"sl":levels["sl"],"tp1":levels["tp1"],"tp2":levels["tp2"],"initial_risk":levels["risk"],"setup_engine":a["setup"],"execution_engine":ex["trigger"],"forecast_engine":"STRUCTURE/FIB_TARGET"}
        self._pending_entry=True
        return Signal(side,self.symbol,float(current_price),0,reason=f"{a['setup']} {a['direction'].upper()} score={a['score']:.2f} | {ex['trigger']}",confidence=confidence,metadata=md)

    # ---------------------------- lifecycle ----------------------------
    def attach_existing_position(self,direction:str,entry_price:float,stop_loss:Optional[float]=None,take_profit:Optional[float]=None):
        self._open_position=direction.lower(); self._entry_price=float(entry_price); self._entry_sl=float(stop_loss) if stop_loss is not None else None; self._entry_tp=float(take_profit) if take_profit is not None else None
        self._initial_risk=abs(self._entry_price-self._entry_sl) if self._entry_sl is not None else None
        self._tp1_done=False; self._pending_entry=False

    def _reset_position_state(self):
        self._open_position=None; self._entry_price=None; self._entry_sl=None; self._entry_tp=None; self._initial_risk=None; self._tp1_done=False; self._pending_entry=False

    def on_entry_filled(self,entry_price:float,stop_loss:Optional[float]=None,take_profit:Optional[float]=None,direction:Optional[str]=None):
        if direction: self._open_position=direction.lower()
        self._entry_price=float(entry_price); self._entry_sl=float(stop_loss) if stop_loss is not None else None; self._entry_tp=float(take_profit) if take_profit is not None else None
        self._initial_risk=abs(self._entry_price-self._entry_sl) if self._entry_sl is not None else None; self._tp1_done=False; self._pending_entry=False

    def tick_open_position(self,current_price:float,position_key:Optional[str]=None):
        if self._open_position is None or self._entry_price is None or self._initial_risk is None or self._initial_risk<=0: return None
        side=self._open_position; p=float(current_price); entry=self._entry_price; risk=self._initial_risk
        profit=p-entry if side=="long" else entry-p; r=profit/risk
        if not self._tp1_done and r>=self.TP1_R:
            lock=entry+self.TP1_LOCK_R*risk if side=="long" else entry-self.TP1_LOCK_R*risk
            self._tp1_done=True; self._entry_sl=lock
            return PositionUpdate(action="partial_tp",close_pct=self.TP1_CLOSE_PCT,new_sl=round(lock,8),reason=f"Sentinel TP1 +{r:.2f}R: close 50%; runner SL +{self.TP1_LOCK_R:.2f}R")
        if self._entry_tp is not None:
            hit=(side=="long" and p>=self._entry_tp) or (side=="short" and p<=self._entry_tp)
            if hit:
                return PositionUpdate(action="close",reason=f"Sentinel TP2 hit +{r:.2f}R")
        if self._entry_sl is not None:
            hit=(side=="long" and p<=self._entry_sl) or (side=="short" and p>=self._entry_sl)
            if hit:
                return PositionUpdate(action="close",reason="Sentinel Stop-Loss")
        return None

    def exit_cooldown_bars(self,hard_sl:bool=False):
        return self.HARD_SL_COOLDOWN_5M if hard_sl else self.NORMAL_COOLDOWN_5M
