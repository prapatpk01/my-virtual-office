"""Sentinel V1.3 — 1H S/R map + fast 15M execution.

Architecture
- 1H ONLY: build confirmed S1/S2/R1/R2 and ATR14 map.
- 1H ATR: measure S1<->R1 profit room, proximity band and SL buffer.
- 15M ONLY: execute proximity rejection / reclaim / displacement.
- No EMA/HMA cross is required for entry.
- Sentinel X + MCDX remain directional/context validators.

LONG
- 1H S1/S2/R1/R2 must exist.
- S1->R1 room >= configured minimum in ATR(1H).
- Price approaches 1H S1 inside the proximity band.
- A completed 15M candle rejects/reclaims S1 OR a bullish displacement candle
  moves away from S1 after proximity.
- SL below 1H S2 by ATR(1H) buffer; TP at 1H R1.

SHORT is the exact inverse around 1H R1/R2 with TP at 1H S1.
"""
from __future__ import annotations

from typing import Optional
import math
import numpy as np

from .base import BaseStrategy, Signal, SignalType


class SentinelStrategy(BaseStrategy):
    VERSION = "1.3"

    def __init__(
        self,
        symbol: str,
        params: Optional[dict] = None,
        min_context_score: float = 65.0,
        min_location_atr: float = 1.20,
        min_rr: float = 1.50,
        entry_zone_atr: float = 0.30,
        reversal_proximity_atr: float = 0.20,
        sl_buffer_atr: float = 0.15,
        sr_merge_atr: float = 0.65,
        pivot_span: int = 3,
    ):
        super().__init__(symbol, params)
        self.min_context_score = float(min_context_score)
        self.min_location_atr = float(min_location_atr)
        self.min_rr = float(min_rr)
        self.entry_zone_atr = float(entry_zone_atr)
        self.reversal_proximity_atr = max(0.05, float(reversal_proximity_atr))
        self.sl_buffer_atr = float(sl_buffer_atr)
        self.sr_merge_atr = float(sr_merge_atr)
        self.pivot_span = max(2, int(pivot_span))
        self.name = f"Sentinel({symbol})"

    @staticmethod
    def _clamp(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    @staticmethod
    def _confirmed_pivots(candles: list, span: int) -> tuple[list[float], list[float]]:
        highs: list[float] = []
        lows: list[float] = []
        if len(candles) < span * 2 + 3:
            return highs, lows
        for i in range(span, len(candles) - span):
            hi = float(candles[i].high)
            lo = float(candles[i].low)
            if hi >= max(float(c.high) for c in candles[i-span:i]) and hi >= max(float(c.high) for c in candles[i+1:i+span+1]):
                highs.append(hi)
            if lo <= min(float(c.low) for c in candles[i-span:i]) and lo <= min(float(c.low) for c in candles[i+1:i+span+1]):
                lows.append(lo)
        return highs, lows

    def _sr_map_1h(self, mtf: dict, price: float) -> dict:
        """Build S1/S2/R1/R2 exclusively from confirmed 1H pivots."""
        h1 = list((mtf or {}).get("1h") or [])
        if len(h1) < 60:
            return {"ready": False, "reason": "1H S/R warmup", "bars_1h": len(h1)}

        atr_arr = self.atr(h1, 14)
        atr_1h = float(atr_arr[-1]) if len(atr_arr) and np.isfinite(atr_arr[-1]) else 0.0
        if atr_1h <= 0:
            return {"ready": False, "reason": "1H ATR unavailable", "bars_1h": len(h1)}

        h1_h, h1_l = self._confirmed_pivots(h1[-160:], self.pivot_span)
        resistance_pool = sorted({float(x) for x in h1_h if float(x) > price})
        support_pool = sorted({float(x) for x in h1_l if float(x) < price}, reverse=True)

        r1 = resistance_pool[0] if resistance_pool else None
        r2 = resistance_pool[1] if len(resistance_pool) > 1 else None
        s1 = support_pool[0] if support_pool else None
        s2 = support_pool[1] if len(support_pool) > 1 else None
        if any(v is None for v in (s1, s2, r1, r2)):
            return {
                "ready": False, "reason": "Need complete 1H S1/S2/R1/R2",
                "s1": s1, "s2": s2, "r1": r1, "r2": r2,
                "atr_1h": atr_1h, "bars_1h": len(h1),
            }

        s1_raw, s2_raw, r1_raw, r2_raw = s1, s2, r1, r2
        if abs(r2-r1) <= atr_1h*self.sr_merge_atr:
            r1 = (r1+r2)/2.0
        if abs(s1-s2) <= atr_1h*self.sr_merge_atr:
            s1 = (s1+s2)/2.0

        return {
            "ready": True,
            "s1": s1, "s2": s2_raw, "r1": r1, "r2": r2_raw,
            "s1_raw": s1_raw, "r1_raw": r1_raw,
            "atr_1h": atr_1h, "bars_1h": len(h1), "source_tf": "1H",
        }

    def _structure(self, candles: list) -> str:
        hs, ls = self._confirmed_pivots(candles[-80:], 3)
        if len(hs) >= 2 and len(ls) >= 2:
            if hs[-1] > hs[-2] and ls[-1] > ls[-2]:
                return "BULL"
            if hs[-1] < hs[-2] and ls[-1] < ls[-2]:
                return "BEAR"
        return "MIXED"

    @staticmethod
    def _norm(v: float, lo: float, hi: float) -> float:
        if not math.isfinite(v) or not math.isfinite(lo) or not math.isfinite(hi) or hi == lo:
            return 50.0
        return max(0.0, min(100.0, (v-lo)/(hi-lo)*100.0))

    def _mcdx_context(self, candles: list, mtf: dict) -> dict:
        """Compact MCDX context; 15M flow with 1H trend confirmation."""
        n = len(candles)
        if n < 120:
            return {"long_score": 0.0, "short_score": 0.0, "smart_flow": 50.0, "ready": False}

        closes = np.asarray([float(c.close) for c in candles], dtype=float)
        highs = np.asarray([float(c.high) for c in candles], dtype=float)
        lows = np.asarray([float(c.low) for c in candles], dtype=float)
        vols = np.asarray([max(0.0, float(c.volume)) for c in candles], dtype=float)

        lo50=float(np.min(lows[-50:])); hi50=float(np.max(highs[-50:])); rng=max(hi50-lo50,1e-12)
        smart=self._clamp(((closes[-1]*0.96)-lo50)/rng*100.0,0,100)
        float_total=self._clamp(((closes[-1]*1.04)-lo50)/rng*100.0,0,100)
        retail=self._clamp(100.0-float_total,0,100)

        signed=np.sign(np.diff(closes,prepend=closes[0]))*vols
        obv=np.cumsum(signed); obv20=obv[-20:]; std=float(np.std(obv20))
        obv_z=0.0 if std<=1e-12 else (float(obv[-1])-float(np.mean(obv20)))/std
        obv_norm=self._clamp((obv_z+3.0)/6.0*100.0,0,100)
        pv=closes[-80:]*vols[-80:]
        vwap=float(np.sum(pv)/max(float(np.sum(vols[-80:])),1e-12))
        dev=float(np.std(closes[-20:]-vwap))
        vwap_score=self._clamp(50+(closes[-1]-vwap)/max(dev,1e-9)*20,0,100)
        upvol=float(np.sum(vols[-20:][np.diff(closes[-21:])>0]))
        dnvol=float(np.sum(vols[-20:][np.diff(closes[-21:])<0]))
        vol_agree=50.0 if upvol+dnvol<=0 else 100.0*upvol/(upvol+dnvol)
        volume_flow=0.50*vol_agree+0.30*obv_norm+0.20*vwap_score

        rsi_arr=self.rsi(list(closes),14); _,_,hist=self.macd(list(closes),12,26,9)
        rsi=float(rsi_arr[-1]) if np.isfinite(rsi_arr[-1]) else 50.0
        hh=hist[-100:][np.isfinite(hist[-100:])]
        macd_score=50.0 if len(hh)==0 else self._norm(float(hist[-1]),float(np.min(hh)),float(np.max(hh)))
        momentum=0.45*rsi+0.55*macd_score

        h1=list((mtf or {}).get("1h") or [])
        htf_bull=htf_bear=False; mtf_score=50.0
        if len(h1)>=55:
            hc=[float(c.close) for c in h1]; he20=self.ema(hc,20); he50=self.ema(hc,50)
            htf_bull=hc[-1]>he20[-1]>he50[-1]
            htf_bear=hc[-1]<he20[-1]<he50[-1]
            mtf_score=80.0 if htf_bull else 20.0 if htf_bear else 50.0

        structure=self._structure(candles)
        struct_bias=1 if structure=="BULL" else -1 if structure=="BEAR" else 0
        e20=self.ema(list(closes),20); e50=self.ema(list(closes),50)
        trend_score=75.0 if e20[-1]>e50[-1] else 25.0 if e20[-1]<e50[-1] else 50.0
        chip_score=0.70*smart+0.30*(100-retail)
        smart_flow=self._clamp(chip_score*0.35+volume_flow*0.25+momentum*0.15+trend_score*0.15+mtf_score*0.10,0,100)

        long_score=(30.0 if smart>=55 else 18.0 if smart>=50 else 0.0)+(25.0 if smart_flow>=60 else 15.0 if smart_flow>=52 else 0.0)+(20.0 if struct_bias==1 else 8.0 if struct_bias==0 else 0.0)+(10.0 if htf_bull else 0.0)
        short_score=(30.0 if smart<=45 else 18.0 if smart<=50 else 0.0)+(25.0 if smart_flow<=40 else 15.0 if smart_flow<=48 else 0.0)+(20.0 if struct_bias==-1 else 8.0 if struct_bias==0 else 0.0)+(10.0 if htf_bear else 0.0)
        return {"long_score":round(long_score,1),"short_score":round(short_score,1),"smart_money":round(smart,1),"smart_flow":round(smart_flow,1),"structure":structure,"ready":True}

    def _sentinel_context(self, candles: list, mtf: dict) -> dict:
        closes=[float(c.close) for c in candles]
        if len(closes)<60:
            return {"bias":"NEUTRAL","structure":"MIXED","ready":False}
        e20=self.ema(closes,20); e50=self.ema(closes,50); hma16=self.hma(closes,16)
        structure=self._structure(candles)
        bull=bool(e20[-1]>=e50[-1] and e20[-1]>e20[-4] and closes[-1]>=e20[-1])
        bear=bool(e20[-1]<=e50[-1] and e20[-1]<e20[-4] and closes[-1]<=e20[-1])
        _,_,hist=self.macd(closes,12,26,9); r=self.rsi(closes,14)
        fast=float(hma16[-1]-hma16[-3]) if np.isfinite(hma16[-1]) and np.isfinite(hma16[-3]) else 0.0
        histv=float(hist[-1]) if np.isfinite(hist[-1]) else 0.0
        rsiv=float(r[-1]) if np.isfinite(r[-1]) else 50.0
        return {"bias":"BULL" if bull else "BEAR" if bear else "BALANCED","structure":structure,"sme_bull":fast>0 and histv>=0 and rsiv>=45,"sme_bear":fast<0 and histv<=0 and rsiv<=55,"ready":True}

    async def analyze(self, candles: list, current_price: float, mtf_candles: dict = None) -> Signal:
        if len(candles) < 120:
            return Signal(SignalType.HOLD,self.symbol,current_price,0.0,"15M Sentinel warmup",metadata={"strategy":"SENTINEL","version":self.VERSION})

        mtf=mtf_candles or {}
        bar=candles[-1]; prev=candles[-2]
        close=float(bar.close); op=float(bar.open); low=float(bar.low); high=float(bar.high)
        prev_close=float(prev.close); prev_open=float(prev.open); prev_low=float(prev.low); prev_high=float(prev.high)

        sr=self._sr_map_1h(mtf,close)
        meta={"strategy":"SENTINEL","version":self.VERSION,"sr_tf":"1H","entry_tf":"15M","entry_engine":"PROXIMITY_REJECTION_DISPLACEMENT"}
        meta.update({k:v for k,v in sr.items() if k not in {"ready"}})
        if not sr.get("ready"):
            return Signal(SignalType.HOLD,self.symbol,current_price,0.0,sr.get("reason","1H map unavailable"),metadata=meta)

        s1,s2,r1,r2=sr["s1"],sr["s2"],sr["r1"],sr["r2"]
        atr_1h=float(sr["atr_1h"])
        atr15_arr=self.atr(candles,14)
        atr_15m=float(atr15_arr[-1]) if len(atr15_arr) and np.isfinite(atr15_arr[-1]) else 0.0
        sx=self._sentinel_context(candles,mtf)
        mc=self._mcdx_context(candles,mtf)
        meta.update({"s1":s1,"s2":s2,"r1":r1,"r2":r2,"atr_1h":round(atr_1h,8),"atr_15m":round(atr_15m,8),"sentinel_x":sx,"mcdx":mc})

        # HARD location gate: if S1 and R1 are too close, no 15M signal may trade.
        location_atr=(r1-s1)/atr_1h
        meta.update({"location_atr":round(location_atr,2),"min_location_atr":self.min_location_atr})
        if location_atr < self.min_location_atr:
            meta["room_gate"]="BLOCK"
            return Signal(SignalType.HOLD,self.symbol,current_price,0.0,f"1H S1-R1 room BLOCK ({location_atr:.2f} ATR1H < {self.min_location_atr:.2f})",metadata=meta)
        meta["room_gate"]="PASS"

        zone=self.entry_zone_atr*atr_1h
        proximity=self.reversal_proximity_atr*atr_1h

        # FAST 15M entry engine. No EMA/HMA cross is required.
        # 1) Same-bar proximity rejection: price only needs to reach the 0.20 ATR1H
        #    band (not literally touch S1/R1) and then close back away from it.
        bar_range=max(high-low,1e-12)
        body=abs(close-op)
        lower_wick=min(op,close)-low
        upper_wick=high-max(op,close)
        long_near_now=(low >= s1 and low <= s1+proximity)
        short_near_now=(high <= r1 and high >= r1-proximity)
        long_rejection=bool(long_near_now and close>op and close>=low+0.55*bar_range and (lower_wick>=0.25*bar_range or body>=0.45*bar_range))
        short_rejection=bool(short_near_now and close<op and close<=high-0.55*bar_range and (upper_wick>=0.25*bar_range or body>=0.45*bar_range))

        # 2) Direct reclaim/rejection for a candle that actually probes the wider
        #    entry zone but closes back on the correct side of the 1H level.
        long_reclaim=bool(low <= s1+zone and close >= s1 and close>op)
        short_reclaim=bool(high >= r1-zone and close <= r1 and close<op)

        # 3) Two-candle proximity -> displacement. Previous 15M bar approaches the
        #    1H level; the next bar can fire as soon as it shows real displacement.
        long_near_prev=(prev_low >= s1 and prev_low <= s1+proximity)
        short_near_prev=(prev_high <= r1 and prev_high >= r1-proximity)
        long_displacement=bool(long_near_prev and close>prev_close and close>op and atr_15m>0 and body>=0.60*atr_15m and close>s1+0.10*atr_1h)
        short_displacement=bool(short_near_prev and close<prev_close and close<op and atr_15m>0 and body>=0.60*atr_15m and close<r1-0.10*atr_1h)

        long_trigger=long_rejection or long_reclaim or long_displacement
        short_trigger=short_rejection or short_reclaim or short_displacement

        long_sl=s2-self.sl_buffer_atr*atr_1h; long_tp=r1
        short_sl=r2+self.sl_buffer_atr*atr_1h; short_tp=s1
        long_risk=close-long_sl; long_reward=long_tp-close
        short_risk=short_sl-close; short_reward=close-short_tp
        long_rr=long_reward/long_risk if long_risk>0 else 0.0
        short_rr=short_reward/short_risk if short_risk>0 else 0.0

        meta.update({
            "long_rr":round(long_rr,2),"short_rr":round(short_rr,2),
            "long_sl":long_sl,"long_tp":long_tp,"short_sl":short_sl,"short_tp":short_tp,
            "reversal_proximity_atr_1h":self.reversal_proximity_atr,
            "long_near_now":long_near_now,"short_near_now":short_near_now,
            "long_rejection":long_rejection,"short_rejection":short_rejection,
            "long_reclaim":long_reclaim,"short_reclaim":short_reclaim,
            "long_near_prev":long_near_prev,"short_near_prev":short_near_prev,
            "long_displacement":long_displacement,"short_displacement":short_displacement,
            "ema_hma_cross_required":False,
        })

        # Context validates direction; location is still the actual entry authority.
        long_context=(sx.get("bias")!="BEAR" and sx.get("structure")!="BEAR" and (sx.get("sme_bull") or sx.get("bias")=="BULL") and mc.get("long_score",0)>=self.min_context_score and mc.get("smart_flow",50)>=52)
        short_context=(sx.get("bias")!="BULL" and sx.get("structure")!="BULL" and (sx.get("sme_bear") or sx.get("bias")=="BEAR") and mc.get("short_score",0)>=self.min_context_score and mc.get("smart_flow",50)<=48)

        if long_trigger and long_context and long_rr>=self.min_rr and long_reward>0:
            trigger="1H_S1__15M_REJECTION" if long_rejection else "1H_S1__15M_RECLAIM" if long_reclaim else "1H_S1__15M_DISPLACEMENT"
            meta.update({"entry_location":"1H_S1","stop_basis":"BELOW_1H_S2","tp_basis":"1H_R1","stop_loss":long_sl,"take_profit":long_tp,"rr_ratio":round(long_rr,2),"entry_trigger":trigger})
            return Signal(SignalType.BUY,self.symbol,current_price,0.0,f"SENTINEL LONG {trigger} | room {location_atr:.2f}ATR1H | RR {long_rr:.2f}",confidence=min(1.0,0.50+mc.get("long_score",0)/200.0),metadata=meta)

        if short_trigger and short_context and short_rr>=self.min_rr and short_reward>0:
            trigger="1H_R1__15M_REJECTION" if short_rejection else "1H_R1__15M_RECLAIM" if short_reclaim else "1H_R1__15M_DISPLACEMENT"
            meta.update({"entry_location":"1H_R1","stop_basis":"ABOVE_1H_R2","tp_basis":"1H_S1","stop_loss":short_sl,"take_profit":short_tp,"rr_ratio":round(short_rr,2),"entry_trigger":trigger})
            return Signal(SignalType.SELL,self.symbol,current_price,0.0,f"SENTINEL SHORT {trigger} | room {location_atr:.2f}ATR1H | RR {short_rr:.2f}",confidence=min(1.0,0.50+mc.get("short_score",0)/200.0),metadata=meta)

        reasons=[]
        if long_trigger and not long_context: reasons.append("15M LONG rejection/displacement ready at 1H S1; context not confirmed")
        elif short_trigger and not short_context: reasons.append("15M SHORT rejection/displacement ready at 1H R1; context not confirmed")
        elif long_trigger and long_rr<self.min_rr: reasons.append(f"LONG trigger but RR {long_rr:.2f} < {self.min_rr:.2f}")
        elif short_trigger and short_rr<self.min_rr: reasons.append(f"SHORT trigger but RR {short_rr:.2f} < {self.min_rr:.2f}")
        elif long_near_now or long_near_prev: reasons.append(f"ARMED LONG: 15M near 1H S1 within {self.reversal_proximity_atr:.2f} ATR1H; waiting rejection/displacement")
        elif short_near_now or short_near_prev: reasons.append(f"ARMED SHORT: 15M near 1H R1 within {self.reversal_proximity_atr:.2f} ATR1H; waiting rejection/displacement")
        else: reasons.append("WAIT 15M proximity at 1H S1/R1")
        return Signal(SignalType.HOLD,self.symbol,current_price,0.0,"; ".join(reasons),metadata=meta)
