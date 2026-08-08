"""Sentinel strategy — S1/R1 location execution using Sentinel X + MCDX context.

Source concepts are ported from the user's Sentinel X v2.3 and MCDX Sentinel v3 Pine
indicators. This strategy intentionally keeps responsibilities simple:

1) Sentinel X supplies adaptive confirmed S/R (S1/S2/R1/R2), structure and trend context.
2) MCDX Sentinel supplies Smart Money / Smart Flow participation context.
3) Location is the actual entry engine.

LONG
- enter from S1 only after a confirmed reclaim/rejection at S1
- SL below S2 by a small ATR buffer
- TP at R1

SHORT
- enter from R1 only after a confirmed rejection/reclaim downward at R1
- SL above R2 by a small ATR buffer
- TP at S1

A trade is rejected when S1<->R1 room is too small or the actual TP/SL R:R is below
minimum. S2/R2 are mandatory because the requested stop is beyond those levels.
"""
from __future__ import annotations

from typing import Optional
import math
import numpy as np

from .base import BaseStrategy, Signal, SignalType


class SentinelStrategy(BaseStrategy):
    VERSION = "1.0"

    def __init__(
        self,
        symbol: str,
        params: Optional[dict] = None,
        min_context_score: float = 65.0,
        min_location_atr: float = 1.20,
        min_rr: float = 1.50,
        entry_zone_atr: float = 0.30,
        sl_buffer_atr: float = 0.15,
        sr_merge_atr: float = 0.65,
        pivot_span: int = 4,
    ):
        super().__init__(symbol, params)
        self.min_context_score = float(min_context_score)
        self.min_location_atr = float(min_location_atr)
        self.min_rr = float(min_rr)
        self.entry_zone_atr = float(entry_zone_atr)
        self.sl_buffer_atr = float(sl_buffer_atr)
        self.sr_merge_atr = float(sr_merge_atr)
        self.pivot_span = max(2, int(pivot_span))
        self.name = f"Sentinel({symbol})"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _clamp(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    @staticmethod
    def _confirmed_pivots(candles: list, span: int) -> tuple[list[float], list[float]]:
        """Confirmed pivots only; no future-looking live pivot is used."""
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

    def _adaptive_sr(self, candles_15m: list, mtf: dict, price: float, atr: float) -> dict:
        """Port of Sentinel X nearest local + HTF confirmed pivot S/R pool."""
        local_h, local_l = self._confirmed_pivots(candles_15m[-120:], self.pivot_span)
        h1 = list((mtf or {}).get("1h") or [])
        h4 = list((mtf or {}).get("4h") or [])
        h1_h, h1_l = self._confirmed_pivots(h1[-100:], 3) if h1 else ([], [])
        h4_h, h4_l = self._confirmed_pivots(h4[-100:], 3) if h4 else ([], [])

        # Keep the last two local pivots and the last HTF pivot from each timeframe,
        # mirroring the four-candidate pool used by Sentinel X.
        resistance_pool = (local_h[-2:] + h1_h[-1:] + h4_h[-1:])
        support_pool = (local_l[-2:] + h1_l[-1:] + h4_l[-1:])
        resistance_pool = sorted({float(x) for x in resistance_pool if float(x) > price})
        support_pool = sorted({float(x) for x in support_pool if float(x) < price}, reverse=True)

        r1 = resistance_pool[0] if resistance_pool else None
        r2 = resistance_pool[1] if len(resistance_pool) > 1 else None
        s1 = support_pool[0] if support_pool else None
        s2 = support_pool[1] if len(support_pool) > 1 else None

        # Merge close levels the same way Sentinel X does. If merging would remove
        # S2/R2 we keep the farther raw level for stop construction; entry/TP levels
        # use the merged primary zone.
        r1_raw, r2_raw, s1_raw, s2_raw = r1, r2, s1, s2
        if r1 is not None and r2 is not None and abs(r2-r1) <= atr*self.sr_merge_atr:
            r1 = (r1+r2)/2.0
        if s1 is not None and s2 is not None and abs(s1-s2) <= atr*self.sr_merge_atr:
            s1 = (s1+s2)/2.0

        return {
            "s1": s1, "s2": s2_raw, "r1": r1, "r2": r2_raw,
            "s1_raw": s1_raw, "r1_raw": r1_raw,
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
        """MCDX Sentinel v3 context: chips + Smart Flow + structure + MTF."""
        n = len(candles)
        if n < 120:
            return {"long_score": 0.0, "short_score": 0.0, "ready": False, "reason": "MCDX warmup"}

        closes = np.asarray([float(c.close) for c in candles], dtype=float)
        highs = np.asarray([float(c.high) for c in candles], dtype=float)
        lows = np.asarray([float(c.low) for c in candles], dtype=float)
        vols = np.asarray([max(0.0, float(c.volume)) for c in candles], dtype=float)
        look = 50
        lo50 = float(np.min(lows[-look:])); hi50 = float(np.max(highs[-look:]))
        rng = max(hi50-lo50, 1e-12)
        smart = self._clamp(((closes[-1]*0.96)-lo50)/rng*100.0, 0, 100)
        float_total = self._clamp(((closes[-1]*1.04)-lo50)/rng*100.0, 0, 100)
        retail = self._clamp(100.0-float_total, 0, 100)

        smart_hist = []
        retail_hist = []
        for i in range(n-6, n):
            j0 = max(0, i-look+1)
            l = float(np.min(lows[j0:i+1])); h = float(np.max(highs[j0:i+1])); d=max(h-l,1e-12)
            sm = self._clamp(((closes[i]*0.96)-l)/d*100.0,0,100)
            ft = self._clamp(((closes[i]*1.04)-l)/d*100.0,0,100)
            smart_hist.append(sm); retail_hist.append(self._clamp(100-ft,0,100))
        smart_sma = float(np.mean(smart_hist[-5:])); retail_sma = float(np.mean(retail_hist[-5:]))
        smart_rising = smart > smart_sma and smart > smart_hist[-2]
        smart_falling = smart < smart_sma and smart < smart_hist[-2]
        retail_falling = retail < retail_sma and retail < retail_hist[-2]
        retail_rising = retail > retail_sma and retail > retail_hist[-2]

        # Volume flow: OBV + VWAP location + recent up/down volume agreement.
        signed = np.sign(np.diff(closes, prepend=closes[0])) * vols
        obv = np.cumsum(signed)
        obv20 = obv[-20:]
        obv_std = float(np.std(obv20))
        obv_z = 0.0 if obv_std <= 1e-12 else (float(obv[-1])-float(np.mean(obv20)))/obv_std
        obv_norm = self._clamp((obv_z+3.0)/6.0*100.0,0,100)
        pv = closes[-80:]*vols[-80:]
        vwap = float(np.sum(pv)/max(float(np.sum(vols[-80:])),1e-12))
        dev = float(np.std(closes[-20:]-vwap)); vwap_score=self._clamp(50+(closes[-1]-vwap)/max(dev,1e-9)*20,0,100)
        upvol = float(np.sum(vols[-20:][np.diff(closes[-21:]) > 0])) if n >= 21 else 0.0
        dnvol = float(np.sum(vols[-20:][np.diff(closes[-21:]) < 0])) if n >= 21 else 0.0
        vol_agree = 50.0 if upvol+dnvol <= 0 else 100.0*upvol/(upvol+dnvol)
        volume_flow = 0.50*vol_agree + 0.30*obv_norm + 0.20*vwap_score

        rsi_arr = self.rsi(list(closes),14)
        _, _, hist = self.macd(list(closes),12,26,9)
        rsi = float(rsi_arr[-1]) if np.isfinite(rsi_arr[-1]) else 50.0
        hh = hist[-100:][np.isfinite(hist[-100:])]
        macd_score = 50.0 if len(hh)==0 else self._norm(float(hist[-1]),float(np.min(hh)),float(np.max(hh)))
        momentum = 0.45*rsi + 0.55*macd_score

        e20=self.ema(list(closes),20); e50=self.ema(list(closes),50)
        adx_arr, pdi, mdi=self.adx(candles,14)
        adx=float(adx_arr[-1]) if np.isfinite(adx_arr[-1]) else 0.0
        bull_trend = e20[-1] > e50[-1] and e20[-1] > e20[-4]
        bear_trend = e20[-1] < e50[-1] and e20[-1] < e20[-4]
        trend_dir = 75.0 if bull_trend else 25.0 if bear_trend else 50.0
        dmi_score = self._clamp(50+adx*0.8,50,100) if pdi[-1]>mdi[-1] else self._clamp(50-adx*0.8,0,50) if mdi[-1]>pdi[-1] else 50.0
        trend_score=0.55*trend_dir+0.45*dmi_score

        h1=list((mtf or {}).get("1h") or [])
        mtf_score=50.0
        htf_bull=htf_bear=False
        if len(h1)>=55:
            hc=[float(c.close) for c in h1]; he20=self.ema(hc,20); he50=self.ema(hc,50)
            htf_bull=hc[-1]>he20[-1]>he50[-1]; htf_bear=hc[-1]<he20[-1]<he50[-1]
            mtf_score=80.0 if htf_bull else 20.0 if htf_bear else 50.0

        chip_score=0.70*smart+0.30*(100-retail)
        smart_flow=self._clamp(chip_score*0.35+volume_flow*0.25+momentum*0.15+trend_score*0.15+mtf_score*0.10,0,100)
        flow_bull=smart_flow>=55
        flow_bear=smart_flow<=45
        structure=self._structure(candles)
        struct_bias=1 if structure=="BULL" else -1 if structure=="BEAR" else 0

        # Preserve MCDX v3 score architecture. Liquidity memory is deliberately
        # neutral here (0) because this location strategy already uses S1/R1 as
        # the execution location rather than double-counting a sweep condition.
        long_score=(30.0 if smart>=55 else 18.0 if smart>=50 else 0.0)+(25.0 if smart_flow>=60 else 15.0 if smart_flow>=52 else 0.0)+(20.0 if struct_bias==1 else 8.0 if struct_bias==0 else 0.0)+(10.0 if htf_bull else 0.0)
        short_score=(30.0 if smart<=45 else 18.0 if smart<=50 else 0.0)+(25.0 if smart_flow<=40 else 15.0 if smart_flow<=48 else 0.0)+(20.0 if struct_bias==-1 else 8.0 if struct_bias==0 else 0.0)+(10.0 if htf_bear else 0.0)

        return {
            "long_score": round(long_score,1), "short_score": round(short_score,1),
            "smart_money": round(smart,1), "smart_flow": round(smart_flow,1),
            "flow_bull": flow_bull, "flow_bear": flow_bear,
            "smart_rising": smart_rising, "smart_falling": smart_falling,
            "retail_falling": retail_falling, "retail_rising": retail_rising,
            "structure": structure, "adx": round(adx,1), "ready": True,
        }

    def _sentinel_context(self, candles: list, mtf: dict) -> dict:
        closes=[float(c.close) for c in candles]
        if len(closes)<60:
            return {"bias":"NEUTRAL","structure":"MIXED","ready":False}
        e20=self.ema(closes,20); e50=self.ema(closes,50); hma16=self.hma(closes,16)
        structure=self._structure(candles)
        bull=bool(e20[-1]>=e50[-1] and e20[-1]>e20[-4] and closes[-1]>=e20[-1])
        bear=bool(e20[-1]<=e50[-1] and e20[-1]<e20[-4] and closes[-1]<=e20[-1])
        # SME-style fast/core directional proxy: HMA slope + MACD histogram + RSI.
        _,_,hist=self.macd(closes,12,26,9); r=self.rsi(closes,14)
        fast=(float(hma16[-1]-hma16[-3]) if np.isfinite(hma16[-1]) and np.isfinite(hma16[-3]) else 0.0)
        histv=float(hist[-1]) if np.isfinite(hist[-1]) else 0.0; rsiv=float(r[-1]) if np.isfinite(r[-1]) else 50.0
        sme_bull=fast>0 and histv>=0 and rsiv>=45
        sme_bear=fast<0 and histv<=0 and rsiv<=55
        bias="BULL" if bull else "BEAR" if bear else "BALANCED"
        return {"bias":bias,"structure":structure,"sme_bull":sme_bull,"sme_bear":sme_bear,"ready":True}

    # ------------------------------------------------------------------
    # Strategy
    # ------------------------------------------------------------------
    async def analyze(self, candles: list, current_price: float, mtf_candles: dict = None) -> Signal:
        if len(candles) < 120:
            return Signal(SignalType.HOLD,self.symbol,current_price,0.0,"Sentinel warmup",metadata={"strategy":"SENTINEL","version":self.VERSION})

        atr_arr=self.atr(candles,14)
        atr=float(atr_arr[-1]) if np.isfinite(atr_arr[-1]) else 0.0
        if atr<=0:
            return Signal(SignalType.HOLD,self.symbol,current_price,0.0,"ATR unavailable",metadata={"strategy":"SENTINEL","version":self.VERSION})

        # Use the last completed bar for trigger confirmation.
        bar=candles[-1]; prev=candles[-2]
        close=float(bar.close); op=float(bar.open); low=float(bar.low); high=float(bar.high)
        sr=self._adaptive_sr(candles,mtf_candles or {},close,atr)
        s1,s2,r1,r2=sr["s1"],sr["s2"],sr["r1"],sr["r2"]
        sx=self._sentinel_context(candles,mtf_candles or {})
        mc=self._mcdx_context(candles,mtf_candles or {})

        meta={"strategy":"SENTINEL","version":self.VERSION,"s1":s1,"s2":s2,"r1":r1,"r2":r2,"sentinel_x":sx,"mcdx":mc}
        if any(v is None for v in (s1,s2,r1,r2)):
            return Signal(SignalType.HOLD,self.symbol,current_price,0.0,"Need complete S1/S2/R1/R2 structure",metadata=meta)

        location_atr=(r1-s1)/atr
        meta["location_atr"]=round(location_atr,2)
        if location_atr < self.min_location_atr:
            return Signal(SignalType.HOLD,self.symbol,current_price,0.0,f"S1-R1 room too small ({location_atr:.2f} ATR < {self.min_location_atr:.2f})",metadata=meta)

        zone=self.entry_zone_atr*atr
        long_at_s1=(low <= s1+zone and close >= s1 and close>op)
        short_at_r1=(high >= r1-zone and close <= r1 and close<op)

        # Stops are explicitly beyond S2/R2 as requested.
        long_sl=s2-self.sl_buffer_atr*atr
        long_tp=r1
        short_sl=r2+self.sl_buffer_atr*atr
        short_tp=s1
        long_risk=close-long_sl; long_reward=long_tp-close
        short_risk=short_sl-close; short_reward=close-short_tp
        long_rr=(long_reward/long_risk) if long_risk>0 else 0.0
        short_rr=(short_reward/short_risk) if short_risk>0 else 0.0
        meta.update({"long_rr":round(long_rr,2),"short_rr":round(short_rr,2),"long_sl":long_sl,"long_tp":long_tp,"short_sl":short_sl,"short_tp":short_tp})

        long_context=(sx.get("bias")!="BEAR" and sx.get("structure")!="BEAR" and (sx.get("sme_bull") or sx.get("bias")=="BULL") and mc.get("long_score",0)>=self.min_context_score and mc.get("smart_flow",50)>=52)
        short_context=(sx.get("bias")!="BULL" and sx.get("structure")!="BULL" and (sx.get("sme_bear") or sx.get("bias")=="BEAR") and mc.get("short_score",0)>=self.min_context_score and mc.get("smart_flow",50)<=48)

        if long_at_s1 and long_context and long_rr>=self.min_rr and long_reward>0:
            meta.update({"entry_location":"S1","stop_basis":"BELOW_S2","tp_basis":"R1","stop_loss":long_sl,"take_profit":long_tp,"rr_ratio":round(long_rr,2),"entry_trigger":"S1_RECLAIM"})
            return Signal(SignalType.BUY,self.symbol,current_price,0.0,f"SENTINEL LONG: S1 reclaim | TP R1 | SL below S2 | room {location_atr:.2f}ATR | RR {long_rr:.2f} | MCDX {mc.get('long_score')}",confidence=min(1.0,0.50+mc.get("long_score",0)/200.0),metadata=meta)

        if short_at_r1 and short_context and short_rr>=self.min_rr and short_reward>0:
            meta.update({"entry_location":"R1","stop_basis":"ABOVE_R2","tp_basis":"S1","stop_loss":short_sl,"take_profit":short_tp,"rr_ratio":round(short_rr,2),"entry_trigger":"R1_REJECTION"})
            return Signal(SignalType.SELL,self.symbol,current_price,0.0,f"SENTINEL SHORT: R1 rejection | TP S1 | SL above R2 | room {location_atr:.2f}ATR | RR {short_rr:.2f} | MCDX {mc.get('short_score')}",confidence=min(1.0,0.50+mc.get("short_score",0)/200.0),metadata=meta)

        reasons=[]
        if long_at_s1 and not long_context: reasons.append("S1 touched but LONG context not confirmed")
        elif short_at_r1 and not short_context: reasons.append("R1 touched but SHORT context not confirmed")
        elif long_at_s1 and long_rr<self.min_rr: reasons.append(f"LONG RR {long_rr:.2f} < {self.min_rr:.2f}")
        elif short_at_r1 and short_rr<self.min_rr: reasons.append(f"SHORT RR {short_rr:.2f} < {self.min_rr:.2f}")
        else: reasons.append("WAIT location: LONG@S1 / SHORT@R1")
        return Signal(SignalType.HOLD,self.symbol,current_price,0.0,"; ".join(reasons),metadata=meta)
