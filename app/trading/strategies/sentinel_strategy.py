"""Sentinel V1.10 — 1H S/R map + fast 15M execution + dynamic proximity.

Core rules
- 1H ONLY builds S1/S2/R1/R2 and ATR14.
- 15M ONLY executes proximity rejection / reclaim / displacement.
- LONG requires only 1H S1. S2 is preferred but is NOT required.
  If S2 exists, SL sits below S2 + buffer. If S2 is missing, SL falls back to
  S1 - configurable ATR distance. If R1 exists it remains the TP and actual RR
  must pass; if R1 does not exist, LONG is OPEN_SKY with dynamic exit.
- SHORT requires only 1H R1. R2 is preferred but is NOT required.
  If R2 exists, SL sits above R2 + buffer. If R2 is missing, SL falls back to
  R1 + configurable ATR distance. If S1 exists it remains the TP and actual RR
  must pass; if S1 does not exist, SHORT is OPEN_FLOOR with dynamic exit.
- Normal mapped proximity is 0.30 ATR while S1-R1 room is <= 3.0 ATR.
- If mapped room is > 3.0 ATR, proximity expands to room/5, capped at 1.00 ATR.
- OPEN_SKY/OPEN_FLOOR keep the dedicated 0.60 ATR proximity.
- MCDX relative dominance: LONG L>=45, L-S>10, flow>52; SHORT S>=45, S-L>10, flow<48.
"""
from __future__ import annotations

from typing import Optional
import numpy as np

from .base import BaseStrategy, Signal, SignalType


class SentinelStrategy(BaseStrategy):
    VERSION = "1.10"
    entry_tf = "15m"

    def __init__(
        self,
        symbol: str,
        params: Optional[dict] = None,
        min_context_score: float = 45.0,
        mcdx_dominance_gap: float = 10.0,
        long_flow_min: float = 52.0,
        short_flow_max: float = 48.0,
        min_location_atr: float = 1.20,
        min_rr: float = 1.50,
        entry_zone_atr: float = 0.30,
        reversal_proximity_atr: float = 0.30,
        open_ended_proximity_atr: float = 0.60,
        sl_buffer_atr: float = 0.15,
        missing_outer_sl_atr: float = 0.30,
        pivot_span: int = 3,
    ):
        super().__init__(symbol, params)
        self.min_context_score = float(min_context_score)
        self.mcdx_dominance_gap = max(0.0, float(mcdx_dominance_gap))
        self.long_flow_min = float(long_flow_min)
        self.short_flow_max = float(short_flow_max)
        self.min_location_atr = float(min_location_atr)
        self.min_rr = float(min_rr)
        self.entry_zone_atr = float(entry_zone_atr)
        self.reversal_proximity_atr = max(0.05, float(reversal_proximity_atr))
        self.open_ended_proximity_atr = max(self.reversal_proximity_atr, min(0.60, float(open_ended_proximity_atr)))
        self.sl_buffer_atr = max(0.0, float(sl_buffer_atr))
        self.missing_outer_sl_atr = max(0.10, float(missing_outer_sl_atr))
        self.pivot_span = max(2, int(pivot_span))
        self.name = f"Sentinel({symbol})"
        self._open_position: Optional[str] = None
        self._entry_price: Optional[float] = None
        self._entry_sl: Optional[float] = None
        self._open_ended: bool = False
        self._latest_15m: list = []
        self._latest_mtf: dict = {}
        self._latest_sr: dict = {}
        self._latest_sx: dict = {}
        self._latest_mc: dict = {}

    @staticmethod
    def _clamp(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    @staticmethod
    def _confirmed_pivots(candles: list, span: int) -> tuple[list[float], list[float]]:
        highs: list[float] = []; lows: list[float] = []
        if len(candles) < span * 2 + 3: return highs, lows
        for i in range(span, len(candles) - span):
            hi=float(candles[i].high); lo=float(candles[i].low)
            if hi >= max(float(c.high) for c in candles[i-span:i]) and hi >= max(float(c.high) for c in candles[i+1:i+span+1]): highs.append(hi)
            if lo <= min(float(c.low) for c in candles[i-span:i]) and lo <= min(float(c.low) for c in candles[i+1:i+span+1]): lows.append(lo)
        return highs, lows

    def _structure(self, candles: list) -> str:
        hs,ls=self._confirmed_pivots(candles[-80:],3)
        if len(hs)>=2 and len(ls)>=2:
            if hs[-1]>hs[-2] and ls[-1]>ls[-2]: return "BULL"
            if hs[-1]<hs[-2] and ls[-1]<ls[-2]: return "BEAR"
        return "MIXED"

    def _sr_map_1h(self, mtf: dict, price: float) -> dict:
        h1=list((mtf or {}).get("1h") or [])
        if len(h1)<60: return {"ready":False,"reason":"1H S/R warmup","bars_1h":len(h1)}
        atr_arr=self.atr(h1,14); atr_1h=float(atr_arr[-1]) if len(atr_arr) and np.isfinite(atr_arr[-1]) else 0.0
        if atr_1h<=0: return {"ready":False,"reason":"1H ATR unavailable","bars_1h":len(h1)}
        h1_h,h1_l=self._confirmed_pivots(h1[-180:],self.pivot_span)
        resistance_pool=sorted({float(x) for x in h1_h if float(x)>price})
        support_pool=sorted({float(x) for x in h1_l if float(x)<price},reverse=True)
        r1=resistance_pool[0] if resistance_pool else None; r2=resistance_pool[1] if len(resistance_pool)>1 else None
        s1=support_pool[0] if support_pool else None; s2=support_pool[1] if len(support_pool)>1 else None
        return {"ready":True,"s1":s1,"s2":s2,"r1":r1,"r2":r2,"atr_1h":atr_1h,"bars_1h":len(h1),"source_tf":"1H","long_map_ready":s1 is not None,"long_has_s2":s2 is not None,"short_map_ready":r1 is not None,"short_has_r2":r2 is not None,"open_sky_long":s1 is not None and r1 is None,"open_floor_short":r1 is not None and s1 is None}

    @staticmethod
    def _norm(v: float, lo: float, hi: float) -> float:
        if not np.isfinite(v) or not np.isfinite(lo) or not np.isfinite(hi) or hi==lo: return 50.0
        return max(0.0,min(100.0,(v-lo)/(hi-lo)*100.0))

    def _mcdx_context(self,candles:list,mtf:dict)->dict:
        n=len(candles)
        if n<120: return {"long_score":0.0,"short_score":0.0,"smart_flow":50.0,"ready":False}
        closes=np.asarray([float(c.close) for c in candles],dtype=float); highs=np.asarray([float(c.high) for c in candles],dtype=float); lows=np.asarray([float(c.low) for c in candles],dtype=float); vols=np.asarray([max(0.0,float(c.volume)) for c in candles],dtype=float)
        lo50=float(np.min(lows[-50:])); hi50=float(np.max(highs[-50:])); rng=max(hi50-lo50,1e-12)
        smart=self._clamp(((closes[-1]*0.96)-lo50)/rng*100.0,0,100); float_total=self._clamp(((closes[-1]*1.04)-lo50)/rng*100.0,0,100); retail=self._clamp(100.0-float_total,0,100)
        signed=np.sign(np.diff(closes,prepend=closes[0]))*vols; obv=np.cumsum(signed); obv20=obv[-20:]; std=float(np.std(obv20)); obv_z=0.0 if std<=1e-12 else (float(obv[-1])-float(np.mean(obv20)))/std; obv_norm=self._clamp((obv_z+3.0)/6.0*100.0,0,100)
        pv=closes[-80:]*vols[-80:]; vwap=float(np.sum(pv)/max(float(np.sum(vols[-80:])),1e-12)); dev=float(np.std(closes[-20:]-vwap)); vwap_score=self._clamp(50+(closes[-1]-vwap)/max(dev,1e-9)*20,0,100)
        upvol=float(np.sum(vols[-20:][np.diff(closes[-21:])>0])); dnvol=float(np.sum(vols[-20:][np.diff(closes[-21:])<0])); vol_agree=50.0 if upvol+dnvol<=0 else 100.0*upvol/(upvol+dnvol); volume_flow=0.50*vol_agree+0.30*obv_norm+0.20*vwap_score
        rsi_arr=self.rsi(list(closes),14); _,_,hist=self.macd(list(closes),12,26,9); rsi=float(rsi_arr[-1]) if np.isfinite(rsi_arr[-1]) else 50.0; hh=hist[-100:][np.isfinite(hist[-100:])]; macd_score=50.0 if len(hh)==0 else self._norm(float(hist[-1]),float(np.min(hh)),float(np.max(hh))); momentum=0.45*rsi+0.55*macd_score
        h1=list((mtf or {}).get("1h") or []); htf_bull=htf_bear=False; mtf_score=50.0
        if len(h1)>=55:
            hc=[float(c.close) for c in h1]; he20=self.ema(hc,20); he50=self.ema(hc,50); htf_bull=hc[-1]>he20[-1]>he50[-1]; htf_bear=hc[-1]<he20[-1]<he50[-1]; mtf_score=80.0 if htf_bull else 20.0 if htf_bear else 50.0
        structure=self._structure(candles); struct_bias=1 if structure=="BULL" else -1 if structure=="BEAR" else 0; e20=self.ema(list(closes),20); e50=self.ema(list(closes),50); trend_score=75.0 if e20[-1]>e50[-1] else 25.0 if e20[-1]<e50[-1] else 50.0
        chip_score=0.70*smart+0.30*(100-retail); smart_flow=self._clamp(chip_score*0.35+volume_flow*0.25+momentum*0.15+trend_score*0.15+mtf_score*0.10,0,100)
        long_score=(30.0 if smart>=55 else 18.0 if smart>=50 else 0.0)+(25.0 if smart_flow>=60 else 15.0 if smart_flow>=52 else 0.0)+(20.0 if struct_bias==1 else 8.0 if struct_bias==0 else 0.0)+(10.0 if htf_bull else 0.0)
        short_score=(30.0 if smart<=45 else 18.0 if smart<=50 else 0.0)+(25.0 if smart_flow<=40 else 15.0 if smart_flow<=48 else 0.0)+(20.0 if struct_bias==-1 else 8.0 if struct_bias==0 else 0.0)+(10.0 if htf_bear else 0.0)
        return {"long_score":round(long_score,1),"short_score":round(short_score,1),"smart_money":round(smart,1),"smart_flow":round(smart_flow,1),"structure":structure,"ready":True}

    def _sentinel_context(self,candles:list,mtf:dict)->dict:
        closes=[float(c.close) for c in candles]
        if len(closes)<60: return {"bias":"NEUTRAL","structure":"MIXED","ready":False}
        e20=self.ema(closes,20); e50=self.ema(closes,50); hma16=self.hma(closes,16); structure=self._structure(candles); bull=bool(e20[-1]>=e50[-1] and e20[-1]>e20[-4] and closes[-1]>=e20[-1]); bear=bool(e20[-1]<=e50[-1] and e20[-1]<e20[-4] and closes[-1]<=e20[-1]); _,_,hist=self.macd(closes,12,26,9); r=self.rsi(closes,14); fast=float(hma16[-1]-hma16[-3]) if np.isfinite(hma16[-1]) and np.isfinite(hma16[-3]) else 0.0; histv=float(hist[-1]) if np.isfinite(hist[-1]) else 0.0; rsiv=float(r[-1]) if np.isfinite(r[-1]) else 50.0
        return {"bias":"BULL" if bull else "BEAR" if bear else "BALANCED","structure":structure,"sme_bull":fast>0 and histv>=0 and rsiv>=45,"sme_bear":fast<0 and histv<=0 and rsiv<=55,"ready":True}

    def _entry_triggers_15m(self,candles:list,s1,r1,atr_1h:float,long_proximity_atr:Optional[float]=None,short_proximity_atr:Optional[float]=None)->dict:
        bar=candles[-1]; prev=candles[-2]; close=float(bar.close); op=float(bar.open); high=float(bar.high); low=float(bar.low); prev_close=float(prev.close); prev_high=float(prev.high); prev_low=float(prev.low)
        long_p=self.reversal_proximity_atr if long_proximity_atr is None else float(long_proximity_atr); short_p=self.reversal_proximity_atr if short_proximity_atr is None else float(short_proximity_atr); long_proximity=max(0.05,long_p)*atr_1h; short_proximity=max(0.05,short_p)*atr_1h; zone=self.entry_zone_atr*atr_1h
        long_trigger=False; long_name=""
        if s1 is not None:
            near_now=low>=s1 and low<=s1+long_proximity; rejection=near_now and close>s1+0.05*atr_1h and close>op; reclaim=(low<=s1+max(zone,long_proximity) and close>=s1 and close>op); prev_near=prev_low>=s1 and prev_low<=s1+long_proximity; displacement=prev_near and close>s1+0.20*atr_1h and close>prev_close and close>op and (close-op)>=0.20*atr_1h
            if rejection: long_trigger=True; long_name="1H_S1__15M_REJECTION"
            elif displacement: long_trigger=True; long_name="1H_S1__15M_DISPLACEMENT"
            elif reclaim: long_trigger=True; long_name="1H_S1__15M_RECLAIM"
        short_trigger=False; short_name=""
        if r1 is not None:
            near_now=high<=r1 and high>=r1-short_proximity; rejection=near_now and close<r1-0.05*atr_1h and close<op; reclaim=(high>=r1-max(zone,short_proximity) and close<=r1 and close<op); prev_near=prev_high<=r1 and prev_high>=r1-short_proximity; displacement=prev_near and close<r1-0.20*atr_1h and close<prev_close and close<op and (op-close)>=0.20*atr_1h
            if rejection: short_trigger=True; short_name="1H_R1__15M_REJECTION"
            elif displacement: short_trigger=True; short_name="1H_R1__15M_DISPLACEMENT"
            elif reclaim: short_trigger=True; short_name="1H_R1__15M_RECLAIM"
        return {"long":long_trigger,"long_name":long_name,"short":short_trigger,"short_name":short_name,"long_proximity_atr":round(long_p,2),"short_proximity_atr":round(short_p,2)}

    def _mapped_proximity_atr(self,s1,r1,atr_1h:float)->tuple[float,Optional[float]]:
        if s1 is None or r1 is None or atr_1h<=0: return self.reversal_proximity_atr,None
        room_atr=(float(r1)-float(s1))/atr_1h
        if room_atr>3.0: return min(room_atr/5.0,1.0),room_atr
        return self.reversal_proximity_atr,room_atr

    async def analyze(self,candles:list,current_price:float,mtf_candles:dict=None)->Signal:
        if len(candles)<120: return Signal(SignalType.HOLD,self.symbol,current_price,0.0,"15M Sentinel warmup",metadata={"strategy":"SENTINEL","version":self.VERSION})
        mtf=mtf_candles or {}; self._latest_15m=list(candles); self._latest_mtf=mtf; close=float(candles[-1].close); sr=self._sr_map_1h(mtf,close); meta={"strategy":"SENTINEL","version":self.VERSION,"sr_tf":"1H","entry_tf":"15M"}; meta.update({k:v for k,v in sr.items() if k!="ready"})
        if not sr.get("ready"): return Signal(SignalType.HOLD,self.symbol,current_price,0.0,sr.get("reason","1H map unavailable"),metadata=meta)
        self._latest_sr=sr; atr_1h=float(sr["atr_1h"]); s1,s2,r1,r2=sr.get("s1"),sr.get("s2"),sr.get("r1"),sr.get("r2"); sx=self._sentinel_context(candles,mtf); mc=self._mcdx_context(candles,mtf); self._latest_sx=sx; self._latest_mc=mc; meta.update({"sentinel_x":sx,"mcdx":mc,"atr_1h":round(atr_1h,8)})
        if self._open_position is not None: return Signal(SignalType.HOLD,self.symbol,current_price,0.0,f"Managing {self._open_position.upper()} | open_ended={self._open_ended}",metadata=meta)

        mapped_prox,room_for_prox=self._mapped_proximity_atr(s1,r1,atr_1h)
        long_prox=self.open_ended_proximity_atr if sr.get("open_sky_long") else mapped_prox
        short_prox=self.open_ended_proximity_atr if sr.get("open_floor_short") else mapped_prox
        proximity_mode="OPEN_ENDED_0.60" if (sr.get("open_sky_long") or sr.get("open_floor_short")) else ("WIDE_ROOM_1_5" if room_for_prox is not None and room_for_prox>3.0 else "NORMAL_0.30")
        trg=self._entry_triggers_15m(candles,s1,r1,atr_1h,long_proximity_atr=long_prox,short_proximity_atr=short_prox)
        meta.update({"long_entry_proximity_atr":round(long_prox,2),"short_entry_proximity_atr":round(short_prox,2),"proximity_mode":proximity_mode,"proximity_room_atr":round(room_for_prox,2) if room_for_prox is not None else None})

        long_score=float(mc.get("long_score",0) or 0); short_score=float(mc.get("short_score",0) or 0); smart_flow=float(mc.get("smart_flow",50) or 50); long_gap=long_score-short_score; short_gap=short_score-long_score; mcdx_long_pass=bool(long_score>=self.min_context_score and long_gap>self.mcdx_dominance_gap and smart_flow>self.long_flow_min); mcdx_short_pass=bool(short_score>=self.min_context_score and short_gap>self.mcdx_dominance_gap and smart_flow<self.short_flow_max)
        meta["mcdx_gate"]={"long_pass":mcdx_long_pass,"short_pass":mcdx_short_pass,"long_score":round(long_score,1),"short_score":round(short_score,1),"long_gap":round(long_gap,1),"short_gap":round(short_gap,1),"min_score_gte":self.min_context_score,"dominance_gap_strict_gt":self.mcdx_dominance_gap,"long_flow_strict_gt":self.long_flow_min,"short_flow_strict_lt":self.short_flow_max}
        long_context=(sx.get("bias")!="BEAR" and sx.get("structure")!="BEAR" and (sx.get("sme_bull") or sx.get("bias")=="BULL") and mcdx_long_pass); short_context=(sx.get("bias")!="BULL" and sx.get("structure")!="BULL" and (sx.get("sme_bear") or sx.get("bias")=="BEAR") and mcdx_short_pass)

        if sr.get("long_map_ready") and trg["long"] and long_context:
            if s2 is not None: long_sl=float(s2)-self.sl_buffer_atr*atr_1h; long_stop_basis="BELOW_1H_S2"
            else: long_sl=float(s1)-self.missing_outer_sl_atr*atr_1h; long_stop_basis=f"S1_MINUS_{self.missing_outer_sl_atr:.2f}ATR1H"
            long_risk=close-long_sl; meta.update({"planned_long_sl":round(long_sl,8),"long_stop_basis":long_stop_basis})
            if long_risk>0:
                if r1 is None:
                    self._open_position="long"; self._entry_price=close; self._entry_sl=long_sl; self._open_ended=True; meta.update({"entry_location":"1H_S1","entry_trigger":trg["long_name"],"stop_loss":long_sl,"take_profit":None,"open_ended_tp":True,"tp_basis":"DYNAMIC_R1_OR_STRUCTURE_EXIT","stop_basis":long_stop_basis,"room_mode":"OPEN_SKY","entry_proximity_atr":round(long_prox,2)}); return Signal(SignalType.BUY,self.symbol,current_price,0.0,f"SENTINEL LONG {trg['long_name']} | OPEN_SKY | prox {long_prox:.2f}ATR | SL {long_stop_basis} | MCDX L={long_score:.1f} S={short_score:.1f} Δ={long_gap:.1f} flow={smart_flow:.1f}",confidence=min(1.0,0.50+long_score/200.0),metadata=meta)
                location_atr=(float(r1)-float(s1))/atr_1h; long_reward=float(r1)-close; long_rr=long_reward/long_risk if long_risk>0 else 0.0; meta.update({"location_atr":round(location_atr,2),"long_rr":round(long_rr,2),"planned_long_tp":float(r1)})
                if location_atr>=self.min_location_atr and long_reward>0 and long_rr>=self.min_rr:
                    self._open_position="long"; self._entry_price=close; self._entry_sl=long_sl; self._open_ended=False; meta.update({"entry_location":"1H_S1","entry_trigger":trg["long_name"],"stop_loss":long_sl,"take_profit":float(r1),"open_ended_tp":False,"tp_basis":"1H_R1","stop_basis":long_stop_basis,"rr_ratio":round(long_rr,2),"room_mode":"FIXED_R1","entry_proximity_atr":round(long_prox,2)}); return Signal(SignalType.BUY,self.symbol,current_price,0.0,f"SENTINEL LONG {trg['long_name']} | room {location_atr:.2f}ATR | prox {long_prox:.2f}ATR | RR {long_rr:.2f} | MCDX L={long_score:.1f} S={short_score:.1f} Δ={long_gap:.1f} flow={smart_flow:.1f}",confidence=min(1.0,0.50+long_score/200.0),metadata=meta)

        if sr.get("short_map_ready") and trg["short"] and short_context:
            if r2 is not None: short_sl=float(r2)+self.sl_buffer_atr*atr_1h; short_stop_basis="ABOVE_1H_R2"
            else: short_sl=float(r1)+self.missing_outer_sl_atr*atr_1h; short_stop_basis=f"R1_PLUS_{self.missing_outer_sl_atr:.2f}ATR1H"
            short_risk=short_sl-close; meta.update({"planned_short_sl":round(short_sl,8),"short_stop_basis":short_stop_basis})
            if short_risk>0:
                if s1 is None:
                    self._open_position="short"; self._entry_price=close; self._entry_sl=short_sl; self._open_ended=True; meta.update({"entry_location":"1H_R1","entry_trigger":trg["short_name"],"stop_loss":short_sl,"take_profit":None,"open_ended_tp":True,"tp_basis":"DYNAMIC_S1_OR_STRUCTURE_EXIT","stop_basis":short_stop_basis,"room_mode":"OPEN_FLOOR","entry_proximity_atr":round(short_prox,2)}); return Signal(SignalType.SELL,self.symbol,current_price,0.0,f"SENTINEL SHORT {trg['short_name']} | OPEN_FLOOR | prox {short_prox:.2f}ATR | SL {short_stop_basis} | MCDX S={short_score:.1f} L={long_score:.1f} Δ={short_gap:.1f} flow={smart_flow:.1f}",confidence=min(1.0,0.50+short_score/200.0),metadata=meta)
                location_atr=(float(r1)-float(s1))/atr_1h; short_reward=close-float(s1); short_rr=short_reward/short_risk if short_risk>0 else 0.0; meta.update({"location_atr":round(location_atr,2),"short_rr":round(short_rr,2),"planned_short_tp":float(s1)})
                if location_atr>=self.min_location_atr and short_reward>0 and short_rr>=self.min_rr:
                    self._open_position="short"; self._entry_price=close; self._entry_sl=short_sl; self._open_ended=False; meta.update({"entry_location":"1H_R1","entry_trigger":trg["short_name"],"stop_loss":short_sl,"take_profit":float(s1),"open_ended_tp":False,"tp_basis":"1H_S1","stop_basis":short_stop_basis,"rr_ratio":round(short_rr,2),"room_mode":"FIXED_S1","entry_proximity_atr":round(short_prox,2)}); return Signal(SignalType.SELL,self.symbol,current_price,0.0,f"SENTINEL SHORT {trg['short_name']} | room {location_atr:.2f}ATR | prox {short_prox:.2f}ATR | RR {short_rr:.2f} | MCDX S={short_score:.1f} L={long_score:.1f} Δ={short_gap:.1f} flow={smart_flow:.1f}",confidence=min(1.0,0.50+short_score/200.0),metadata=meta)

        reasons=[]
        if not sr.get("long_map_ready"): reasons.append("LONG needs 1H S1")
        elif s2 is None: reasons.append(f"LONG S1-only ready; fallback SL=S1-{self.missing_outer_sl_atr:.2f}ATR1H")
        if not sr.get("short_map_ready"): reasons.append("SHORT needs 1H R1")
        elif r2 is None: reasons.append(f"SHORT R1-only ready; fallback SL=R1+{self.missing_outer_sl_atr:.2f}ATR1H")
        if room_for_prox is not None: reasons.append(f"PROX {proximity_mode}: room={room_for_prox:.2f}ATR -> zone={mapped_prox:.2f}ATR")
        if sr.get("open_sky_long"): reasons.append(f"LONG OPEN_SKY armed zone <= {long_prox:.2f} ATR1H from S1")
        if sr.get("open_floor_short"): reasons.append(f"SHORT OPEN_FLOOR armed zone <= {short_prox:.2f} ATR1H from R1")
        if not mcdx_long_pass and not mcdx_short_pass: reasons.append(f"MCDX wait L={long_score:.1f} S={short_score:.1f} ΔL={long_gap:.1f} ΔS={short_gap:.1f} flow={smart_flow:.1f}")
        if not reasons: reasons.append("WAIT 15M trigger/context at 1H S1/R1")
        return Signal(SignalType.HOLD,self.symbol,current_price,0.0,"; ".join(reasons),metadata=meta)

    def tick_open_position(self,current_price:float,position_key:Optional[str]=None):
        if self._open_position is None: return None
        from ..engines.position_manager import PositionUpdate
        if not self._open_ended: return PositionUpdate(action="hold",reason=f"Sentinel {self._open_position.upper()} fixed 1H target active")
        sr=self._latest_sr or {}; atr_1h=float(sr.get("atr_1h") or 0.0); candles=self._latest_15m or []
        if atr_1h<=0 or len(candles)<3: return PositionUpdate(action="hold",reason="Sentinel open-ended runner: waiting refreshed 1H/15M map")
        structure=self._structure(candles); bar=candles[-1]; prev=candles[-2]; close=float(bar.close); op=float(bar.open); high=float(bar.high); low=float(bar.low); prev_close=float(prev.close); proximity=self.reversal_proximity_atr*atr_1h
        if self._open_position=="long":
            r1=sr.get("r1")
            if r1 is None: return PositionUpdate(action="hold",reason="LONG OPEN_SKY: no 1H R1 yet — keep runner")
            near_r=high>=float(r1)-proximity; bearish_reversal=near_r and close<float(r1)-0.05*atr_1h and close<op and close<prev_close
            if bearish_reversal or structure=="BEAR":
                reason=(f"OPEN_SKY exit: 1H R1 formed at {float(r1):.6f} + 15M bearish reversal" if bearish_reversal else f"OPEN_SKY exit: 1H R1 formed at {float(r1):.6f} + 15M structure BEAR"); self._reset_position_state(); return PositionUpdate(action="close",close_pct=1.0,reason=reason)
            return PositionUpdate(action="hold",reason=f"LONG OPEN_SKY: 1H R1={float(r1):.6f} formed; waiting 15M reversal/BEAR structure")
        s1=sr.get("s1")
        if s1 is None: return PositionUpdate(action="hold",reason="SHORT OPEN_FLOOR: no 1H S1 yet — keep runner")
        near_s=low<=float(s1)+proximity; bullish_reversal=near_s and close>float(s1)+0.05*atr_1h and close>op and close>prev_close
        if bullish_reversal or structure=="BULL":
            reason=(f"OPEN_FLOOR exit: 1H S1 formed at {float(s1):.6f} + 15M bullish reversal" if bullish_reversal else f"OPEN_FLOOR exit: 1H S1 formed at {float(s1):.6f} + 15M structure BULL"); self._reset_position_state(); return PositionUpdate(action="close",close_pct=1.0,reason=reason)
        return PositionUpdate(action="hold",reason=f"SHORT OPEN_FLOOR: 1H S1={float(s1):.6f} formed; waiting 15M reversal/BULL structure")

    def attach_existing_position(self,direction:str,entry_price:float,stop_loss:Optional[float]=None,take_profit:Optional[float]=None)->None:
        self._open_position=str(direction).lower(); self._entry_price=float(entry_price); self._entry_sl=float(stop_loss) if stop_loss is not None else None; self._open_ended=take_profit is None
    def cancel_pending_entry(self,reason:str="")->None: self._reset_position_state()
    def _reset_position_state(self)->None:
        self._open_position=None; self._entry_price=None; self._entry_sl=None; self._open_ended=False
