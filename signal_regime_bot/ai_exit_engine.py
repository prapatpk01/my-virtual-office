"""Stateful multi-factor AI Exit Engine.

A fast adverse move starts WATCH; it does not close a position by itself.
Normal close requires persistent multi-factor confirmation from EMA, structure,
momentum and candle/volume evidence. Only a true near-stop/extreme acceleration
condition can bypass confirmation as EMERGENCY.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict
import numpy as np
import pandas as pd

import indicators as ind
from config import Config

HOLD = "HOLD"
WATCH = "WATCH"
CLOSE = "CLOSE"
EMERGENCY = "EMERGENCY"

@dataclass
class ExitDecision:
    action: str
    score: float
    threshold: float
    adverse_r: float
    reason: str
    signals: Dict[str, float] = field(default_factory=dict)
    confirmations: int = 0

@dataclass
class _WatchState:
    last_bar_ts: object = None
    confirmed_bars: int = 0
    peak_score: float = 0.0

class AIExitEngine:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._state: dict[str, _WatchState] = {}

    def clear(self, symbol: str) -> None:
        self._state.pop(symbol, None)

    @staticmethod
    def _bar(df: pd.DataFrame, atr_v: float):
        o, h, l, c = [float(df[x].iloc[-1]) for x in ("open", "high", "low", "close")]
        rng = max(h-l, 1e-12)
        body = abs(c-o)
        direction = 1 if c >= o else -1
        close_frac = (c-l)/rng if direction > 0 else (h-c)/rng
        vol = float(df["volume"].iloc[-1])
        vma = float(df["volume"].iloc[-21:-1].mean()) if len(df) >= 21 else 0.0
        return direction, rng/max(atr_v,1e-12), body/rng, close_frac, (vol/vma if vma>0 else 1.0)

    def evaluate(self, symbol: str, pos, df_5m: pd.DataFrame, df_15m: pd.DataFrame,
                 current_price: float) -> ExitDecision:
        c = self.cfg
        if not getattr(c, "ai_exit_enabled", True) or pos.one_r <= 0 or current_price <= 0:
            return ExitDecision(HOLD,0,100,0,"disabled/invalid")
        if df_5m is None or df_15m is None or len(df_5m)<55 or len(df_15m)<55:
            return ExitDecision(HOLD,0,100,0,"insufficient history")

        is_long = pos.side == "long"
        against = -1 if is_long else 1
        adverse = (pos.entry_price-current_price) if is_long else (current_price-pos.entry_price)
        adverse_r = adverse/max(pos.one_r,1e-12)

        atr5 = float(ind.atr(df_5m,14).iloc[-1]); atr15 = float(ind.atr(df_15m,14).iloc[-1])
        if not np.isfinite(atr5) or atr5<=0 or not np.isfinite(atr15) or atr15<=0:
            return ExitDecision(HOLD,0,100,adverse_r,"invalid ATR")

        # True emergency only: close-to-stop or extreme live acceleration + meaningful loss.
        last5 = float(df_5m["close"].iloc[-1])
        live_move = ((last5-current_price) if is_long else (current_price-last5))/atr5
        emergency = (
            adverse_r >= getattr(c,"ai_exit_emergency_adverse_r",0.82)
            and live_move >= getattr(c,"ai_exit_emergency_live_atr",2.8)
        ) or adverse_r >= getattr(c,"ai_exit_absolute_emergency_r",0.94)
        if emergency:
            return ExitDecision(EMERGENCY,100,100,adverse_r,
                f"true emergency: live={live_move:.2f}ATR adverse={adverse_r:.2f}R", {"emergency":100}, 1)

        signals: Dict[str,float] = {}
        confirmations = 0
        d5, range5, body5, cf5, vol5 = self._bar(df_5m,atr5)
        d15, range15, body15, cf15, vol15 = self._bar(df_15m,atr15)

        if d5==against and range5>=1.25 and body5>=0.55 and cf5>=0.68:
            signals["5m_reversal_candle"] = 15; confirmations += 1
        if d15==against and range15>=1.10 and body15>=0.52 and cf15>=0.65:
            signals["15m_reversal_candle"] = 20; confirmations += 1

        e8_5 = float(ind.ema(df_5m["close"],8).iloc[-1]); e13_5 = float(ind.ema(df_5m["close"],13).iloc[-1])
        e20_15 = float(ind.ema(df_15m["close"],20).iloc[-1])
        ema_invalid = (is_long and e8_5<e13_5 and current_price<e13_5) or ((not is_long) and e8_5>e13_5 and current_price>e13_5)
        if ema_invalid:
            signals["ema8_13_invalid"] = 20; confirmations += 1
        htf_ema_invalid = (is_long and current_price<e20_15) or ((not is_long) and current_price>e20_15)
        if htf_ema_invalid:
            signals["15m_ema20_invalid"] = 10

        look = max(4, getattr(c,"ai_exit_structure_lookback",6))
        prev_low5 = float(df_5m["low"].iloc[-look-1:-1].min()); prev_high5 = float(df_5m["high"].iloc[-look-1:-1].max())
        struct5 = (is_long and float(df_5m["close"].iloc[-1])<prev_low5) or ((not is_long) and float(df_5m["close"].iloc[-1])>prev_high5)
        if struct5:
            signals["5m_structure_break"] = 20; confirmations += 1
        prev_low15 = float(df_15m["low"].iloc[-look-1:-1].min()); prev_high15 = float(df_15m["high"].iloc[-look-1:-1].max())
        struct15 = (is_long and float(df_15m["close"].iloc[-1])<prev_low15) or ((not is_long) and float(df_15m["close"].iloc[-1])>prev_high15)
        if struct15:
            signals["15m_structure_break"] = 30; confirmations += 1

        _,_,hist5 = ind.macd(df_5m["close"]); roc5 = ind.roc(df_5m["close"],9)
        momentum_bad = (is_long and float(hist5.iloc[-1])<0 and float(roc5.iloc[-1])<0) or ((not is_long) and float(hist5.iloc[-1])>0 and float(roc5.iloc[-1])>0)
        if momentum_bad:
            signals["momentum_flip"] = 15; confirmations += 1
        if live_move >= getattr(c,"ai_exit_watch_live_atr",1.8):
            signals["live_acceleration"] = 10  # watch evidence, never sufficient alone
        if max(vol5,vol15)>=getattr(c,"ai_exit_volume_ratio",1.8):
            signals["volume_expansion"] = 5

        score = min(100.0,sum(signals.values()))
        base = getattr(c,"ai_exit_close_score",70.0)
        regime_name = str(getattr(pos,"regime_at_entry","")).upper()
        strong = "STRONG" in regime_name
        high_entry = float(getattr(pos,"entry_score",0)) >= 80
        threshold = base + (8 if strong else 0) + (4 if high_entry else 0)
        required_conf = getattr(c,"ai_exit_confirmations",2) + (1 if strong else 0)
        min_adverse = getattr(c,"ai_exit_min_adverse_r",0.30)

        # Grace period protects fresh positions from one-bar noise.
        bars_held = 99
        if getattr(pos,"entry_bar_ts",None) is not None:
            try:
                bars_held = int((pd.Timestamp(df_5m.index[-1])-pd.Timestamp(pos.entry_bar_ts)).total_seconds()//300)
            except Exception:
                pass
        if bars_held < getattr(c,"ai_exit_grace_bars",2):
            return ExitDecision(HOLD,score,threshold,adverse_r,f"grace {bars_held}/{c.ai_exit_grace_bars} bars",signals,confirmations)

        meaningful = score >= getattr(c,"ai_exit_watch_score",45.0) or live_move >= getattr(c,"ai_exit_watch_live_atr",1.8)
        st = self._state.setdefault(symbol,_WatchState())
        bar_ts = df_5m.index[-1]
        if bar_ts != st.last_bar_ts:
            st.last_bar_ts = bar_ts
            if meaningful and confirmations >= 1:
                st.confirmed_bars += 1
            else:
                st.confirmed_bars = 0
            st.peak_score = max(st.peak_score,score) if meaningful else 0.0

        persistence = getattr(c,"ai_exit_persistence_bars",2)
        close_ok = (adverse_r>=min_adverse and score>=threshold and confirmations>=required_conf
                    and st.confirmed_bars>=persistence)
        if close_ok:
            reason = f"confirmed exit score={score:.0f}/{threshold:.0f}, confirms={confirmations}/{required_conf}, persistence={st.confirmed_bars}, adverse={adverse_r:.2f}R; " + ", ".join(signals)
            return ExitDecision(CLOSE,score,threshold,adverse_r,reason,signals,confirmations)
        if meaningful:
            return ExitDecision(WATCH,score,threshold,adverse_r,
                f"watch score={score:.0f}/{threshold:.0f}, confirms={confirmations}/{required_conf}, persistence={st.confirmed_bars}/{persistence}, adverse={adverse_r:.2f}R",signals,confirmations)
        return ExitDecision(HOLD,score,threshold,adverse_r,f"hold score={score:.0f}, adverse={adverse_r:.2f}R",signals,confirmations)
