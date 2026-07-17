"""Market Structure Engine (spec §8) — HH/HL vs LH/LL, BOS, CHOCH, False BOS.

Close-confirmed only. BOS requires body_atr >= cfg.bos_min_body_atr; volume
adds quality but is never a hard gate here.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .config import Config
from .enums import StructureState
from .indicator_engine import EPS
from .models import StructureEvent, StructureView, SwingPoint
from .swing_engine import swings_of


class StructureEngine:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def evaluate(self, candles: list, swings: list, indicators) -> StructureView:
        tf = swings[0].timeframe if swings else "?"
        view = StructureView(timeframe=tf, state=StructureState.RANGE.value, swings=swings)
        if len(candles) < 30 or indicators is None:
            return view

        closes = np.array([c.close for c in candles])
        atr_arr = getattr(indicators, "atr", None)
        atr_v = float(atr_arr[-1]) if atr_arr is not None and np.isfinite(atr_arr[-1]) else 0.0
        atr_v = max(atr_v, EPS)
        vols = np.array([c.volume for c in candles])
        vol_sma = float(np.mean(vols[-21:-1])) if len(vols) >= 21 else 0.0

        highs_s = swings_of(swings, "high")
        lows_s = swings_of(swings, "low")

        # ── base state from swing sequence ──────────────────────────────────
        state = StructureState.RANGE
        if len(highs_s) >= 2 and len(lows_s) >= 2:
            hh = highs_s[-1].price > highs_s[-2].price
            hl = lows_s[-1].price > lows_s[-2].price
            lh = highs_s[-1].price < highs_s[-2].price
            ll = lows_s[-1].price < lows_s[-2].price
            if hh and hl:
                state = StructureState.BULL
            elif lh and ll:
                state = StructureState.BEAR
            elif (hh and ll) or (lh and hl):
                state = StructureState.TRANSITION

        # ── events on the LAST closed bar ────────────────────────────────────
        events: list = []
        last = candles[-1]
        body_atr = last.body / atr_v
        vol_ratio = last.volume / max(vol_sma, EPS) if vol_sma > 0 else 1.0
        now_ts = int(last.timestamp)

        def _mk(event_type: str, direction: str, level: float) -> StructureEvent:
            disp = min(body_atr / max(self.cfg.bos_min_body_atr, EPS), 3.0) / 3.0
            return StructureEvent(timeframe=tf, event_type=event_type, direction=direction,
                                  level=float(level), confirmed_at=now_ts,
                                  body_atr=body_atr, volume_ratio=vol_ratio,
                                  displacement_quality=disp)

        # BOS: close beyond the latest confirmed swing (confirmed BEFORE this bar)
        usable_highs = [s for s in highs_s if s.confirmed_at < now_ts]
        usable_lows = [s for s in lows_s if s.confirmed_at < now_ts]
        bos_long = (usable_highs and last.close > usable_highs[-1].price
                    and body_atr >= self.cfg.bos_min_body_atr
                    and closes[-2] <= usable_highs[-1].price)
        bos_short = (usable_lows and last.close < usable_lows[-1].price
                     and body_atr >= self.cfg.bos_min_body_atr
                     and closes[-2] >= usable_lows[-1].price)
        if bos_long:
            events.append(_mk("BOS", "LONG", usable_highs[-1].price))
        if bos_short:
            events.append(_mk("BOS", "SHORT", usable_lows[-1].price))

        # CHOCH: prior structure bearish (LH/LL) and close > latest significant LH
        if len(highs_s) >= 2 and len(lows_s) >= 2:
            was_bear = highs_s[-2].price > highs_s[-1].price or lows_s[-2].price > lows_s[-1].price
            was_bull = highs_s[-2].price < highs_s[-1].price or lows_s[-2].price < lows_s[-1].price
            lh_level = usable_highs[-1].price if usable_highs else None
            hl_level = usable_lows[-1].price if usable_lows else None
            if was_bear and lh_level is not None and last.close > lh_level \
                    and closes[-2] <= lh_level and state in (StructureState.BEAR, StructureState.TRANSITION):
                events.append(_mk("CHOCH", "LONG", lh_level))
            if was_bull and hl_level is not None and last.close < hl_level \
                    and closes[-2] >= hl_level and state in (StructureState.BULL, StructureState.TRANSITION):
                events.append(_mk("CHOCH", "SHORT", hl_level))

        # False BOS: a breakout close that fell back inside within 1-2 bars
        false_bos = None
        lb = self.cfg.false_bos_lookback
        if usable_highs and len(closes) >= lb + 2:
            lvl = usable_highs[-1].price
            for k in range(2, 2 + lb):
                if k + 1 <= len(closes) and closes[-k - 1] > lvl >= closes[-1]:
                    false_bos = _mk("FALSE_BOS", "SHORT", lvl)
                    break
        if false_bos is None and usable_lows and len(closes) >= lb + 2:
            lvl = usable_lows[-1].price
            for k in range(2, 2 + lb):
                if k + 1 <= len(closes) and closes[-k - 1] < lvl <= closes[-1]:
                    false_bos = _mk("FALSE_BOS", "LONG", lvl)
                    break
        if false_bos is not None:
            events.append(false_bos)

        # ── historical BOS/CHOCH scan (for last_bos/last_choch context) ─────
        hist_events = self._historic_events(candles, swings, atr_v, tf)
        all_events = hist_events + [e for e in events if e.event_type != "FALSE_BOS"]
        boss = [e for e in all_events if e.event_type == "BOS"]
        chochs = [e for e in all_events if e.event_type == "CHOCH"]

        # ── strength upgrade + quality score ─────────────────────────────────
        quality = 0.0
        if state == StructureState.BULL:
            quality += 30
        elif state == StructureState.BEAR:
            quality += 30
        if boss:
            e = boss[-1]
            quality += 15 * e.displacement_quality
            if e.volume_ratio >= 1.2:
                quality += 10
            if e.held_after_break:
                quality += 10
        if chochs and chochs[-1].confirmed_at == now_ts:
            quality += 10
        if false_bos is None:
            quality += 15
        if len(highs_s) >= 3 and len(lows_s) >= 3:
            seq_bull = highs_s[-1].price > highs_s[-2].price > highs_s[-3].price
            seq_bear = lows_s[-1].price < lows_s[-2].price < lows_s[-3].price
            if seq_bull or seq_bear:
                quality += 10

        adx_arr = getattr(indicators, "adx", None)
        adx_v = float(adx_arr[-1]) if adx_arr is not None and np.isfinite(adx_arr[-1]) else 0.0
        if state == StructureState.BULL and adx_v >= self.cfg.strong_adx and quality >= 60:
            state = StructureState.STRONG_BULL
        if state == StructureState.BEAR and adx_v >= self.cfg.strong_adx and quality >= 60:
            state = StructureState.STRONG_BEAR

        view.state = state.value
        view.events = all_events[-10:]
        view.last_bos = boss[-1] if boss else None
        view.last_choch = chochs[-1] if chochs else None
        view.last_false_bos = false_bos
        view.quality = min(quality, 100.0)
        return view

    def _historic_events(self, candles: list, swings: list, atr_v: float, tf: str) -> list:
        """Rebuild recent BOS/CHOCH events by replaying closes vs confirmed
        swings (bounded window — enough for 'recent opposite CHOCH' checks)."""
        out: list = []
        closes = [c.close for c in candles]
        ts = [int(c.timestamp) for c in candles]
        window = min(len(candles), 120)
        start = len(candles) - window
        highs = swings_of(swings, "high")
        lows = swings_of(swings, "low")
        for i in range(max(start, 2), len(candles) - 1):   # exclude last bar (handled live)
            t = ts[i]
            uh = [s for s in highs if s.confirmed_at < t]
            ul = [s for s in lows if s.confirmed_at < t]
            body = abs(candles[i].close - candles[i].open) / max(atr_v, EPS)
            if uh and closes[i] > uh[-1].price and closes[i - 1] <= uh[-1].price \
                    and body >= self.cfg.bos_min_body_atr:
                held = i + 1 < len(closes) and closes[i + 1] > uh[-1].price
                out.append(StructureEvent(tf, "BOS", "LONG", uh[-1].price, t,
                                          body_atr=body, held_after_break=held))
                # CHOCH heuristic: BOS through a lower-high after a bear leg
                if len(uh) >= 2 and uh[-1].price < uh[-2].price:
                    out.append(StructureEvent(tf, "CHOCH", "LONG", uh[-1].price, t, body_atr=body))
            if ul and closes[i] < ul[-1].price and closes[i - 1] >= ul[-1].price \
                    and body >= self.cfg.bos_min_body_atr:
                held = i + 1 < len(closes) and closes[i + 1] < ul[-1].price
                out.append(StructureEvent(tf, "BOS", "SHORT", ul[-1].price, t,
                                          body_atr=body, held_after_break=held))
                if len(ul) >= 2 and ul[-1].price > ul[-2].price:
                    out.append(StructureEvent(tf, "CHOCH", "SHORT", ul[-1].price, t, body_atr=body))
        return out[-20:]
