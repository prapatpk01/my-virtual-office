"""Fast Pullback Entry Engine (spec §18).

Enter NEAR the zone, fast: any single sufficient trigger at a valid
location fires — never a stack of confirmations. Hard gates are only the
spec-mandated ones; everything else scores.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .config import Config, TF_MS
from .enums import (ConflictLevel, PullbackType, ReasonCode, RegimeState,
                    SetupType, StructureState, SymbolStatus)
from .indicator_engine import EPS, EntryIndicators
from .models import SignalCandidate, StructureView, SymbolState
from .support_resistance_engine import nearest_opposing_zone, zones_at_price


class PullbackEngine:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.last_block: dict = {}   # symbol -> {direction: why} — view-log diagnostics

    # ── public ───────────────────────────────────────────────────────────────
    def evaluate(self, symbol: str, state: SymbolState,
                 indicators_15m: Optional[EntryIndicators], bias, macro_context,
                 regime, structure_15m: StructureView, structure_1h: StructureView,
                 zones: list, patterns: list, candle_context, liquidity_context,
                 sd_zones: list) -> Optional[SignalCandidate]:
        if indicators_15m is None:
            return None
        for direction in ("LONG", "SHORT"):
            cand = self._evaluate_side(direction, symbol, state, indicators_15m, bias,
                                       macro_context, regime, structure_15m, structure_1h,
                                       zones, patterns, candle_context, liquidity_context,
                                       sd_zones)
            if cand is not None:
                return cand
        return None

    # ── core ─────────────────────────────────────────────────────────────────
    def _evaluate_side(self, direction, symbol, state, i, bias, macro, regime,
                       s15, s1h, zones, patterns, candle_ctx, liq, sd_zones):
        c = self.cfg

        def blk(why: str):
            self.last_block.setdefault(symbol, {})[direction] = why
            return None
        long = direction == "LONG"
        price = i.val(i.closes)
        a = i.last_atr or EPS
        now_ts = int(i.timestamps[-1])
        bar_ms = TF_MS[c.entry_timeframe]

        # ── regime gate (hard: chop / wrong regime) ──────────────────────────
        if regime.is_chop:
            return blk("CHOP")
        if not regime.allows_pullback(direction):
            return blk(f"regime={regime.confirmed_regime}")

        # ── bias soft-mode gate ──────────────────────────────────────────────
        allowed, bias_risk_mod, bias_why = bias.allows(
            direction, s15, zones, price, s1h, now_ts, TF_MS[c.bias_timeframe])
        if not allowed:
            return blk(f"bias:{bias_why}")

        # ── 4H conflict ──────────────────────────────────────────────────────
        conflict = macro.conflict_for(direction, price, zones, self._macro_view(macro, s1h))
        if conflict == ConflictLevel.STRONG:
            return blk("4H_STRONG_CONFLICT")
        # 1H confirmed opposite CHOCH (hard)
        if s1h.recent_choch_against(direction, TF_MS[c.bias_timeframe] * 8, now_ts):
            return blk("1H_OPP_CHOCH")

        # ── prior context (A structure / B displacement / C momentum) ───────
        prior_ok, prior_detail = self._prior_context(direction, i, s15)
        if not prior_ok:
            return blk("no_prior_context")

        # ── location ─────────────────────────────────────────────────────────
        loc = self._location(direction, i, zones, sd_zones, liq, s15, patterns)
        if loc is None:
            return blk("waiting_location")
        pb_type, zone, zone_score, loc_score = loc

        # deep pullback allowance check
        hf, hs = i.val(i.hma_fast), i.val(i.hma_slow)
        deep = (hf < hs) if long else (hf > hs)
        if deep:
            ok_deep = (
                (s1h.state not in ((StructureState.BEAR.value, StructureState.STRONG_BEAR.value)
                                   if long else
                                   (StructureState.BULL.value, StructureState.STRONG_BULL.value)))
                and zone is not None and zone.timeframe in ("1h", "4h")
                and (s15.last_choch is not None and s15.last_choch.direction == direction
                     or self._reclaimed(direction, i))
            )
            if not ok_deep:
                return blk("deep_pb_unconfirmed")
            pb_type = PullbackType.DEEP_PULLBACK.value

        # ── trigger (tier 1 fast / tier 2 confirmed) ─────────────────────────
        trig = self._trigger(direction, i, candle_ctx, liq, zone_score, s15)
        if trig is None:
            return blk(f"waiting_trigger@{pb_type}")
        trigger_name, trigger_tier, trig_pts = trig

        # extension guard: don't buy a pullback that already rebounded too far
        ext = i.long_extension_atr if long else i.short_extension_atr
        if ext > c.pullback_max_extension_atr:
            return blk(f"extended_{ext:.1f}ATR")

        # ── stops & structure room (hard gates) ──────────────────────────────
        stop_cands = self._stop_candidates(direction, i, state, zone, liq, s15)
        invalidation = self._invalidation(direction, i, state, zone, s15)
        opp = nearest_opposing_zone(zones, direction, price, min_score=c.zone_min_score)
        entry_ref = price
        prelim_stop = stop_cands[0][1] if stop_cands else (
            price - a if long else price + a)
        risk_dist = abs(entry_ref - prelim_stop)
        if risk_dist <= 0:
            return blk("no_stop_structure")
        if opp is not None:
            room = (opp.lower_price - entry_ref) if long else (entry_ref - opp.upper_price)
            room_r = room / risk_dist
        else:
            room_r = 99.0
        if room_r < c.pullback_min_structure_room_r:
            return blk(f"room_{room_r:.1f}R")

        # ── score (spec 18.10) ───────────────────────────────────────────────
        score = 0.0
        score += self._bias_pts(bias, direction)                        # 0-15
        score += self._macro_pts(macro, direction)                      # 0-10
        score += self._structure_1h_pts(s1h, direction)                 # 0-15
        score += min(20.0, zone_score / 5.0 + loc_score)                # HTF location 0-20
        score += prior_detail                                           # 0-10
        score += self._pullback_quality_pts(direction, i, pb_type)      # 0-10
        score += trig_pts                                               # 0-10
        score += (candle_ctx.bull_quality if long else candle_ctx.bear_quality) * 5.0  # 0-5
        score += self._momentum_pts(direction, i)                       # 0-5
        score = min(score, 100.0)

        # ── dynamic threshold ────────────────────────────────────────────────
        thr = c.pullback_threshold
        if pb_type == PullbackType.DEEP_PULLBACK.value:
            thr = max(thr, c.deep_pullback_threshold) + 0.0
        if trigger_tier == 1 and zone_score < c.early_trigger_zone_score:
            thr = max(thr, c.early_pullback_threshold)
        if conflict == ConflictLevel.MILD:
            thr += c.mod_mild_conflict
        elif macro.direction == direction:
            thr += c.mod_alignment
        if zone_score >= c.zone_hq_score:
            thr += c.mod_hq_zone
        if room_r < c.pullback_hq_room_r:
            thr += c.mod_opposing_near
        if room_r >= c.pullback_bonus_room_r:
            score = min(100.0, score + 3.0)
        thr = float(np.clip(thr, c.pullback_threshold_min, c.pullback_threshold_max))
        if score < thr:
            return blk(f"score_{score:.0f}<{thr:.0f}")
        # 1.00-1.10R room passes only for high-quality setups
        if room_r < c.pullback_hq_room_r and score < thr + 4:
            return blk(f"room_{room_r:.1f}R_needs_hq")

        self.last_block.setdefault(symbol, {})[direction] = f"READY score={score:.0f}"
        risk_mod = bias_risk_mod
        if pb_type == PullbackType.DEEP_PULLBACK.value:
            risk_mod *= 0.8
        if conflict == ConflictLevel.MILD:
            risk_mod *= 0.85

        target_ref = (opp.lower_price - a * c.target_buffer_atr) if (long and opp is not None) else \
                     (opp.upper_price + a * c.target_buffer_atr) if (not long and opp is not None) else \
                     (entry_ref + risk_dist * c.risk_reward if long
                      else entry_ref - risk_dist * c.risk_reward)

        return SignalCandidate(
            symbol=symbol, direction=direction, setup_type=SetupType.FAST_PULLBACK.value,
            score=score, threshold=thr, edge_score=score - thr,
            entry_reference=entry_ref, structure_stop=prelim_stop, target_reference=target_ref,
            breakout_level=None, retest_level=None, invalidation_level=invalidation,
            signal_timestamp=now_ts,
            signal_expiry=now_ts + bar_ms * (c.signal_expiry_bars + 1),
            htf_structure=macro.classification, bias=bias.bias, regime=regime.confirmed_regime,
            nearest_support=None if long else (opp.upper_price if opp else None),
            nearest_resistance=(opp.lower_price if opp else None) if long else None,
            structure_room_r=float(room_r),
            active_zone=zone, zone_score=zone_score,
            pattern_type=None, pattern_status=None,
            candle_pattern=trigger_name,
            candle_quality=(candle_ctx.bull_quality if long else candle_ctx.bear_quality),
            candle_location_score=candle_ctx.location_score,
            risk_modifier=risk_mod,
            reason_codes=[f"PB_TYPE:{pb_type}", f"TRIGGER_TIER:{trigger_tier}", f"BIAS:{bias_why}"],
            stop_candidates=stop_cands,
        )

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _macro_view(macro, s1h) -> StructureView:
        v = StructureView(timeframe="4h", state=macro.structure_state)
        return v

    def _prior_context(self, direction, i: EntryIndicators, s15: StructureView):
        """Set A (structure) / B (displacement) / C (momentum); structure must
        not contradict. Returns (ok, points 0-10)."""
        long = direction == "LONG"
        contra = (StructureState.STRONG_BEAR.value,) if long else (StructureState.STRONG_BULL.value,)
        if s15.state in contra:
            return False, 0.0
        pts = 0.0
        # A: structure impulse
        a_ok = s15.last_bos is not None and s15.last_bos.direction == direction
        if a_ok:
            pts = 10.0
        # B: displacement within last 3 bars
        b_ok = False
        if len(i.closes) >= 4:
            for k in range(1, 4):
                move = (i.closes[-k] - i.opens[-min(k + 2, len(i.opens))]) / max(i.last_atr, EPS)
                if long and move >= self.cfg.prior_context_displacement_atr:
                    b_ok = True
                if not long and -move >= self.cfg.prior_context_displacement_atr:
                    b_ok = True
        if b_ok:
            pts = max(pts, 7.0)
        # C: momentum
        r = i.val(i.roc)
        di_ok = (i.di_spread_long > 0) if long else (i.di_spread_short > 0)
        c_ok = ((r > 0) if long else (r < 0)) and di_ok and i.val(i.adx) >= self.cfg.min_adx * 0.7
        if c_ok:
            pts = max(pts, 5.0)
        return (a_ok or b_ok or c_ok), pts

    def _location(self, direction, i, zones, sd_zones, liq, s15, patterns):
        """Valid pullback location (spec 18.3). Returns (pb_type, zone|None,
        zone_score 0-100, extra_loc_pts 0-4) or None."""
        c = self.cfg
        long = direction == "LONG"
        price = i.val(i.closes)
        low = float(i.lows[-1])
        high = float(i.highs[-1])
        a = i.last_atr or EPS
        hf, hs = i.val(i.hma_fast), i.val(i.hma_slow)

        hz_upper = max(hf, hs) + a * c.hma_pullback_zone_upper_atr
        hz_lower = min(hf, hs) - a * c.hma_pullback_zone_lower_atr
        touched_hma = (low <= hz_upper and high >= hz_lower) if long else \
                      (high >= hz_lower and low <= hz_upper)

        support_side = long
        zs = zones_at_price(zones, price, support_side)
        # probe zone touch by the bar's extreme too
        probe = low if long else high
        zs += [z for z in zones if not z.broken and z.contains(probe)
               and (z.is_support_like if long else z.is_resistance_like) and z not in zs]
        zone = zs[0] if zs else None
        zone_score = zone.strength if zone else 0.0

        # supply/demand
        sd_hit = None
        for z in sd_zones:
            lo, hi = min(z.proximal_line, z.distal_line), max(z.proximal_line, z.distal_line)
            if z.direction == direction and lo - a * 0.2 <= probe <= hi + a * 0.2:
                sd_hit = z
                break

        sweep_ok = (liq.bullish_sweep if long else liq.bearish_sweep) and liq.sweep_location_ok

        # breakout-retest
        br_zone = next((z for z in zones if z.zone_type == "BREAKOUT_RETEST"
                        and not z.broken and z.contains(probe)), None)

        if sweep_ok:
            return ("LIQUIDITY_SWEEP_PULLBACK", zone or br_zone,
                    max(zone_score, 65.0), 4.0)
        if br_zone is not None:
            return ("BREAKOUT_RETEST", br_zone, max(br_zone.strength, 60.0), 3.0)
        if zone is not None and zone.timeframe in ("1h", "4h"):
            return ("HTF_ZONE_PULLBACK", zone, zone_score, 3.0)
        if sd_hit is not None:
            return ("HTF_ZONE_PULLBACK", zone, max(zone_score, 62.0), 3.0)
        if touched_hma:
            return ("SHALLOW_PULLBACK", zone, max(zone_score, 45.0), 1.5)
        if zone is not None:      # 15m zone
            return ("SHALLOW_PULLBACK", zone, zone_score, 1.0)
        return None

    def _reclaimed(self, direction, i: EntryIndicators) -> bool:
        long = direction == "LONG"
        hf = i.val(i.hma_fast)
        last_close = i.val(i.closes)
        prev_close = i.val(i.closes, -2)
        if long:
            return prev_close <= hf < last_close or last_close > float(i.highs[-2])
        return prev_close >= hf > last_close or last_close < float(i.lows[-2])

    def _trigger(self, direction, i, candle_ctx, liq, zone_score, s15):
        """Tier1 fast trigger at strong location, tier2 confirmed elsewhere.
        Returns (name, tier, points 0-10) or None."""
        c = self.cfg
        long = direction == "LONG"
        last_close = i.val(i.closes)
        prev_high, prev_low = float(i.highs[-2]), float(i.lows[-2])
        hf = i.val(i.hma_fast)
        reclaim_hma = (i.val(i.closes, -2) <= i.val(i.hma_fast, -2) and last_close > hf) if long \
            else (i.val(i.closes, -2) >= i.val(i.hma_fast, -2) and last_close < hf)
        close_beyond_prev = last_close > prev_high if long else last_close < prev_low
        sweep = (liq.bullish_sweep if long else liq.bearish_sweep)
        micro_bos = (s15.last_bos is not None and s15.last_bos.direction == direction
                     and s15.last_bos.confirmed_at == int(i.timestamps[-1]))
        rejection = ("SUPPORT_REJECTION" in candle_ctx.bull_triggers
                     or "BULLISH_PIN_BAR" in candle_ctx.bull_triggers) if long else \
                    ("RESISTANCE_REJECTION" in candle_ctx.bear_triggers
                     or "BEARISH_PIN_BAR" in candle_ctx.bear_triggers)
        engulf = ("BULLISH_ENGULFING" in candle_ctx.bull_triggers) if long \
            else ("BEARISH_ENGULFING" in candle_ctx.bear_triggers)
        close_q = i.bull_close_quality if long else i.bear_close_quality
        same_bar_ok = (zone_score >= c.same_bar_zone_score and i.body_atr >= c.same_bar_body_atr
                       and close_q >= c.same_bar_close_quality
                       and (i.volume_ratio >= c.pullback_volume_ratio or i.body_atr >= 0.25))

        # Tier 1 — strong location, one clean signal is enough
        if zone_score >= c.early_trigger_zone_score or (sweep and liq.sweep_location_ok):
            if reclaim_hma or sweep or same_bar_ok or close_beyond_prev:
                name = ("SWEEP_RECLAIM" if sweep else
                        "HMA10_RECLAIM" if reclaim_hma else
                        "SAME_BAR_RECLAIM" if same_bar_ok else "CLOSE_BEYOND_PREV")
                return name, 1, 9.0
        # Tier 2 — confirmed trigger
        if close_beyond_prev or micro_bos or reclaim_hma or engulf or rejection:
            name = ("MICRO_BOS" if micro_bos else
                    "CLOSE_BEYOND_PREV" if close_beyond_prev else
                    "HMA10_RECLAIM" if reclaim_hma else
                    "ENGULFING" if engulf else "REJECTION")
            return name, 2, 7.0
        return None

    def _stop_candidates(self, direction, i, state, zone, liq, s15) -> list:
        """Setup-relevant stop candidates (spec 23.1 list, pre-buffer)."""
        long = direction == "LONG"
        out = []
        if state.setup_low is not None and long:
            out.append(("SETUP_LOW", state.setup_low))
        if state.setup_high is not None and not long:
            out.append(("SETUP_HIGH", state.setup_high))
        if liq.sweep_level is not None:
            out.append(("SWEEP_LEVEL", liq.sweep_level))
        if zone is not None:
            out.append(("ZONE_BOUNDARY", zone.lower_price if long else zone.upper_price))
        swing = [s for s in s15.swings if s.swing_type == ("low" if long else "high")
                 and s.confirmed_at <= int(i.timestamps[-1])]
        if swing:
            out.append(("LOCAL_SWING", swing[-1].price))
        # local bar extreme fallback
        out.append(("BAR_EXTREME", float(np.min(i.lows[-4:])) if long
                    else float(np.max(i.highs[-4:]))))
        price = i.val(i.closes)
        valid = [(n, p) for n, p in out if (p < price if long else p > price)]
        # nearest first (deterministic tie-break by name)
        valid.sort(key=lambda t: (abs(price - t[1]), t[0]))
        return valid

    def _invalidation(self, direction, i, state, zone, s15) -> float:
        long = direction == "LONG"
        cands = []
        if state.invalidation_level is not None:
            cands.append(state.invalidation_level)
        if zone is not None:
            cands.append(zone.lower_price if long else zone.upper_price)
        swing = [s for s in s15.swings if s.swing_type == ("low" if long else "high")]
        if swing:
            cands.append(swing[-1].price)
        if not cands:
            a = i.last_atr or EPS
            return i.val(i.closes) - a * 1.2 if long else i.val(i.closes) + a * 1.2
        return min(cands) if long else max(cands)

    # score helpers -----------------------------------------------------------
    @staticmethod
    def _bias_pts(bias, direction) -> float:
        s = bias.score_long if direction == "LONG" else bias.score_short
        return min(15.0, s * 0.15)

    @staticmethod
    def _macro_pts(macro, direction) -> float:
        if macro.direction == direction:
            return min(10.0, macro.score * 0.10)
        if macro.classification in ("TRANSITION", "RANGE", "NEUTRAL"):
            return 4.0
        return 0.0

    @staticmethod
    def _structure_1h_pts(s1h: StructureView, direction) -> float:
        aligned = {"LONG": (StructureState.BULL.value, StructureState.STRONG_BULL.value),
                   "SHORT": (StructureState.BEAR.value, StructureState.STRONG_BEAR.value)}[direction]
        if s1h.state in aligned:
            return 12.0 + min(3.0, s1h.quality / 33.3)
        if s1h.state == StructureState.TRANSITION.value:
            return 6.0
        return 2.0

    def _pullback_quality_pts(self, direction, i, pb_type) -> float:
        # shallow orderly pullbacks near HMA score best; overextended rebounds score less
        ext = i.long_extension_atr if direction == "LONG" else i.short_extension_atr
        pts = 6.0
        if pb_type in ("HTF_ZONE_PULLBACK", "LIQUIDITY_SWEEP_PULLBACK", "BREAKOUT_RETEST"):
            pts += 2.0
        if 0 <= ext <= 0.45:
            pts += 2.0
        return min(pts, 10.0)

    def _momentum_pts(self, direction, i) -> float:
        long = direction == "LONG"
        pts = 0.0
        if (i.val(i.roc) > 0) == long:
            pts += 2.0
        if (i.di_spread_long > 0) == long:
            pts += 1.5
        if i.val(i.adx) >= self.cfg.min_adx:
            pts += 1.5
        return min(pts, 5.0)
