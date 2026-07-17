"""Momentum Breakout Engine (spec §19).

Structure-backed breakout/continuation only — entry on the breakout candle
or the very next bar; never chases beyond that.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .config import Config, TF_MS
from .enums import ConflictLevel, ReasonCode, SetupType, StructureState
from .indicator_engine import EPS, EntryIndicators
from .models import SignalCandidate, StructureView, SymbolState
from .pattern_engine import best_pattern_for
from .support_resistance_engine import nearest_opposing_zone
from .swing_engine import swings_of


class MomentumEngine:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.last_block: dict = {}   # symbol -> {direction: why} — view-log diagnostics

    def evaluate(self, symbol: str, state: SymbolState,
                 indicators_15m: Optional[EntryIndicators], bias, macro_context,
                 regime, structure_15m: StructureView, structure_1h: StructureView,
                 zones: list, patterns: list, candle_context,
                 shock_lockout: bool = False) -> Optional[SignalCandidate]:
        if indicators_15m is None:
            return None
        for direction in ("LONG", "SHORT"):
            cand = self._side(direction, symbol, state, indicators_15m, bias, macro_context,
                              regime, structure_15m, structure_1h, zones, patterns,
                              candle_context, shock_lockout)
            if cand is not None:
                return cand
        return None

    def _side(self, direction, symbol, state, i, bias, macro, regime, s15, s1h,
              zones, patterns, candle_ctx, shock_lockout):
        c = self.cfg

        def blk(why: str):
            self.last_block.setdefault(symbol, {})[direction] = why
            return None
        long = direction == "LONG"
        price = i.val(i.closes)
        a = i.last_atr or EPS
        now_ts = int(i.timestamps[-1])
        bar_ms = TF_MS[c.entry_timeframe]
        last_close = price
        last_open = i.val(i.opens)
        body_atr = i.body_atr
        close_q = i.bull_close_quality if long else i.bear_close_quality

        # ── breakout event detection (any one) ───────────────────────────────
        event, level, major = self._breakout_event(direction, i, s15, s1h, zones, patterns)
        if event is None:
            return blk("no_breakout_event")

        # entry timing: breakout candle or next bar only
        bars_since_break = self._bars_since_break(direction, i, level)
        if bars_since_break is None or bars_since_break > c.momentum_expiry_bars:
            return blk(f"break_expired({event})")
        retest_mode = bars_since_break == 1
        if retest_mode:
            # next bar must not close back inside
            if (last_close <= level) if long else (last_close >= level):
                return blk("retest_closed_back_inside")

        # ── volatility shock lockout (unless high-quality breakout allowed) ──
        if shock_lockout:
            hq = (body_atr >= c.strong_breakout_body_atr and close_q >= c.strong_breakout_close_quality
                  and major and c.allow_high_quality_shock_breakout)
            if not hq:
                return blk("shock_lockout")

        # ── hard core gates ──────────────────────────────────────────────────
        if regime.is_chop:
            return blk("CHOP")
        wick_only = (last_close <= level) if long else (last_close >= level)
        if wick_only:
            return blk("wick_only_break")
        if s1h.recent_choch_against(direction, TF_MS[c.bias_timeframe] * 8, now_ts):
            return blk("1H_OPP_CHOCH")
        conflict = macro.conflict_for(direction, price, zones, StructureView("4h", macro.structure_state))
        if conflict == ConflictLevel.STRONG:
            return blk("4H_STRONG_CONFLICT")
        allowed, bias_risk_mod, bias_why = bias.allows(
            direction, s15, zones, price, s1h, now_ts, TF_MS[c.bias_timeframe])
        if not allowed:
            return blk(f"bias:{bias_why}")

        # extension (measured vs breakout level when near, else HMA16)
        dist_level = abs(last_close - level) / a
        dist_hma = i.long_extension_atr if long else i.short_extension_atr
        ext = dist_level if dist_level <= dist_hma else min(dist_level, dist_hma)
        max_ext = c.strong_trend_extension_atr if "STRONG" in regime.confirmed_regime \
            else c.momentum_max_extension_atr
        if ext > max_ext:
            return blk(f"extended_{ext:.1f}ATR")

        # indicator support — flexible 2 of 4 (1 of 4 for major+strong displacement)
        hf, hs = i.val(i.hma_fast), i.val(i.hma_slow)
        sup = 0
        sup += 1 if ((hf > hs) if long else (hf < hs)) else 0
        sup += 1 if ((i.val(i.roc) > 0) if long else (i.val(i.roc) < 0)) else 0
        sup += 1 if ((i.di_spread_long > 0) if long else (i.di_spread_short > 0)) else 0
        adx_now, adx_prev = i.val(i.adx), i.val(i.adx, -2)
        sup += 1 if (adx_now >= c.momentum_min_adx or adx_now > adx_prev) else 0
        strong_disp = body_atr >= c.strong_breakout_body_atr and close_q >= c.strong_breakout_close_quality
        need = 1 if (major and strong_disp) else 2
        if sup < need:
            return blk(f"indicator_support_{sup}/{need}")

        # candle quality sets (standard / strong / retest)
        std_ok = body_atr >= c.std_breakout_body_atr and close_q >= c.std_breakout_close_quality
        strong_ok = strong_disp and (i.volume_ratio >= c.momentum_volume_ratio or body_atr >= 0.35)
        retest_ok = retest_mode and ((last_close > level) if long else (last_close < level))
        if not (std_ok or strong_ok or retest_ok):
            return blk("weak_breakout_candle")

        # false breakout check on the prior bar
        if s15.last_false_bos is not None and s15.last_false_bos.direction != direction \
                and now_ts - s15.last_false_bos.confirmed_at <= bar_ms * 3:
            return blk("recent_false_bos")

        # ── stop candidates & structure room ─────────────────────────────────
        stop_cands = self._stop_candidates(direction, i, level, s15, patterns)
        prelim_stop = stop_cands[0][1] if stop_cands else (price - a if long else price + a)
        risk_dist = abs(price - prelim_stop)
        if risk_dist <= 0:
            return blk("no_stop_structure")
        opp = nearest_opposing_zone(zones, direction, price, min_score=c.zone_min_score)
        if opp is not None:
            room = (opp.lower_price - price) if long else (price - opp.upper_price)
            room_r = room / risk_dist
        else:
            room_r = 99.0
        if room_r < c.momentum_min_structure_room_r:
            return blk(f"room_{room_r:.1f}R")

        # ── breakout quality score (0-30) ────────────────────────────────────
        bq = 0.0
        bq += 8.0 if major else 5.0                                       # structure break
        bq += 5.0 * min(1.0, (abs(last_close - level) / max(a * 0.1, EPS)) / 3.0 + 0.4)  # real close beyond
        bq += 5.0 * min(1.0, body_atr / max(c.strong_breakout_body_atr, EPS)) * close_q  # candle quality
        bq += 4.0 if self._compressed(i) else 0.0
        bq += 4.0 * min(1.0, i.volume_ratio / max(c.momentum_volume_ratio, EPS)) \
            if i.volume_ratio > 0 else 2.0
        bq += 4.0 if retest_ok or self._held_level(direction, i, level) else 2.0
        bq = min(bq, 30.0)
        breakout_quality_high = bq >= 20.0

        if not regime.allows_momentum(direction, breakout_quality_high, confirmed_breakout=True):
            return blk(f"regime={regime.confirmed_regime}_bq{bq:.0f}")

        # ── momentum score (0-100, spec 19.7) ────────────────────────────────
        score = 0.0
        score += min(15.0, (bias.score_long if long else bias.score_short) * 0.15)
        score += (min(10.0, macro.score * 0.10) if macro.direction == direction
                  else 4.0 if macro.classification in ("TRANSITION", "RANGE", "NEUTRAL") else 0.0)
        p = best_pattern_for(patterns, direction)
        s1h_pts = 10.0 if s1h.state in (
            (StructureState.BULL.value, StructureState.STRONG_BULL.value) if long
            else (StructureState.BEAR.value, StructureState.STRONG_BEAR.value)) else 4.0
        score += min(15.0, s1h_pts + (5.0 if p is not None else 0.0))
        score += bq
        r_now = i.val(i.roc)
        score += min(8.0, abs(r_now) * 2.0 if ((r_now > 0) == long) else 0.0)
        di = i.di_spread_long if long else i.di_spread_short
        score += min(8.0, max(0.0, di) * 0.5 + (4.0 if adx_now >= c.momentum_min_adx else 0.0))
        score += (candle_ctx.bull_quality if long else candle_ctx.bear_quality) * 8.0
        score += min(6.0, (room_r - c.momentum_min_structure_room_r) * 4.0 + 2.0)
        score = min(score, 100.0)

        thr = c.strong_breakout_threshold if (major and strong_disp) else c.momentum_threshold
        if conflict == ConflictLevel.MILD:
            thr += c.mod_mild_conflict
        elif macro.direction == direction:
            thr += c.mod_alignment
        if major:
            thr += c.mod_major_breakout
        if room_r < c.momentum_hq_room_r:
            thr += c.mod_opposing_near
        thr = float(np.clip(thr, c.momentum_threshold_min, c.momentum_threshold_max))
        if score < thr:
            return blk(f"score_{score:.0f}<{thr:.0f}")
        if room_r < c.momentum_hq_room_r and not breakout_quality_high:
            return blk(f"room_{room_r:.1f}R_needs_bq")

        self.last_block.setdefault(symbol, {})[direction] = f"READY score={score:.0f}"
        risk_mod = bias_risk_mod * (0.85 if conflict == ConflictLevel.MILD else 1.0)
        target_ref = (opp.lower_price - a * c.target_buffer_atr) if (long and opp is not None) else \
                     (opp.upper_price + a * c.target_buffer_atr) if (not long and opp is not None) else \
                     (price + risk_dist * c.risk_reward if long else price - risk_dist * c.risk_reward)

        return SignalCandidate(
            symbol=symbol, direction=direction, setup_type=SetupType.MOMENTUM.value,
            score=score, threshold=thr, edge_score=score - thr,
            entry_reference=price, structure_stop=prelim_stop, target_reference=target_ref,
            breakout_level=level, retest_level=level,
            invalidation_level=level - a * 0.3 if long else level + a * 0.3,
            signal_timestamp=now_ts, signal_expiry=now_ts + bar_ms * (c.momentum_expiry_bars + 1),
            htf_structure=macro.classification, bias=bias.bias, regime=regime.confirmed_regime,
            nearest_support=(opp.upper_price if opp else None) if not long else None,
            nearest_resistance=(opp.lower_price if opp else None) if long else None,
            structure_room_r=float(room_r), active_zone=opp, zone_score=(opp.strength if opp else 0.0),
            pattern_type=(p.pattern_type if p else None), pattern_status=(p.status if p else None),
            candle_pattern=(candle_ctx.best_bull() if long else candle_ctx.best_bear()),
            candle_quality=(candle_ctx.bull_quality if long else candle_ctx.bear_quality),
            candle_location_score=candle_ctx.location_score,
            risk_modifier=risk_mod,
            reason_codes=[f"EVENT:{event}", f"BQ:{bq:.0f}", f"MAJOR:{major}",
                          f"RETEST:{retest_mode}", f"BIAS:{bias_why}"],
            stop_candidates=stop_cands,
        )

    # ── event/levels ─────────────────────────────────────────────────────────

    def _breakout_event(self, direction, i: EntryIndicators, s15, s1h, zones, patterns):
        """Return (event_name, level, is_major) — first satisfied source wins.
        The level is what the breakout must CLOSE beyond."""
        c = self.cfg
        long = direction == "LONG"
        closes = i.closes
        last_close = float(closes[-1])
        lb = c.breakout_lookback

        # 1) highest-high(4) / lowest-low(4)
        if len(i.highs) > lb + 1:
            hh = float(np.max(i.highs[-lb - 1:-1]))
            ll = float(np.min(i.lows[-lb - 1:-1]))
            if long and last_close > hh:
                pass_level, name, major = hh, "HH4_BREAK", False
            elif not long and last_close < ll:
                pass_level, name, major = ll, "LL4_BREAK", False
            else:
                pass_level = None
        else:
            pass_level = None

        # 2) 15M confirmed swing break (major)
        swing = swings_of(s15.swings, "high" if long else "low")
        swing = [s for s in swing if s.confirmed_at < int(i.timestamps[-1])]
        if swing:
            lvl = swing[-1].price
            if (long and last_close > lvl) or (not long and last_close < lvl):
                return "15M_SWING_BREAK", lvl, True

        # 3) 1H zone break (major)
        for z in zones:
            if z.timeframe != "1h" or z.broken:
                continue
            if long and z.is_resistance_like and last_close > z.upper_price:
                return "1H_RESISTANCE_BREAK", z.upper_price, True
            if not long and z.is_support_like and last_close < z.lower_price:
                return "1H_SUPPORT_BREAK", z.lower_price, True

        # 4) range high/low break
        for z in zones:
            if z.zone_type not in ("RANGE_HIGH", "RANGE_LOW") or z.broken:
                continue
            if long and z.zone_type == "RANGE_HIGH" and last_close > z.upper_price:
                return "RANGE_BREAK", z.upper_price, True
            if not long and z.zone_type == "RANGE_LOW" and last_close < z.lower_price:
                return "RANGE_BREAK", z.lower_price, True

        # 5) pattern neckline
        p = best_pattern_for(patterns, direction)
        if p is not None and p.breakout_level is not None:
            if (long and last_close > p.breakout_level) or \
               (not long and last_close < p.breakout_level):
                return f"PATTERN_{p.pattern_type}", p.breakout_level, True

        # 6) fresh HMA cross w/ expansion
        hf_arr, hs_arr = i.hma_fast, i.hma_slow
        if len(hf_arr) > 3:
            crossed = False
            for k in (1, 2):
                a1, b1 = hf_arr[-k] - hs_arr[-k], hf_arr[-k - 1] - hs_arr[-k - 1]
                if np.isfinite(a1) and np.isfinite(b1) and (a1 > 0 >= b1 if long else a1 < 0 <= b1):
                    crossed = True
                    break
            if crossed and i.body_atr >= 0.25:
                lvl = float(i.highs[-3]) if long else float(i.lows[-3])
                if (long and last_close > lvl) or (not long and last_close < lvl):
                    return "FRESH_HMA_CROSS", lvl, False

        # 7) compression breakout
        if self._compressed(i) and pass_level is not None:
            return "COMPRESSION_BREAK", pass_level, False

        if pass_level is not None:
            return name, pass_level, major
        return None, None, False

    def _bars_since_break(self, direction, i, level) -> Optional[int]:
        long = direction == "LONG"
        closes = i.closes
        for k in (1, 2):
            if len(closes) <= k:
                return None
            beyond = closes[-k] > level if long else closes[-k] < level
            before = closes[-k - 1] <= level if long else closes[-k - 1] >= level
            if beyond and before:
                return k - 1
        return None

    def _held_level(self, direction, i, level) -> bool:
        long = direction == "LONG"
        return (float(i.lows[-1]) > level) if long else (float(i.highs[-1]) < level)

    def _compressed(self, i: EntryIndicators) -> bool:
        c = self.cfg
        rng = i.highs - i.lows
        if len(rng) < 22:
            return False
        recent = float(np.mean(rng[-4:-1]))
        normal = float(np.mean(rng[-21:-1]))
        if normal > 0 and recent <= normal * c.compression_recent_mult:
            return True
        tr3 = float(np.mean(rng[-3:]))
        atr14 = i.last_atr
        return atr14 > 0 and tr3 / atr14 <= c.compression_atr_ratio

    def _stop_candidates(self, direction, i, level, s15, patterns) -> list:
        long = direction == "LONG"
        price = i.val(i.closes)
        out = []
        out.append(("RETEST_LOW" if long else "RETEST_HIGH",
                    float(i.lows[-1]) if long else float(i.highs[-1])))
        out.append(("BREAKOUT_LEVEL", level))
        base = float(np.min(i.lows[-6:])) if long else float(np.max(i.highs[-6:]))
        out.append(("BREAKOUT_BASE", base))
        p = best_pattern_for(patterns, direction)
        if p is not None and p.invalidation_level is not None:
            out.append(("PATTERN_INVALIDATION", p.invalidation_level))
        swing = swings_of(s15.swings, "low" if long else "high")
        if swing:
            out.append(("LOCAL_SWING", swing[-1].price))
        valid = [(n, p_) for n, p_ in out if (p_ < price if long else p_ > price)]
        valid.sort(key=lambda t: (abs(price - t[1]), t[0]))
        return valid
