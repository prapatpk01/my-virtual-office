"""Confluence Engine — HYBRID entry mode (regime trigger -> Dual evaluation).

Per user's design: the regime bot's 3-TF cross confluence is the FIRST filter
(the trigger); the Dual pipeline then evaluates the trade lightly (soft bias
gate, structure stop, room) and, if it passes, trades immediately. The strict
pullback/momentum scoring engines are bypassed in this mode — the cross IS
the signal. Portfolio gate, risk manager (structure stop/target, min RR,
effective-risk sizing) and execution quality still apply downstream.

Layers (same as signal_regime_bot):
    L3a — HMA(10)/HMA(16) cross on 30M
    L3b — EMA(5)/EMA(9)   cross on 15M
    L3c — EMA(10)/EMA(20) cross on 5M

Any layer cross ARMS a setup; >= conf_min_layers (2) layers crossing the SAME
direction within conf_window_min (15) minutes fires a candidate. One entry per
setup: re-firing requires a cross NEWER than the newest one already consumed.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from .config import Config, TF_MS
from .enums import SetupType
from .indicator_engine import ema, hma
from .models import SignalCandidate
from .support_resistance_engine import nearest_opposing_zone

logger = logging.getLogger("dual_entry.confluence")

EPS = 1e-12


class ConfluenceEngine:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        # symbol -> {layer: (cross_close_ms, direction)}
        self._crosses: dict = {}
        # symbol -> newest cross_close_ms consumed by the last fired entry
        self._consumed: dict = {}
        # symbol -> {direction: reason} for the view log (same shape as pb/mo)
        self.last_block: dict = {}

    # ── cross detection ──────────────────────────────────────────────────────

    @staticmethod
    def _last_cross(candles: list, fast_arr: np.ndarray, slow_arr: np.ndarray,
                    tf: str, lookback: int) -> Optional[tuple]:
        """Most recent fast/slow cross within `lookback` closed bars.
        Returns (cross_close_ms, direction) or None. Scanning a window (not
        just the last bar) keeps live and backtest identical even though the
        pipeline only runs once per closed 15M bar."""
        n = len(candles)
        if n < 3:
            return None
        d = fast_arr - slow_arr
        start = max(1, n - lookback)
        best = None
        for i in range(start, n):
            a, b = d[i - 1], d[i]
            if not (np.isfinite(a) and np.isfinite(b)):
                continue
            if a <= 0 < b:
                best = (int(candles[i].timestamp) + TF_MS[tf], "LONG")
            elif a >= 0 > b:
                best = (int(candles[i].timestamp) + TF_MS[tf], "SHORT")
        return best

    def observe(self, symbol: str, c30: list, c15: list, c5: list) -> None:
        """Record the freshest cross per layer. Called once per closed 15M bar,
        so each layer scans enough bars to cover the gap between calls."""
        c = self.cfg
        store = self._crosses.setdefault(symbol, {})
        layers = (
            ("L3a", c30, "30m", lambda x: (hma(x, c.conf_hma_fast), hma(x, c.conf_hma_slow)), 2),
            ("L3b", c15, "15m", lambda x: (ema(x, c.conf_ema_fast), ema(x, c.conf_ema_slow)), 2),
            ("L3c", c5, "5m", lambda x: (ema(x, c.conf_l3c_fast), ema(x, c.conf_l3c_slow)), 4),
        )
        for name, candles, tf, mk, lb in layers:
            if not candles or len(candles) < c.conf_min_layer_candles:
                continue
            closes = np.array([x.close for x in candles], dtype=float)
            fast_arr, slow_arr = mk(closes)
            hit = self._last_cross(candles, fast_arr, slow_arr, tf, lb)
            if hit is not None:
                prev = store.get(name)
                if prev is None or hit[0] > prev[0]:
                    store[name] = hit

    def on_position_closed(self, symbol: str) -> None:
        """Consume every armed cross so re-entry needs a genuinely new one."""
        store = self._crosses.get(symbol, {})
        if store:
            self._consumed[symbol] = max(ts for ts, _ in store.values())

    # ── evaluation ───────────────────────────────────────────────────────────

    def evaluate(self, symbol: str, state, ind_15m, bias, macro_ctx, regime,
                 s15, s1h, zones, c30: list, c15: list, c5: list,
                 now_ms: int) -> Optional[SignalCandidate]:
        c = self.cfg
        self.observe(symbol, c30, c15, c5)

        def blk(direction: str, why: str) -> None:
            self.last_block.setdefault(symbol, {})[direction] = why

        store = self._crosses.get(symbol, {})
        if not store:
            blk("LONG", "no_cross_yet"); blk("SHORT", "no_cross_yet")
            return None

        window_ms = c.conf_window_min * 60_000
        active: dict = {"LONG": [], "SHORT": []}
        for name, (ts, direction) in store.items():
            if now_ms - ts <= window_ms:
                active[direction].append((name, ts))

        direction = None
        for side in ("LONG", "SHORT"):
            n = len(active[side])
            if n >= c.conf_min_layers:
                direction = side
            else:
                armed = ",".join(sorted(nm for nm, _ in active[side])) or "-"
                blk(side, f"conf {n}/{c.conf_min_layers} ({armed})")
        if direction is None:
            return None
        if len(active["LONG"]) >= c.conf_min_layers and \
                len(active["SHORT"]) >= c.conf_min_layers:
            blk("LONG", "ambiguous_both_sides"); blk("SHORT", "ambiguous_both_sides")
            return None

        newest = max(ts for _, ts in active[direction])
        if self._consumed.get(symbol) is not None and newest <= self._consumed[symbol]:
            blk(direction, "no_new_cross_since_last_entry")
            return None

        # ── light Dual evaluation (soft bias gate — the first filter) ───────
        allowed, bias_risk_mod, bias_why = bias.allows(
            direction, s15, s1h, TF_MS["1h"], now_ms)
        if not allowed:
            blk(direction, f"bias:{bias_why}")
            return None

        long = direction == "LONG"
        i = ind_15m
        price = float(i.closes[-1])
        a = float(i.atr[-1]) if np.isfinite(i.atr[-1]) else 0.0
        if a <= 0 or price <= 0:
            blk(direction, "no_atr")
            return None

        # structure stop candidates: recent 15M swing + ATR fallback
        base = float(np.min(i.lows[-6:])) if long else float(np.max(i.highs[-6:]))
        stop_cands = [("SWING6", base),
                      ("ATR", price - a if long else price + a)]
        prelim_stop = stop_cands[0][1]
        risk_dist = abs(price - prelim_stop)
        if risk_dist <= 0:
            prelim_stop = stop_cands[1][1]
            risk_dist = abs(price - prelim_stop)

        opp = nearest_opposing_zone(zones, direction, price, min_score=c.zone_min_score)
        if opp is not None:
            room = (opp.lower_price - price) if long else (price - opp.upper_price)
            room_r = room / max(risk_dist, EPS)
        else:
            room_r = 99.0
        if room_r < c.conf_min_room_r:
            blk(direction, f"room_{room_r:.1f}R")
            return None

        target_ref = (opp.lower_price - a * c.target_buffer_atr) if (long and opp is not None) else \
                     (opp.upper_price + a * c.target_buffer_atr) if (not long and opp is not None) else \
                     (price + risk_dist * c.risk_reward if long else price - risk_dist * c.risk_reward)

        n_layers = len(active[direction])
        score = 55.0 + 15.0 * n_layers          # 2 layers -> 85, 3 -> 100
        now_ts = int(c15[-1].timestamp) if c15 else int(now_ms - TF_MS["15m"])
        bar_ms = TF_MS["15m"]

        self.last_block.setdefault(symbol, {})[direction] = \
            f"READY conf {n_layers}/3 ({','.join(sorted(nm for nm, _ in active[direction]))})"
        self._consumed[symbol] = newest

        return SignalCandidate(
            symbol=symbol, direction=direction, setup_type=SetupType.CONFLUENCE.value,
            score=score, threshold=50.0, edge_score=score - 50.0,
            entry_reference=price, structure_stop=prelim_stop, target_reference=target_ref,
            breakout_level=None, retest_level=None,
            invalidation_level=prelim_stop,
            signal_timestamp=now_ts, signal_expiry=now_ts + bar_ms * 2,
            htf_structure=macro_ctx.classification, bias=bias.bias,
            regime=regime.confirmed_regime,
            nearest_support=(opp.upper_price if opp else None) if not long else None,
            nearest_resistance=(opp.lower_price if opp else None) if long else None,
            structure_room_r=float(room_r), active_zone=opp,
            zone_score=(opp.strength if opp else 0.0),
            pattern_type=None, pattern_status=None,
            candle_pattern=None, candle_quality=0.5, candle_location_score=0.5,
            risk_modifier=bias_risk_mod,
            reason_codes=[f"CONF:{n_layers}/3",
                          f"LAYERS:{','.join(sorted(nm for nm, _ in active[direction]))}",
                          f"BIAS:{bias_why}"],
            stop_candidates=stop_cands,
        )
