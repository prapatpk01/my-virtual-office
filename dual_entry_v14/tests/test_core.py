"""Unit tests for DUAL ENTRY PRECISION V1.4 core invariants."""
from __future__ import annotations

import asyncio
import math
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from dual_entry_v14.config import Config, TF_MS
from dual_entry_v14.models import Candle, SignalCandidate, SymbolState
from dual_entry_v14.enums import SymbolStatus
from dual_entry_v14.swing_engine import SwingEngine
from dual_entry_v14.structure_engine import StructureEngine
from dual_entry_v14.indicator_engine import IndicatorEngine
from dual_entry_v14.risk_manager import RiskManager
from dual_entry_v14.regime_engine import RegimeEngine
from dual_entry_v14.data_quality_gate import DataQualityGate
from dual_entry_v14.state_store import StateStore
from dual_entry_v14.candidate_selector import CandidateSelector
from dual_entry_v14.monte_carlo import monte_carlo


def cfg_for_test() -> Config:
    c = Config(symbols=["BTC/USDT:USDT"])
    c.validate()
    return c


def mk_candles(closes, tf="15m", start=1_700_000_000_000, wig=0.4, vol=1000.0):
    step = TF_MS[tf]
    out = []
    prev = closes[0]
    for k, cl in enumerate(closes):
        o = prev
        hi, lo = max(o, cl) + wig, min(o, cl) - wig
        out.append(Candle(start + k * step, o, hi, lo, cl, vol))
        prev = cl
    return out


class TestSwings(unittest.TestCase):
    def test_no_future_pivot(self):
        """A pivot is only confirmed after right-side bars close."""
        c = cfg_for_test()
        eng = SwingEngine(c)
        closes = list(np.linspace(100, 110, 20)) + [112, 109, 108, 107, 106, 105, 104]
        candles = mk_candles(closes)
        swings = eng.calculate(candles, "15m")
        for s in swings:
            self.assertGreaterEqual(s.confirmed_at, s.timestamp + c.swing_right_bars * TF_MS["15m"] - 1,
                                    "swing confirmed before right bars closed")

    def test_swing_detection(self):
        c = cfg_for_test()
        eng = SwingEngine(c)
        closes = [100, 101, 102, 105, 102, 101, 100, 99, 98, 101, 104, 106, 103, 101, 100]
        swings = eng.calculate(mk_candles(closes), "15m")
        kinds = {s.swing_type for s in swings}
        self.assertIn("high", kinds)
        self.assertIn("low", kinds)


class TestRiskManager(unittest.TestCase):
    def _cand(self, entry=100.0, stops=None, direction="LONG"):
        return SignalCandidate(
            symbol="BTC/USDT:USDT", direction=direction, setup_type="FAST_PULLBACK",
            score=70, threshold=64, edge_score=6, entry_reference=entry,
            structure_stop=stops[0][1] if stops else entry - 1,
            target_reference=entry + 2, breakout_level=None, retest_level=None,
            invalidation_level=entry - 2, signal_timestamp=0, signal_expiry=10**15,
            htf_structure="BULL", bias="BULL", regime="BULL_TREND",
            nearest_support=None, nearest_resistance=None, structure_room_r=2.0,
            active_zone=None, zone_score=70.0, pattern_type=None, pattern_status=None,
            candle_pattern=None, candle_quality=0.7, candle_location_score=6.0,
            risk_modifier=1.0, stop_candidates=stops or [])

    def test_deterministic_stop_nearest_valid(self):
        """Chooses the NEAREST structural stop in the ATR band, not the deepest."""
        c = cfg_for_test()
        rm = RiskManager(c)
        atr = 1.0
        stops = [("SETUP_LOW", 99.3), ("ZONE_BOUNDARY", 98.5), ("LOCAL_SWING", 97.0)]
        stop, codes = rm.select_stop(self._cand(100.0, stops), atr)
        # 99.3 - buffer(0.08) = 99.22 -> dist 0.78 ATR in [0.45, 1.60] band
        self.assertAlmostEqual(stop, 99.3 - atr * c.stop_buffer_atr, places=6)
        self.assertIn("STOP:SETUP_LOW", codes)

    def test_stop_too_tight_widens(self):
        c = cfg_for_test()
        rm = RiskManager(c)
        atr = 1.0
        stops = [("SETUP_LOW", 99.9)]     # 0.18 ATR — below min 0.45
        stop, codes = rm.select_stop(self._cand(100.0, stops), atr)
        self.assertAlmostEqual(stop, 100.0 - atr * c.min_stop_atr, places=6)
        self.assertIn("STOP:WIDENED_TO_MIN", codes)

    def test_sizing_includes_fees_and_slippage(self):
        c = cfg_for_test()
        rm = RiskManager(c)
        cand = self._cand(100.0, [("SETUP_LOW", 99.0)])
        plan = rm.build_trade_plan(cand, {"equity": 10_000, "free_margin": 10_000,
                                          "leverage": 10},
                                   {"atr": 1.0, "contract_size": 0.001, "lot_step": 1.0,
                                    "min_qty": 1.0, "tick_size": 0.0, "min_notional": 0.0})
        self.assertTrue(plan.is_valid, plan.reason_codes)
        eff = plan.effective_risk_distance
        raw = plan.risk_distance
        self.assertGreater(eff, raw, "effective risk must include slippage + fees")
        # quantity * effective distance ≈ risk cash (within lot rounding)
        self.assertLessEqual(plan.quantity * eff, plan.risk_cash * 1.001)

    def test_low_rr_rejected(self):
        c = cfg_for_test()
        rm = RiskManager(c)
        cand = self._cand(100.0, [("SETUP_LOW", 99.0)])
        cand.nearest_resistance = 100.5      # structure target too close -> RR < min
        plan = rm.build_trade_plan(cand, {"equity": 10_000, "free_margin": 10_000,
                                          "leverage": 10},
                                   {"atr": 1.0, "contract_size": 0.001, "lot_step": 1.0,
                                    "min_qty": 1.0, "tick_size": 0.0, "min_notional": 0.0})
        self.assertFalse(plan.is_valid)
        self.assertTrue(any("REJECT_LOW_RR" in r for r in plan.reason_codes))


class TestStateStore(unittest.TestCase):
    def test_atomic_roundtrip_and_version(self):
        with tempfile.TemporaryDirectory() as d:
            ss = StateStore(d)
            st = ss.get("BTC/USDT:USDT")
            st.status = SymbolStatus.LONG_OPEN.value
            st.actual_entry = 123.456
            v0 = st.state_version
            ss.save_atomic("BTC/USDT:USDT", st)
            ss._cache.clear()
            st2 = ss.get("BTC/USDT:USDT")
            self.assertEqual(st2.status, SymbolStatus.LONG_OPEN.value)
            self.assertEqual(st2.actual_entry, 123.456)
            self.assertEqual(st2.state_version, v0 + 1)

    def test_journal_intent(self):
        with tempfile.TemporaryDirectory() as d:
            ss = StateStore(d)
            ss.journal("X", "ORDER_INTENT", {"client_order_id": "abc123"})
            self.assertTrue(ss.journal_has_intent("abc123"))
            self.assertFalse(ss.journal_has_intent("zzz"))


class TestSignalKey(unittest.TestCase):
    def test_deterministic_client_order_id(self):
        a = TestRiskManager()._cand(100.0, [("S", 99.0)])
        b = TestRiskManager()._cand(100.0, [("S", 99.0)])
        self.assertEqual(a.client_order_id, b.client_order_id)
        self.assertEqual(len(a.client_order_id), 24)
        b2 = TestRiskManager()._cand(100.0, [("S", 99.0)], direction="SHORT")
        self.assertNotEqual(a.client_order_id, b2.client_order_id)


class TestSelector(unittest.TestCase):
    def test_directional_ambiguity_rejects_both(self):
        sel = CandidateSelector()
        st = SymbolState(symbol="X")
        pb = TestRiskManager()._cand(100.0, [("S", 99.0)], "LONG")
        mo = TestRiskManager()._cand(100.0, [("S", 101.0)], "SHORT")
        mo.setup_type = "MOMENTUM"
        self.assertIsNone(sel.select(pb, mo, st))

    def test_pullback_preferred_when_close(self):
        sel = CandidateSelector()
        st = SymbolState(symbol="X")
        pb = TestRiskManager()._cand(100.0, [("S", 99.0)], "LONG")
        mo = TestRiskManager()._cand(100.0, [("S", 99.0)], "LONG")
        mo.setup_type = "MOMENTUM"
        pb.edge_score, mo.edge_score = 5.0, 6.5     # within 2 -> pullback
        self.assertEqual(sel.select(pb, mo, st).setup_type, "FAST_PULLBACK")

    def test_blocked_when_position_open(self):
        sel = CandidateSelector()
        st = SymbolState(symbol="X", status=SymbolStatus.LONG_OPEN.value)
        pb = TestRiskManager()._cand(100.0, [("S", 99.0)], "LONG")
        self.assertIsNone(sel.select(pb, None, st))


class TestRegime(unittest.TestCase):
    def test_chop_detected(self):
        c = cfg_for_test()
        ind = IndicatorEngine(c)
        reg = RegimeEngine(c)
        rng = np.random.default_rng(7)
        closes = 100 + np.cumsum(rng.normal(0, 0.03, 400)) * 0     # dead-flat noise
        closes = 100 + rng.normal(0, 0.05, 400)
        candles = mk_candles(list(closes))
        i = ind.calculate_entry(candles)
        from dual_entry_v14.models import StructureView
        sv = StructureView(timeframe="15m", state="RANGE")
        res = reg.classify(i, sv, [], None, None, 0)
        self.assertTrue(res.is_chop, f"flat noise must classify CHOP, got {res.raw_regime}")


class TestDataQuality(unittest.TestCase):
    def test_rejects_short_series(self):
        c = cfg_for_test()
        g = DataQualityGate(c)
        candles = mk_candles(list(np.linspace(100, 101, 50)))
        res = g.evaluate("X", candles, candles, candles,
                         now_ms=candles[-1].timestamp + TF_MS["15m"])
        self.assertFalse(res.valid)

    def test_shock_flagged(self):
        c = cfg_for_test()
        g = DataQualityGate(c)
        closes = list(np.linspace(100, 101, 320))
        candles = mk_candles(closes, wig=0.1)
        big = candles[-1]
        candles[-1] = Candle(big.timestamp, big.open, big.open + 5, big.open - 5,
                             big.open + 4.5, 5000)
        res = g.evaluate("X", candles,
                         mk_candles(list(np.linspace(100, 101, 220)), "1h"),
                         mk_candles(list(np.linspace(100, 101, 220)), "4h"),
                         now_ms=candles[-1].timestamp + TF_MS["15m"])
        self.assertTrue(res.valid)
        self.assertTrue(res.shock)


class TestMonteCarlo(unittest.TestCase):
    def test_bootstrap_shapes(self):
        rs = [1.2, -1.0, 0.8, -1.0, 1.5, -1.0, 0.3]
        out = monte_carlo(rs, 0.01, n_paths=200, seed=1)
        self.assertEqual(out["paths"], 200)
        self.assertGreater(out["median_final"], 0)
        self.assertLessEqual(out["prob_ruin_pct"], 100.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
