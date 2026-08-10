"""Sentinel log-clarity overlay.

This module changes display text/metadata only. Trading gates, entries, exits,
SL/TP, MCDX thresholds, proximity and forecast behaviour are untouched.
"""
from __future__ import annotations


def install_sentinel_log_clarity(strategy_cls) -> None:
    """Make Railway HOLD traces describe map readiness and Fib blockers clearly."""
    if getattr(strategy_cls, "_sentinel_log_clarity_installed", False):
        return

    original_analyze = strategy_cls.analyze
    strategy_cls._sentinel_log_clarity_installed = True

    async def analyze(self, candles: list, current_price: float, mtf_candles: dict = None):
        signal = await original_analyze(self, candles, current_price, mtf_candles)

        reason = str(getattr(signal, "reason", "") or "")
        # S1/R1-only means the risk map exists; it does NOT mean all entry gates pass.
        reason = reason.replace(
            "LONG S1-only ready;",
            "LONG MAP READY: S1-only (entry gates still required);",
        )
        reason = reason.replace(
            "SHORT R1-only ready;",
            "SHORT MAP READY: R1-only (entry gates still required);",
        )

        md = dict(getattr(signal, "metadata", {}) or {})
        fib = dict(md.get("fib") or {})
        fib_reason = str(fib.get("reason", "") or "")

        # V2 used to print only `Fib leg inactive (x.xxATR)`, which could make a
        # large leg look as though size were the blocker. Split SIZE from
        # direction/structure/BOS alignment so the real bottleneck is visible.
        if fib_reason.startswith("Fib leg inactive"):
            leg_atr = float(fib.get("leg_atr", 0.0) or 0.0)
            min_leg = float(getattr(self, "fib_min_leg_atr", 2.0) or 2.0)
            sxv = dict(md.get("probabilistic_target") or {})
            structure = str(sxv.get("structure", "MIXED") or "MIXED")
            trend_dir = float(sxv.get("trend_dir", 0.0) or 0.0)
            trend_txt = "BULL" if trend_dir > 0 else "BEAR" if trend_dir < 0 else "NEUTRAL"

            bos_up = False
            bos_dn = False
            try:
                pivot_fn = getattr(self, "_sentinel_v2_pivot_points", None)
                if callable(pivot_fn) and candles:
                    highs, lows = pivot_fn(candles[-160:], 4)
                    close = float(candles[-1].close)
                    if highs:
                        last_ph = float(highs[-1][1])
                        prev_high = float(highs[-2][1]) if len(highs) >= 2 else last_ph
                        bos_up = close > prev_high
                    if lows:
                        last_pl = float(lows[-1][1])
                        prev_low = float(lows[-2][1]) if len(lows) >= 2 else last_pl
                        bos_dn = close < prev_low
            except Exception:
                # Logging must never break trading.
                bos_up = bos_dn = False

            size_pass = leg_atr >= min_leg
            valid_bull = (structure == "BULL" or bos_up) and trend_dir >= 0
            valid_bear = (structure == "BEAR" or bos_dn) and trend_dir < 0
            alignment_pass = valid_bull or valid_bear

            if not size_pass:
                clearer = (
                    f"FIB2 INACTIVE leg={leg_atr:.2f}ATR [SIZE BLOCK min={min_leg:.2f}] | "
                    f"structure={structure} trend={trend_txt}({trend_dir:.1f}) "
                    f"BOS_UP={'YES' if bos_up else 'NO'} BOS_DN={'YES' if bos_dn else 'NO'}"
                )
            else:
                clearer = (
                    f"FIB2 INACTIVE leg={leg_atr:.2f}ATR [SIZE PASS min={min_leg:.2f}] | "
                    f"structure={structure} trend={trend_txt}({trend_dir:.1f}) "
                    f"BOS_UP={'YES' if bos_up else 'NO'} BOS_DN={'YES' if bos_dn else 'NO'} | "
                    f"ALIGNMENT {'PASS' if alignment_pass else 'BLOCK'}"
                )

            old_fragment = f"FIB2: {fib_reason}"
            if old_fragment in reason:
                reason = reason.replace(old_fragment, clearer)
            elif fib_reason in reason:
                reason = reason.replace(fib_reason, clearer)
            else:
                reason = f"{reason} | {clearer}" if reason else clearer

            fib["size_pass"] = size_pass
            fib["min_leg_atr"] = min_leg
            fib["structure"] = structure
            fib["trend_dir"] = round(trend_dir, 1)
            fib["bos_up"] = bos_up
            fib["bos_dn"] = bos_dn
            fib["alignment_pass"] = alignment_pass
            fib["reason_detail"] = clearer
            md["fib"] = fib

        signal.reason = reason
        signal.metadata = md
        return signal

    strategy_cls.analyze = analyze
