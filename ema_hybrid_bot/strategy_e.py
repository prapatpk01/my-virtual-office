"""EMA Hybrid Quality V2.5 — Setup E + A/B RSI/SMA confirmation.

Extends V2.4 A/B/C/D with:
- Setup E: MACD + Volume + Price Action confirmation, including trend-aligned
  MACD cross and confirmed MACD divergence variants.
- Setup A/B confirmation: 5M price SMA14 + RSI14 + SMA14(RSI) momentum layer.

Priority: A > E > C > D > B on the same closed 5M candle.
All engines remain aligned with the existing 15M directional bias.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
_BASE_PATH = os.path.join(HERE, "strategy_d.py")
_SPEC = importlib.util.spec_from_file_location("ema_hybrid_quality_v24_abcd", _BASE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Cannot load EMA Hybrid A/B/C/D strategy: {_BASE_PATH}")
_BASE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _BASE
_SPEC.loader.exec_module(_BASE)

Side = _BASE.Side
TriggerView = _BASE.TriggerView


class EMAHybridProStrategy(_BASE.EMAHybridProStrategy):
    """Quality V2.5: A/B RSI-SMA confirm + C + D + E MACD/Volume."""

    # A/B 5M confirmation ---------------------------------------------------
    AB_SMA_LEN = max(5, int(os.getenv("EMA_5M_AB_SMA_LEN", "14")))
    AB_RSI_LEN = max(5, int(os.getenv("EMA_5M_AB_RSI_LEN", "14")))
    AB_RSI_SMA_LEN = max(3, int(os.getenv("EMA_5M_AB_RSI_SMA_LEN", "14")))

    A_RSI_LONG_MIN = float(os.getenv("EMA_5M_A_RSI_LONG_MIN", "52"))
    A_RSI_SHORT_MAX = float(os.getenv("EMA_5M_A_RSI_SHORT_MAX", "48"))
    B_RSI_LONG_MIN = float(os.getenv("EMA_5M_B_RSI_LONG_MIN", "50"))
    B_RSI_SHORT_MAX = float(os.getenv("EMA_5M_B_RSI_SHORT_MAX", "50"))

    # Setup E — MACD + Volume ---------------------------------------------
    E_ENABLED = os.getenv("EMA_5M_SETUP_E_ENABLED", "true").strip().lower() in {
        "1", "true", "yes", "on",
    }
    E_VOLUME_LEN = max(5, int(os.getenv("EMA_5M_E_VOLUME_LEN", "20")))
    E_VOLUME_RATIO_MIN = float(os.getenv("EMA_5M_E_VOLUME_RATIO_MIN", "1.10"))
    E_ADX_MIN = float(os.getenv("EMA_5M_E_ADX_MIN", "15"))
    E_CHOP_MAX = float(os.getenv("EMA_5M_E_CHOP_MAX", "58"))
    E_DIV_LOOKBACK = max(12, int(os.getenv("EMA_5M_E_DIV_LOOKBACK", "30")))
    E_DIV_PIVOT_SPAN = max(1, int(os.getenv("EMA_5M_E_DIV_PIVOT_SPAN", "2")))
    E_DIV_MIN_PRICE_ATR = float(os.getenv("EMA_5M_E_DIV_MIN_PRICE_ATR", "0.05"))

    def _prep5(self, frame: pd.DataFrame) -> pd.DataFrame:
        d = super()._prep5(frame)
        c = d["close"].astype(float)
        d["ab_sma14"] = c.rolling(self.AB_SMA_LEN, min_periods=self.AB_SMA_LEN).mean()
        d["ab_rsi14"] = self._rsi(c, self.AB_RSI_LEN)
        d["ab_rsi_sma14"] = d["ab_rsi14"].rolling(
            self.AB_RSI_SMA_LEN, min_periods=self.AB_RSI_SMA_LEN
        ).mean()
        # Previous-bar average prevents the current volume spike from inflating its own baseline.
        d["e_vol_avg"] = d["volume"].astype(float).rolling(
            self.E_VOLUME_LEN, min_periods=self.E_VOLUME_LEN
        ).mean().shift(1)
        return d

    def _filter_ab_confirmation(self, df5: pd.DataFrame, view: TriggerView) -> TriggerView:
        """Add 5M price-SMA14 and RSI/SMA(RSI) confirmation to A and B only."""
        if not view.ready or view.setup not in {"A_EMA_CROSS", "B_PULLBACK_RECLAIM"}:
            return view

        d = self._prep5(df5)
        if len(d) < max(self.AB_SMA_LEN, self.AB_RSI_LEN + self.AB_RSI_SMA_LEN) + 2:
            return TriggerView(view.side, False, view.adx, view.chop, "NONE", "NONE", "5M A/B FILTERED: RSI/SMA warmup")

        r, p = d.iloc[-1], d.iloc[-2]
        vals = (r.ab_sma14, p.ab_sma14, r.ab_rsi14, p.ab_rsi14, r.ab_rsi_sma14)
        if any(pd.isna(x) for x in vals):
            return TriggerView(view.side, False, view.adx, view.chop, "NONE", "NONE", "5M A/B FILTERED: RSI/SMA warmup")

        close_now = float(r.close)
        sma_now = float(r.ab_sma14)
        sma_prev = float(p.ab_sma14)
        rsi_now = float(r.ab_rsi14)
        rsi_prev = float(p.ab_rsi14)
        rsi_sma = float(r.ab_rsi_sma14)

        if view.setup == "A_EMA_CROSS":
            long_min = self.A_RSI_LONG_MIN
            short_max = self.A_RSI_SHORT_MAX
            name = "A"
        else:
            long_min = self.B_RSI_LONG_MIN
            short_max = self.B_RSI_SHORT_MAX
            name = "B"

        if view.side == Side.LONG:
            price_ok = close_now > sma_now and sma_now > sma_prev
            rsi_ok = rsi_now >= long_min and rsi_now > rsi_sma and rsi_now >= rsi_prev
        else:
            price_ok = close_now < sma_now and sma_now < sma_prev
            rsi_ok = rsi_now <= short_max and rsi_now < rsi_sma and rsi_now <= rsi_prev

        if price_ok and rsi_ok:
            return TriggerView(
                view.side, True, view.adx, view.chop, view.setup, view.trigger,
                f"{view.reason} | {name} RSI/SMA CONFIRM: Close/SMA{self.AB_SMA_LEN}=OK "
                f"RSI={rsi_now:.1f} RSI-SMA={rsi_sma:.1f} momentum=OK",
            )

        return TriggerView(
            view.side, False, view.adx, view.chop, "NONE", "NONE",
            f"5M {name}_RSI_SMA_FILTERED {view.side.value.upper()} | Close={close_now:.6g} "
            f"SMA{self.AB_SMA_LEN}={sma_now:.6g} slope={sma_now-sma_prev:.6g} "
            f"RSI={rsi_now:.1f} RSI-SMA={rsi_sma:.1f}",
        )

    def _macd_divergence(self, d: pd.DataFrame) -> tuple[bool, bool]:
        """Confirmed two-pivot MACD divergence inside a bounded recent window."""
        span = self.E_DIV_PIVOT_SPAN
        start = max(span, len(d) - self.E_DIV_LOOKBACK)
        end = len(d) - span
        lows: list[int] = []
        highs: list[int] = []

        for i in range(start, end):
            w = d.iloc[i - span:i + span + 1]
            if float(d.low.iloc[i]) <= float(w.low.min()):
                lows.append(i)
            if float(d.high.iloc[i]) >= float(w.high.max()):
                highs.append(i)

        atr = max(float(d.atr.iloc[-1]), 1e-12)
        min_move = self.E_DIV_MIN_PRICE_ATR * atr
        bull = False
        bear = False

        if len(lows) >= 2:
            a, b = lows[-2], lows[-1]
            bull = (
                float(d.low.iloc[b]) <= float(d.low.iloc[a]) - min_move
                and float(d.macd.iloc[b]) > float(d.macd.iloc[a])
            )
        if len(highs) >= 2:
            a, b = highs[-2], highs[-1]
            bear = (
                float(d.high.iloc[b]) >= float(d.high.iloc[a]) + min_move
                and float(d.macd.iloc[b]) < float(d.macd.iloc[a])
            )
        return bull, bear

    def _trigger_e(self, df5: pd.DataFrame, bias_side):
        """MACD cross/divergence confirmed by volume, price action and regime quality."""
        if not self.E_ENABLED or bias_side is None or len(df5) < 80:
            return None

        d = self._prep5(df5)
        r, p = d.iloc[-1], d.iloc[-2]
        needed = (
            r.macd, r.macd_signal, r.macd_hist, p.macd, p.macd_signal, p.macd_hist,
            r.e_vol_avg, r.ab_sma14,
        )
        if any(pd.isna(x) for x in needed):
            return None

        macd_now = float(r.macd)
        sig_now = float(r.macd_signal)
        hist_now = float(r.macd_hist)
        macd_prev = float(p.macd)
        sig_prev = float(p.macd_signal)
        hist_prev = float(p.macd_hist)

        golden = macd_prev <= sig_prev and macd_now > sig_now
        death = macd_prev >= sig_prev and macd_now < sig_now
        bull_div, bear_div = self._macd_divergence(d)

        vol_avg = max(float(r.e_vol_avg), 1e-12)
        vol_ratio = float(r.volume) / vol_avg
        volume_ok = vol_ratio >= self.E_VOLUME_RATIO_MIN

        adx, chop = self._quality_values_5m(d)
        quality_ok = adx >= self.E_ADX_MIN and chop <= self.E_CHOP_MAX

        close_now = float(r.close)
        close_prev = float(p.close)
        sma14 = float(r.ab_sma14)
        bull_price = close_now > float(r.open) and close_now > close_prev and close_now > sma14
        bear_price = close_now < float(r.open) and close_now < close_prev and close_now < sma14
        macd_improving = hist_now > hist_prev and macd_now > macd_prev
        macd_weakening = hist_now < hist_prev and macd_now < macd_prev

        if bias_side == Side.LONG:
            signal = golden or bull_div
            if signal and macd_improving and bull_price:
                source = "GOLDEN_CROSS" if golden else "BULL_DIV"
                if volume_ok and quality_ok:
                    zone = "ABOVE_ZERO" if macd_now >= 0 else "BELOW_ZERO"
                    return TriggerView(
                        Side.LONG, True, adx, chop,
                        "E_MACD_VOLUME", f"MACD_VOLUME_{source}_LONG",
                        f"5M READY LONG: E {source} + Volume {vol_ratio:.2f}x + price confirm "
                        f"| MACD={macd_now:.6g} Hist={hist_now:.6g} {zone} ADX={adx:.1f} CHOP={chop:.1f}",
                    )
                return TriggerView(
                    Side.LONG, False, adx, chop, "NONE", "NONE",
                    f"5M E_FILTERED LONG {source} | Vol={vol_ratio:.2f}x ADX={adx:.1f} CHOP={chop:.1f}",
                )
            return None

        signal = death or bear_div
        if signal and macd_weakening and bear_price:
            source = "DEATH_CROSS" if death else "BEAR_DIV"
            if volume_ok and quality_ok:
                zone = "BELOW_ZERO" if macd_now <= 0 else "ABOVE_ZERO"
                return TriggerView(
                    Side.SHORT, True, adx, chop,
                    "E_MACD_VOLUME", f"MACD_VOLUME_{source}_SHORT",
                    f"5M READY SHORT: E {source} + Volume {vol_ratio:.2f}x + price confirm "
                    f"| MACD={macd_now:.6g} Hist={hist_now:.6g} {zone} ADX={adx:.1f} CHOP={chop:.1f}",
                )
            return TriggerView(
                Side.SHORT, False, adx, chop, "NONE", "NONE",
                f"5M E_FILTERED SHORT {source} | Vol={vol_ratio:.2f}x ADX={adx:.1f} CHOP={chop:.1f}",
            )
        return None

    def _trigger5(self, df5: pd.DataFrame, bias_side):
        # Existing D resolves A > C > D > B. A keeps first priority when confirmed.
        base_view = super()._trigger5(df5, bias_side)

        if base_view.ready and base_view.setup == "A_EMA_CROSS":
            a_view = self._filter_ab_confirmation(df5, base_view)
            if a_view.ready:
                return a_view
            # A raw signal failed precision confirmation; allow a fully confirmed E alternative.
            e_view = self._trigger_e(df5, bias_side)
            if e_view is not None and e_view.ready:
                return e_view
            return a_view

        # Setup E has stronger cross/divergence + volume confirmation than C/D/B.
        e_view = self._trigger_e(df5, bias_side)
        if e_view is not None and e_view.ready:
            return e_view

        if base_view.ready and base_view.setup == "B_PULLBACK_RECLAIM":
            return self._filter_ab_confirmation(df5, base_view)

        if base_view.ready:
            return base_view
        if e_view is not None:
            return e_view
        return base_view

    def entry_status(self, df4h, df1h, df15, df5):
        base_text = super().entry_status(df4h, df1h, df15, df5)
        return (
            base_text
            + f" | A/B Confirm=Price-SMA{self.AB_SMA_LEN}+RSI{self.AB_RSI_LEN}/RSI-SMA{self.AB_RSI_SMA_LEN}"
            + f" | E=MACD+Volume>={self.E_VOLUME_RATIO_MIN:.2f}x+PriceAction"
            + f" ADX>={self.E_ADX_MIN:.0f} CHOP<={self.E_CHOP_MAX:.0f} DivLB={self.E_DIV_LOOKBACK}"
        )
