"""
SJ WaveTrend + UT Bot Dual-Confirm Strategy.

Signal requires BOTH components to fire within 3 bars:

  WaveTrend side:
    wtBuy  = wt1 crossover wt2  AND  wt1 < -35
    wtSell = wt1 crossunder wt2 AND  wt1 > +45
    Bar-2 candle confirm (use_candle=True by default):
      BUY:  wtBuy[1] AND prev-bar is green AND open > open[1]
      SELL: wtSell[1] AND prev-bar is red  AND open < close[1]

  UT Bot side (mult=1.0, atr_period=10):
    Trailing stop line trails price by mult×ATR, flipping on crossover
    utBuy  = close crosses above trailing stop
    utSell = close crosses below trailing stop

  Final: (wtFinal AND utRecent) OR (utSignal AND wtRecent)
    where Recent = fired within last 3 bars

SL = 1.5×ATR(14), R:R = 1:1.5
"""
import numpy as np
from .base import BaseStrategy, Signal, SignalType

_ATR_PERIOD  = 14
_SL_MULTS    = [1.0, 1.5, 2.0, 2.5]
_LOOKFORWARD = 60


class WTADXStrategy(BaseStrategy):

    def __init__(self, symbol: str, params: dict = None):
        super().__init__(symbol, params)
        self.n1          = self.params.get("wt_channel_len", 10)
        self.n2          = self.params.get("wt_avg_len",     21)
        self.ob_level    = self.params.get("ob_level",       45.0)
        self.os_level    = self.params.get("os_level",      -35.0)
        self.use_candle  = self.params.get("use_candle",     True)
        self.ut_mult     = self.params.get("ut_mult",         1.0)
        self.ut_atr_len  = self.params.get("ut_atr_len",      10)
        self.sl_atr_mult = self.params.get("sl_atr_mult",     1.5)
        self.rr_ratio    = self.params.get("rr_ratio",        1.5)
        self._last_signal = 0

    # ── WaveTrend ──────────────────────────────────────────────────

    def _wavetrend(self, highs, lows, closes):
        h = np.array(highs); l = np.array(lows); c = np.array(closes)
        ap  = (h + l + c) / 3.0
        esa = np.array(self.ema(ap.tolist(), self.n1))
        diff_abs = np.abs(ap - esa)
        first_ok = np.where(~np.isnan(diff_abs))[0]
        if len(first_ok):
            diff_filled = diff_abs.copy()
            diff_filled[:first_ok[0]] = diff_abs[first_ok[0]]
        else:
            diff_filled = diff_abs
        d   = np.array(self.ema(diff_filled.tolist(), self.n1))
        ci  = np.where(d > 1e-10, (ap - esa) / (0.015 * d), 0.0)
        wt1 = np.array(self.ema(ci.tolist(), self.n2))
        wt2 = np.array(self.sma(wt1.tolist(), 4))
        return wt1, wt2

    # ── UT Bot trailing stop ───────────────────────────────────────

    def _ut_trailing_stop(self, closes: np.ndarray, atr_a: np.ndarray) -> np.ndarray:
        """
        Trails price by mult×ATR, flipping direction on price crossover.
        Matches Pine Script UT Bot logic exactly.
        """
        n   = len(closes)
        tsl = np.full(n, np.nan)
        for i in range(n):
            slval = float(atr_a[i]) * self.ut_mult if not np.isnan(atr_a[i]) else 0.0
            src   = float(closes[i])
            if i == 0 or np.isnan(tsl[i - 1]):
                tsl[i] = src + slval
                continue
            tsl_p = float(tsl[i - 1])
            src_p = float(closes[i - 1])
            if src > tsl_p and src_p > tsl_p:
                tsl[i] = max(tsl_p, src - slval)
            elif src < tsl_p and src_p < tsl_p:
                tsl[i] = min(tsl_p, src + slval)
            elif src > tsl_p:
                tsl[i] = src - slval
            else:
                tsl[i] = src + slval
        return tsl

    # ── Pre-compute all signal arrays ─────────────────────────────

    def _build_signals(self, candles: list):
        """
        Pre-compute all signal arrays for the full candle list.

        Returns tuple:
          (wt1, wt2, tsl,
           wt_buy_final, wt_sell_final,
           ut_buy, ut_sell,
           buy_sig, sell_sig,
           atr14, ut_atr)
        """
        n   = len(candles)
        cls = np.array([c.close  for c in candles], dtype=float)
        ops = np.array([c.open   for c in candles], dtype=float)
        hig = np.array([c.high   for c in candles], dtype=float)
        low = np.array([c.low    for c in candles], dtype=float)

        wt1, wt2 = self._wavetrend(hig.tolist(), low.tolist(), cls.tolist())
        atr14  = np.array(self.atr(candles, _ATR_PERIOD),       dtype=float)
        ut_atr = np.array(self.atr(candles, self.ut_atr_len),   dtype=float)
        tsl    = self._ut_trailing_stop(cls, ut_atr)

        # ─ WT base cross signals ─
        wt_buy_base  = np.zeros(n, dtype=bool)
        wt_sell_base = np.zeros(n, dtype=bool)
        for i in range(1, n):
            if np.isnan(wt1[i]) or np.isnan(wt2[i]):
                continue
            cu = float(wt1[i-1]) <= float(wt2[i-1]) and float(wt1[i]) > float(wt2[i])
            cd = float(wt1[i-1]) >= float(wt2[i-1]) and float(wt1[i]) < float(wt2[i])
            wt_buy_base[i]  = cu and float(wt1[i]) < self.os_level
            wt_sell_base[i] = cd and float(wt1[i]) > self.ob_level

        # ─ Bar-2 candle confirm ─
        # BUY:  wtBuy[1] AND prev-bar green AND cur_open > prev_open
        # SELL: wtSell[1] AND prev-bar red  AND cur_open < prev_close
        wt_buy_f  = np.zeros(n, dtype=bool)
        wt_sell_f = np.zeros(n, dtype=bool)
        if self.use_candle:
            for i in range(2, n):
                wt_buy_f[i]  = (wt_buy_base[i-1]
                                and cls[i-1] > ops[i-1]    # prev green
                                and ops[i]   > ops[i-1])   # cur open > prev open
                wt_sell_f[i] = (wt_sell_base[i-1]
                                and cls[i-1] < ops[i-1]    # prev red
                                and ops[i]   < cls[i-1])   # cur open < prev close
        else:
            wt_buy_f[:]  = wt_buy_base
            wt_sell_f[:] = wt_sell_base

        # ─ UT Bot crossover signals ─
        ut_buy  = np.zeros(n, dtype=bool)
        ut_sell = np.zeros(n, dtype=bool)
        for i in range(1, n):
            if np.isnan(tsl[i]) or np.isnan(tsl[i-1]):
                continue
            ut_buy[i]  = cls[i-1] < tsl[i-1] and cls[i] > tsl[i]
            ut_sell[i] = cls[i-1] > tsl[i-1] and cls[i] < tsl[i]

        # ─ Dual confirm — WT + UT must both fire within 3 bars ─
        buy_sig  = np.zeros(n, dtype=bool)
        sell_sig = np.zeros(n, dtype=bool)
        for i in range(2, n):
            wt_b_rec = bool(wt_buy_f[i]  or wt_buy_f[i-1]  or wt_buy_f[i-2])
            wt_s_rec = bool(wt_sell_f[i] or wt_sell_f[i-1] or wt_sell_f[i-2])
            ut_b_rec = bool(ut_buy[i]    or ut_buy[i-1]    or ut_buy[i-2])
            ut_s_rec = bool(ut_sell[i]   or ut_sell[i-1]   or ut_sell[i-2])
            buy_sig[i]  = (wt_buy_f[i]  and ut_b_rec) or (ut_buy[i]  and wt_b_rec)
            sell_sig[i] = (wt_sell_f[i] and ut_s_rec) or (ut_sell[i] and wt_s_rec)

        return (wt1, wt2, tsl,
                wt_buy_f, wt_sell_f,
                ut_buy, ut_sell,
                buy_sig, sell_sig,
                atr14, ut_atr)

    # ── Kept for external gate usage ──────────────────────────────

    @staticmethod
    def compute_wt1(candles: list, n1: int = 10, n2: int = 21) -> float:
        """Return current WT1 value — used as gate by other strategies."""
        if len(candles) < n1 + n2 + 5:
            return float("nan")
        h = np.array([c.high  for c in candles])
        l = np.array([c.low   for c in candles])
        c = np.array([c.close for c in candles])
        ap  = (h + l + c) / 3.0
        esa = np.array(BaseStrategy.ema(ap.tolist(), n1))
        diff_abs = np.abs(ap - esa)
        first_ok = np.where(~np.isnan(diff_abs))[0]
        if len(first_ok):
            diff_filled = diff_abs.copy()
            diff_filled[:first_ok[0]] = diff_abs[first_ok[0]]
        else:
            diff_filled = diff_abs
        d  = np.array(BaseStrategy.ema(diff_filled.tolist(), n1))
        ci = np.where(d > 1e-10, (ap - esa) / (0.015 * d), 0.0)
        return float(BaseStrategy.ema(ci.tolist(), n2)[-1])

    # ── Live analysis ──────────────────────────────────────────────

    async def analyze(self, candles: list, current_price: float,
                      mtf_candles: dict = None) -> Signal:
        min_len = self.n1 + self.n2 + max(_ATR_PERIOD, self.ut_atr_len) + 10
        if len(candles) < min_len:
            return Signal(SignalType.HOLD, self.symbol, current_price, 0, "Not enough data")

        (wt1, wt2, tsl,
         wt_buy_f, wt_sell_f,
         ut_buy, ut_sell,
         buy_sig, sell_sig,
         atr14, _) = self._build_signals(candles)

        n = len(candles) - 1
        if np.isnan(wt1[n]) or np.isnan(atr14[n]):
            return Signal(SignalType.HOLD, self.symbol, current_price, 0, "Indicator NaN")

        curr_wt1    = float(wt1[n])
        curr_wt2    = float(wt2[n]) if not np.isnan(wt2[n]) else 0.0
        current_atr = float(atr14[n])
        tsl_v       = float(tsl[n]) if not np.isnan(tsl[n]) else None
        p           = current_price
        conf        = round(min(0.90, 0.55 + abs(curr_wt1) / 200), 2)

        if buy_sig[n] and self._last_signal != 1:
            self._last_signal = 1
            sl_p = round(p - self.sl_atr_mult * current_atr, 4)
            tp_p = round(p + self.sl_atr_mult * self.rr_ratio * current_atr, 4)
            return Signal(
                type=SignalType.BUY, symbol=self.symbol,
                price=p, amount=0.0, confidence=conf,
                reason=f"[SJ+UT] BUY | WT1={curr_wt1:.1f} + UT confirm",
                metadata={
                    "wt1": round(curr_wt1, 2), "wt2": round(curr_wt2, 2),
                    "ut_tsl": round(tsl_v, 4) if tsl_v else None,
                    "atr": round(current_atr, 4),
                    "stop_loss": sl_p, "take_profit": tp_p, "rr": self.rr_ratio,
                },
            )

        if sell_sig[n] and self._last_signal != -1:
            self._last_signal = -1
            sl_p = round(p + self.sl_atr_mult * current_atr, 4)
            tp_p = round(p - self.sl_atr_mult * self.rr_ratio * current_atr, 4)
            return Signal(
                type=SignalType.SELL, symbol=self.symbol,
                price=p, amount=0.0, confidence=conf,
                reason=f"[SJ+UT] SELL | WT1={curr_wt1:.1f} + UT confirm",
                metadata={
                    "wt1": round(curr_wt1, 2), "wt2": round(curr_wt2, 2),
                    "ut_tsl": round(tsl_v, 4) if tsl_v else None,
                    "atr": round(current_atr, 4),
                    "stop_loss": sl_p, "take_profit": tp_p, "rr": self.rr_ratio,
                },
            )

        if not buy_sig[n] and not sell_sig[n]:
            self._last_signal = 0

        zone = (f"OB({curr_wt1:.1f})" if curr_wt1 > self.ob_level else
                f"OS({curr_wt1:.1f})" if curr_wt1 < self.os_level else
                f"neutral({curr_wt1:.1f})")
        return Signal(
            SignalType.HOLD, self.symbol, current_price, 0,
            f"[SJ+UT] {zone} | waiting dual confirm",
            metadata={"wt1": round(curr_wt1, 2), "wt2": round(curr_wt2, 2)},
        )

    # ── Backtest ───────────────────────────────────────────────────

    async def backtest(self, candles: list) -> tuple[dict, tuple]:
        min_len = self.n1 + self.n2 + max(_ATR_PERIOD, self.ut_atr_len) + 20
        if len(candles) < min_len:
            return {}, None

        cls  = np.array([c.close for c in candles], dtype=float)
        hig  = np.array([c.high  for c in candles], dtype=float)
        low  = np.array([c.low   for c in candles], dtype=float)

        (_, _, _, _, _,
         _, _,
         buy_sig, sell_sig,
         atr14, _) = self._build_signals(candles)

        signal_bars = []
        prev_dir = 0
        for i in range(min_len, len(candles) - 1):
            if np.isnan(atr14[i]):
                continue
            if buy_sig[i] and prev_dir != 1:
                signal_bars.append((i, 1, float(atr14[i])))
                prev_dir = 1
            elif sell_sig[i] and prev_dir != -1:
                signal_bars.append((i, -1, float(atr14[i])))
                prev_dir = -1
            elif not buy_sig[i] and not sell_sig[i]:
                prev_dir = 0

        if not signal_bars:
            return {}, None

        best_score, best_config = -999.0, None
        stats: dict = {}

        for sl_m in _SL_MULTS:
            rr = self.rr_ratio
            wins = losses = 0
            total_r = 0.0
            for idx, direction, atr_val in signal_bars:
                if atr_val <= 0:
                    continue
                entry = float(cls[idx])
                sl_p  = entry - sl_m * atr_val if direction ==  1 else entry + sl_m * atr_val
                tp_p  = entry + sl_m * rr * atr_val if direction == 1 else entry - sl_m * rr * atr_val
                outcome = 0
                for j in range(idx + 1, min(idx + _LOOKFORWARD, len(cls))):
                    if direction == 1:
                        if low[j]  <= sl_p: outcome = -1; break
                        if hig[j]  >= tp_p: outcome =  1; break
                    else:
                        if hig[j]  >= sl_p: outcome = -1; break
                        if low[j]  <= tp_p: outcome =  1; break
                if outcome ==  1: wins   += 1; total_r += rr
                elif outcome == -1: losses += 1; total_r -= 1.0

            total = wins + losses
            wr  = wins / total if total else 0.0
            pf  = (wins * rr) / max(losses, 1)
            key = f"SL={sl_m}xATR  RR=1:{rr}"
            stats[key] = {
                "win_rate": round(wr * 100, 1), "profit_factor": round(pf, 2),
                "total_r":  round(total_r, 1),  "trades": total,
                "wins": wins,                    "losses": losses,
            }
            if total >= 5 and total_r > best_score:
                best_score  = total_r
                best_config = (sl_m, rr)

        return stats, best_config
