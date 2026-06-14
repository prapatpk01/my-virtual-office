"""
Momentum Score Strategy — MC·DSA (8-indicator confluence).

8 Indicators scored -1 / 0 / +1:
  1. EMA 12/26       — price vs EMAs
  2. S/R             — pivot highs/lows (lookback=15)
  3. MACD 12/26/9    — line > signal AND hist rising
  4. RSI 7/14        — fast & slow both directional
  5. StochRSI 14/14/3/3
  6. ROC 12 vs MA6
  7. MFI 14
  8. ADX/DMI 14      (strong ≥ 25)

BUY:  bull_score ≥ 4  AND  ADX+DI+>DI-  AND  MACD bull
      TRIGGER: StochRSI K×D below 50 / MACD cross up / EMA cross up
SELL: bear_score ≥ 4  AND  ADX+DI->DI+  AND  MACD bear
      TRIGGER: StochRSI K×D above 50 / MACD cross down / EMA cross down

SL = 1.5×ATR,  TP = 1.5×1.5×ATR  (R:R 1:1.5)
MTF Bias gate controlled by env MTFV (default: true).
"""
import os
import logging
import numpy as np
from .base import BaseStrategy, Signal, SignalType

logger = logging.getLogger("momentum_score_strategy")

_ATR_P    = 14
_LOOKFWD  = 60
_SL_MULTS = [1.0, 1.5, 2.0, 2.5]


class MomentumScoreStrategy(BaseStrategy):

    def __init__(self, symbol: str, params: dict = None):
        super().__init__(symbol, params)
        # EMA
        self.ema_fast_len   = self.params.get("ema_fast_len",   12)
        self.ema_slow_len   = self.params.get("ema_slow_len",   26)
        # S/R pivot lookback
        self.sr_len         = self.params.get("sr_len",         15)
        # MACD
        self.macd_fast      = self.params.get("macd_fast",      12)
        self.macd_slow_p    = self.params.get("macd_slow",      26)
        self.macd_sig       = self.params.get("macd_signal",     9)
        # RSI
        self.rsi_fast_len   = self.params.get("rsi_fast_len",    7)
        self.rsi_slow_len   = self.params.get("rsi_slow_len",   14)
        # StochRSI
        self.stoch_len      = self.params.get("stoch_len",      14)
        self.stochrsi_len   = self.params.get("stochrsi_len",   14)
        self.stoch_k        = self.params.get("stoch_k",         3)
        self.stoch_d        = self.params.get("stoch_d",         3)
        self.stoch_ob       = self.params.get("stoch_ob",       80)
        self.stoch_os       = self.params.get("stoch_os",       20)
        # ROC
        self.roc_len        = self.params.get("roc_len",        12)
        self.roc_ma_len     = self.params.get("roc_ma_len",      6)
        # MFI
        self.mfi_len        = self.params.get("mfi_len",        14)
        self.mfi_ob         = self.params.get("mfi_ob",         80)
        self.mfi_os         = self.params.get("mfi_os",         20)
        # ADX
        self.adx_len        = self.params.get("adx_len",        14)
        self.adx_strong     = self.params.get("adx_strong",     25)
        # Signal threshold
        self.min_bull_score = self.params.get("min_bull_score",  4)
        self.min_bear_score = self.params.get("min_bear_score",  4)
        # SL/TP
        self.sl_atr_mult    = self.params.get("sl_atr_mult",   1.5)
        self.rr_ratio       = self.params.get("rr_ratio",      1.5)

    # ── Indicator helpers ──────────────────────────────────────────────

    def _stoch_rsi(self, closes: list) -> tuple[np.ndarray, np.ndarray]:
        """Stochastic applied to RSI (Pine: ta.stoch of rsi)."""
        rsi_arr = np.array(self.rsi(closes, self.stochrsi_len), dtype=float)
        n   = len(rsi_arr)
        raw = np.full(n, np.nan)
        for i in range(self.stoch_len - 1, n):
            w  = rsi_arr[max(0, i - self.stoch_len + 1): i + 1]
            lo = np.nanmin(w)
            hi = np.nanmax(w)
            raw[i] = (rsi_arr[i] - lo) / (hi - lo) * 100.0 if hi > lo else 50.0
        k = np.array(self.sma(raw.tolist(), self.stoch_k), dtype=float)
        d = np.array(self.sma(k.tolist(),   self.stoch_d), dtype=float)
        return k, d

    @staticmethod
    def _calc_roc(closes: np.ndarray, length: int) -> np.ndarray:
        """Rate of Change (%)."""
        out = np.full(len(closes), np.nan)
        for i in range(length, len(closes)):
            prev = closes[i - length]
            if prev != 0:
                out[i] = (closes[i] - prev) / prev * 100.0
        return out

    @staticmethod
    def _calc_mfi(candles: list, length: int) -> np.ndarray:
        """Money Flow Index."""
        n    = len(candles)
        hlc3 = np.array([(c.high + c.low + c.close) / 3.0 for c in candles])
        vol  = np.array([c.volume for c in candles], dtype=float)
        mf   = hlc3 * vol
        out  = np.full(n, np.nan)
        for i in range(length, n):
            pos = neg = 0.0
            for j in range(i - length + 1, i + 1):
                if hlc3[j] > hlc3[j - 1]:
                    pos += mf[j]
                elif hlc3[j] < hlc3[j - 1]:
                    neg += mf[j]
            out[i] = 100.0 if neg == 0 else 100.0 - 100.0 / (1.0 + pos / neg)
        return out

    def _pivot_sr(self, highs: np.ndarray, lows: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray]:
        """
        Causal pivot-based S/R.
        Pivot high confirmed at bar j = i - sr_len, using window [j-sr_len, j+sr_len].
        Levels persist forward until next pivot.
        """
        n, L     = len(highs), self.sr_len
        res      = np.full(n, np.nan)
        sup      = np.full(n, np.nan)
        last_res = last_sup = np.nan
        for i in range(2 * L, n):
            j   = i - L
            w_h = highs[j - L: j + L + 1]
            w_l = lows[j - L:  j + L + 1]
            if len(w_h) == 2 * L + 1 and highs[j] == np.max(w_h):
                last_res = highs[j]
            if len(w_l) == 2 * L + 1 and lows[j]  == np.min(w_l):
                last_sup = lows[j]
            res[i] = last_res
            sup[i] = last_sup
        return res, sup

    # ── Score helpers ──────────────────────────────────────────────────

    @staticmethod
    def _sr_score(price: float, res: float, sup: float) -> int:
        res_nan = np.isnan(res)
        sup_nan = np.isnan(sup)
        if res_nan and sup_nan:
            return 0
        near_res  = (not res_nan) and abs(price - res) / price < 0.005
        near_sup  = (not sup_nan) and abs(price - sup) / price < 0.005
        above_res = (not res_nan) and price > res
        below_sup = (not sup_nan) and price < sup
        if above_res or near_sup:  return  1
        if below_sup or near_res:  return -1
        return 0

    def _all_indicators(self, candles: list):
        """Compute and return all indicator arrays. Shared by analyze + backtest."""
        closes = [c.close for c in candles]
        cn     = np.array(closes, dtype=float)
        hn     = np.array([c.high for c in candles], dtype=float)
        ln     = np.array([c.low  for c in candles], dtype=float)

        ef  = np.array(self.ema(closes, self.ema_fast_len),  dtype=float)
        es  = np.array(self.ema(closes, self.ema_slow_len),  dtype=float)
        _ml, _sl, _hi = self.macd(closes, self.macd_fast, self.macd_slow_p, self.macd_sig)
        ml  = np.array(_ml, dtype=float)
        sl  = np.array(_sl, dtype=float)
        hi  = np.array(_hi, dtype=float)
        rf  = np.array(self.rsi(closes, self.rsi_fast_len),  dtype=float)
        rs  = np.array(self.rsi(closes, self.rsi_slow_len),  dtype=float)
        k_a, d_a = self._stoch_rsi(closes)
        roc = self._calc_roc(cn, self.roc_len)
        rocm= np.array(self.sma(roc.tolist(), self.roc_ma_len), dtype=float)
        mfi = self._calc_mfi(candles, self.mfi_len)
        _adx, _dip, _dim = self.adx(candles, self.adx_len)
        adx = np.array(_adx, dtype=float)
        dip = np.array(_dip, dtype=float)
        dim = np.array(_dim, dtype=float)
        atr = np.array(self.atr(candles, _ATR_P), dtype=float)
        res, sup = self._pivot_sr(hn, ln)

        return ef, es, ml, sl, hi, rf, rs, k_a, d_a, roc, rocm, mfi, adx, dip, dim, atr, res, sup, cn, hn, ln

    def _bar_scores(self, i, ef, es, ml, sl, hi, rf, rs, k_a, d_a,
                    roc, rocm, mfi, adx, dip, dim, closes):
        """Compute 8 scores and momentum/trigger flags at bar i."""
        p = float(closes[i])

        ema_bull = ef[i] > es[i] and p > ef[i]
        ema_bear = ef[i] < es[i] and p < ef[i]
        ema_sc   = 1 if ema_bull else (-1 if ema_bear else 0)

        macd_bull = ml[i] > sl[i] and hi[i] > 0 and hi[i] > hi[i-1]
        macd_bear = ml[i] < sl[i] and hi[i] < 0 and hi[i] < hi[i-1]
        macd_sc   = 1 if macd_bull else (-1 if macd_bear else 0)

        rsi_bull  = rf[i] > 50 and rs[i] > 50 and rf[i] > rf[i-1]
        rsi_bear  = rf[i] < 50 and rs[i] < 50 and rf[i] < rf[i-1]
        rsi_sc    = 1 if rsi_bull else (-1 if rsi_bear else 0)

        stoch_bull = k_a[i] > d_a[i] and k_a[i] < self.stoch_ob and k_a[i] > k_a[i-1]
        stoch_bear = k_a[i] < d_a[i] and k_a[i] > self.stoch_os and k_a[i] < k_a[i-1]
        stoch_sc   = 1 if stoch_bull else (-1 if stoch_bear else 0)

        roc_bull = roc[i] > 0 and roc[i] > rocm[i]
        roc_bear = roc[i] < 0 and roc[i] < rocm[i]
        roc_sc   = 1 if roc_bull else (-1 if roc_bear else 0)

        mfi_bull = mfi[i] > 50 and mfi[i] > mfi[i-1] and mfi[i] < self.mfi_ob
        mfi_bear = mfi[i] < 50 and mfi[i] < mfi[i-1] and mfi[i] > self.mfi_os
        mfi_sc   = 1 if mfi_bull else (-1 if mfi_bear else 0)

        adx_v = float(adx[i]) if not np.isnan(adx[i]) else 0.0
        dip_v = float(dip[i]) if not np.isnan(dip[i]) else 0.0
        dim_v = float(dim[i]) if not np.isnan(dim[i]) else 0.0
        adx_bull = adx_v >= self.adx_strong and dip_v > dim_v
        adx_bear = adx_v >= self.adx_strong and dim_v > dip_v
        adx_sc   = 1 if adx_bull else (-1 if adx_bear else 0)

        return (ema_sc, macd_sc, macd_bull, macd_bear,
                rsi_sc, stoch_sc, roc_sc, mfi_sc, adx_sc,
                adx_bull, adx_bear, adx_v, dip_v, dim_v)

    # ── Live analysis ──────────────────────────────────────────────────

    async def analyze(self, candles: list, current_price: float,
                      mtf_candles: dict = None) -> Signal:
        min_len = (max(2 * self.sr_len + 2,
                       self.macd_slow_p + self.macd_sig + self.rsi_slow_len,
                       self.stochrsi_len + self.stoch_len + self.stoch_k + self.stoch_d,
                       self.mfi_len, self.adx_len * 2) + 10)
        if len(candles) < min_len:
            return Signal(SignalType.HOLD, self.symbol, current_price, 0, "Not enough data")

        (ef, es, ml, sl_a, hi, rf, rs, k_a, d_a,
         roc, rocm, mfi, adx, dip, dim, atr, res, sup, cn, hn, ln) = self._all_indicators(candles)

        n = len(candles) - 1  # current bar

        check = [ef[n], es[n], ml[n], sl_a[n], hi[n], rf[n], rs[n],
                 k_a[n], d_a[n], roc[n], rocm[n], mfi[n], atr[n]]
        if any(np.isnan(v) for v in check):
            return Signal(SignalType.HOLD, self.symbol, current_price, 0, "Indicator NaN")

        closes = cn

        # S/R score uses current price
        sr_sc = self._sr_score(current_price, float(res[n]), float(sup[n]))

        # Other 7 scores
        (ema_sc, macd_sc, macd_bull, macd_bear,
         rsi_sc, stoch_sc, roc_sc, mfi_sc, adx_sc,
         adx_bull, adx_bear, adx_v, dip_v, dim_v) = self._bar_scores(
            n, ef, es, ml, sl_a, hi, rf, rs, k_a, d_a, roc, rocm, mfi, adx, dip, dim, closes)

        all_scores = [ema_sc, sr_sc, macd_sc, rsi_sc, stoch_sc, roc_sc, mfi_sc, adx_sc]
        bull_score = sum(1 for s in all_scores if s ==  1)
        bear_score = sum(1 for s in all_scores if s == -1)
        net_score  = bull_score - bear_score

        momentum_bull = bull_score >= self.min_bull_score and adx_bull and macd_bull
        momentum_bear = bear_score >= self.min_bear_score and adx_bear and macd_bear

        # Cross triggers
        stoch_x_bull = k_a[n-1] <= d_a[n-1] and k_a[n] > d_a[n] and k_a[n-1] < 50
        stoch_x_bear = k_a[n-1] >= d_a[n-1] and k_a[n] < d_a[n] and k_a[n-1] > 50
        macd_x_bull  = ml[n-1] <= sl_a[n-1]  and ml[n] > sl_a[n]
        macd_x_bear  = ml[n-1] >= sl_a[n-1]  and ml[n] < sl_a[n]
        ema_x_bull   = ef[n-1] <= es[n-1]     and ef[n] > es[n]
        ema_x_bear   = ef[n-1] >= es[n-1]     and ef[n] < es[n]
        trig_bull = stoch_x_bull or macd_x_bull or ema_x_bull
        trig_bear = stoch_x_bear or macd_x_bear or ema_x_bear

        signal_buy  = momentum_bull and trig_bull
        signal_sell = momentum_bear and trig_bear

        # ── MTFV gate ────────────────────────────────────────────────
        comp_pct, mtf_label = BaseStrategy.compute_mtf_bias(candles, mtf_candles)
        if os.getenv("MTFV", "true").lower() == "true":
            if signal_buy and comp_pct < 33:
                return Signal(SignalType.HOLD, self.symbol, current_price, 0,
                              f"[MTF BLOCK] {mtf_label}",
                              metadata={"bull_score": bull_score,
                                        "mtf_comp_pct": round(comp_pct, 1)})
            if signal_sell and comp_pct > -33:
                return Signal(SignalType.HOLD, self.symbol, current_price, 0,
                              f"[MTF BLOCK] {mtf_label}",
                              metadata={"bear_score": bear_score,
                                        "mtf_comp_pct": round(comp_pct, 1)})

        atr_c = float(atr[n])
        conf  = round(0.55 + max(bull_score, bear_score) / 8.0 * 0.40, 2)
        meta  = {
            "bull_score": bull_score, "bear_score": bear_score, "net": net_score,
            "ema":  f"{ef[n]:.4f}/{es[n]:.4f}",
            "macd_hist": round(float(hi[n]), 5),
            "rsi":  f"{rf[n]:.1f}/{rs[n]:.1f}",
            "stoch_k": round(float(k_a[n]), 1),
            "adx":  round(adx_v, 1), "dip": round(dip_v, 1), "dim": round(dim_v, 1),
            "mfi":  round(float(mfi[n]), 1),
            "roc":  round(float(roc[n]), 2),
            "atr":  round(atr_c, 4),
            "mtf":  mtf_label, "mtf_comp_pct": round(comp_pct, 1),
        }

        p = current_price
        if signal_buy:
            trig = "StochX" if stoch_x_bull else ("MACDX" if macd_x_bull else "EMAX")
            sl_p = round(p - self.sl_atr_mult * atr_c, 4)
            tp_p = round(p + self.sl_atr_mult * self.rr_ratio * atr_c, 4)
            return Signal(
                type=SignalType.BUY, symbol=self.symbol, price=p, amount=0.0,
                confidence=conf,
                reason=f"[MC {bull_score}/8] {trig} | MTF✓ {mtf_label}",
                metadata={**meta, "stop_loss": sl_p, "take_profit": tp_p,
                          "rr": self.rr_ratio, "trigger": trig},
            )

        if signal_sell:
            trig = "StochX" if stoch_x_bear else ("MACDX" if macd_x_bear else "EMAX")
            sl_p = round(p + self.sl_atr_mult * atr_c, 4)
            tp_p = round(p - self.sl_atr_mult * self.rr_ratio * atr_c, 4)
            return Signal(
                type=SignalType.SELL, symbol=self.symbol, price=p, amount=0.0,
                confidence=conf,
                reason=f"[MC {bear_score}/8] {trig} | MTF✓ {mtf_label}",
                metadata={**meta, "stop_loss": sl_p, "take_profit": tp_p,
                          "rr": self.rr_ratio, "trigger": trig},
            )

        # HOLD
        bias    = f"BULL {bull_score}/8" if bull_score >= bear_score else f"BEAR {bear_score}/8"
        mom_str = "mom✓" if (momentum_bull or momentum_bear) else "no-mom"
        return Signal(
            SignalType.HOLD, self.symbol, current_price, 0,
            f"[MC] {bias} | {mom_str} | no trigger",
            metadata={"bull_score": bull_score, "bear_score": bear_score,
                      "adx": round(adx_v, 1), "mfi": round(float(mfi[n]), 1)},
        )

    # ── Backtest ───────────────────────────────────────────────────────

    async def backtest(self, candles: list) -> tuple[dict, tuple]:
        min_len = (max(2 * self.sr_len + 2,
                       self.macd_slow_p + self.macd_sig + self.rsi_slow_len,
                       self.stochrsi_len + self.stoch_len + self.stoch_k + self.stoch_d,
                       self.mfi_len, self.adx_len * 2) + 10)
        if len(candles) < min_len + 20:
            return {}, None

        (ef, es, ml, sl_a, hi, rf, rs, k_a, d_a,
         roc, rocm, mfi, adx, dip, dim, atr, res, sup, cn, hn, ln) = self._all_indicators(candles)

        closes = cn
        signal_bars: list[tuple[int, int, float]] = []
        prev_dir = 0

        for i in range(min_len, len(candles) - 1):
            check = [ef[i], es[i], ml[i], sl_a[i], hi[i], rf[i], rs[i],
                     k_a[i], d_a[i], roc[i], rocm[i], mfi[i], atr[i]]
            if any(np.isnan(v) for v in check):
                continue

            sr_sc = self._sr_score(float(closes[i]), float(res[i]), float(sup[i]))

            (ema_sc, macd_sc, macd_bull, macd_bear,
             rsi_sc, stoch_sc, roc_sc, mfi_sc, adx_sc,
             adx_bull, adx_bear, *_) = self._bar_scores(
                i, ef, es, ml, sl_a, hi, rf, rs, k_a, d_a, roc, rocm, mfi, adx, dip, dim, closes)

            scores     = [ema_sc, sr_sc, macd_sc, rsi_sc, stoch_sc, roc_sc, mfi_sc, adx_sc]
            bull_score = sum(1 for s in scores if s ==  1)
            bear_score = sum(1 for s in scores if s == -1)

            mom_bull = bull_score >= self.min_bull_score and adx_bull and macd_bull
            mom_bear = bear_score >= self.min_bear_score and adx_bear and macd_bear

            trig_bull = ((k_a[i-1] <= d_a[i-1] and k_a[i] > d_a[i] and k_a[i-1] < 50) or
                         (ml[i-1]  <= sl_a[i-1] and ml[i]  > sl_a[i]) or
                         (ef[i-1]  <= es[i-1]   and ef[i]  > es[i]))
            trig_bear = ((k_a[i-1] >= d_a[i-1] and k_a[i] < d_a[i] and k_a[i-1] > 50) or
                         (ml[i-1]  >= sl_a[i-1] and ml[i]  < sl_a[i]) or
                         (ef[i-1]  >= es[i-1]   and ef[i]  < es[i]))

            buy  = mom_bull and trig_bull
            sell = mom_bear and trig_bear

            if buy and prev_dir != 1:
                signal_bars.append((i, 1, float(atr[i])))
                prev_dir = 1
            elif sell and prev_dir != -1:
                signal_bars.append((i, -1, float(atr[i])))
                prev_dir = -1
            elif not buy and not sell:
                prev_dir = 0

        if not signal_bars:
            return {}, None

        best_score, best_config = -999.0, None
        stats: dict[str, dict] = {}

        for sl_m in _SL_MULTS:
            rr = self.rr_ratio
            wins = losses = 0
            total_r = 0.0
            for idx, direction, atr_val in signal_bars:
                if atr_val <= 0:
                    continue
                entry = float(closes[idx])
                sl_p  = entry - sl_m * atr_val if direction == 1 else entry + sl_m * atr_val
                tp_p  = entry + sl_m * rr * atr_val if direction == 1 else entry - sl_m * rr * atr_val
                outcome = 0
                for j in range(idx + 1, min(idx + _LOOKFWD, len(candles))):
                    if direction == 1:
                        if ln[j] <= sl_p: outcome = -1; break
                        if hn[j] >= tp_p: outcome =  1; break
                    else:
                        if hn[j] >= sl_p: outcome = -1; break
                        if ln[j] <= tp_p: outcome =  1; break
                if outcome ==  1: wins   += 1; total_r += rr
                elif outcome == -1: losses += 1; total_r -= 1.0

            total = wins + losses
            wr    = wins / total if total else 0.0
            pf    = (wins * rr) / max(losses, 1)
            key   = f"SL={sl_m}xATR  RR=1:{rr}"
            stats[key] = {
                "win_rate": round(wr * 100, 1), "profit_factor": round(pf, 2),
                "total_r":  round(total_r, 1),  "trades": total,
                "wins":     wins,                "losses": losses,
            }
            if total >= 5 and total_r > best_score:
                best_score = total_r
                best_config = (sl_m, rr)

        return stats, best_config
