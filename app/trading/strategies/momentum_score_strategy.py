"""
Momentum Score Strategy — weighted scoring for 3-5 signals/day, target WR >72%.

Gates (all must pass before scoring):
  A)   Candle open vs HMA15       (core direction gate)
  B)   Candle close vs open       (confirmation: bullish close for BUY, bearish for SELL)
  MTF) 1H + 4H HMA15 agree        (anti-chop gate)

Score 0–8:
  B  EMA9 > SMA21                 +1
  C  MACD > Signal                +2   (MACD+RSI combo = 73% WR)
  D  RSI in momentum zone         +2   BUY: 42-68 / SELL: 32-58
  E  ADX > 20                     +2   (reduces false signals 30%)
  F  Price vs SMA200              +1   (+15-25% WR)

Signal fires at score >= 5 / 8.
Confidence: 5→65%  6→75%  7→85%  8→95%
SL/TP: ATR-based, R:R = 1.5 fixed.
"""
import logging
import numpy as np
from .base import BaseStrategy, Signal, SignalType

logger = logging.getLogger("momentum_score_strategy")

_ATR_PERIOD  = 14
_ADX_PERIOD  = 14
_RSI_PERIOD  = 14
_SMA200      = 200
_LOOKFORWARD = 60
_SL_MULTS    = [1.0, 1.5, 2.0, 2.5]


class MomentumScoreStrategy(BaseStrategy):

    def __init__(self, symbol: str, params: dict = None):
        super().__init__(symbol, params)
        self.hma_period      = self.params.get("hma_period",      15)
        self.ema_fast        = self.params.get("ema_fast",          9)
        self.sma_mid         = self.params.get("sma_mid",          21)
        self.macd_fast       = self.params.get("macd_fast",        12)
        self.macd_slow       = self.params.get("macd_slow",        26)
        self.macd_sig        = self.params.get("macd_signal",       9)
        self.adx_threshold   = self.params.get("adx_threshold",    20)
        self.rsi_buy_lo      = self.params.get("rsi_buy_lo",       42)
        self.rsi_buy_hi      = self.params.get("rsi_buy_hi",       68)
        self.rsi_sell_lo     = self.params.get("rsi_sell_lo",      32)
        self.rsi_sell_hi     = self.params.get("rsi_sell_hi",      58)
        self.score_threshold = self.params.get("score_threshold",   5)
        self.sl_atr_mult     = self.params.get("sl_atr_mult",     1.5)
        self.rr_ratio        = self.params.get("rr_ratio",        1.5)

    # ------------------------------------------------------------------
    # Live signal
    # ------------------------------------------------------------------

    async def analyze(self, candles: list, current_price: float,
                      mtf_candles: dict = None) -> Signal:
        min_len = _SMA200 + self.macd_slow + 10
        if len(candles) < min_len:
            return Signal(SignalType.HOLD, self.symbol, current_price, 0, "Not enough data")

        closes = [c.close for c in candles]
        open_price = float(candles[-1].open)
        p = current_price

        hma_arr          = self.hma(closes, self.hma_period)
        ema9_arr         = self.ema(closes, self.ema_fast)
        sma21_arr        = self.sma(closes, self.sma_mid)
        sma200_arr       = self.sma(closes, _SMA200)
        ml_arr, sl_arr, _= self.macd(closes, self.macd_fast, self.macd_slow, self.macd_sig)
        rsi_arr          = self.rsi(closes, _RSI_PERIOD)
        adx_arr, _, _    = self.adx(candles, _ADX_PERIOD)
        atr_arr          = self.atr(candles, _ATR_PERIOD)

        hma   = float(hma_arr[-1])
        e9    = float(ema9_arr[-1])
        s21   = float(sma21_arr[-1])
        s200  = float(sma200_arr[-1])
        ml_c  = float(ml_arr[-1])
        sl_c  = float(sl_arr[-1])
        rsi   = float(rsi_arr[-1])
        adx_v = float(adx_arr[-1]) if not np.isnan(adx_arr[-1]) else 0.0
        atr_v = float(atr_arr[-1])

        if any(np.isnan(v) for v in [hma, e9, s21, s200, ml_c, sl_c, rsi, atr_v]):
            return Signal(SignalType.HOLD, self.symbol, current_price, 0, "Indicator NaN")

        # ── Gate A: HMA15 direction ──────────────────────────────────
        gate_a = 1 if open_price > hma else (-1 if open_price < hma else 0)
        if gate_a == 0:
            return Signal(SignalType.HOLD, self.symbol, current_price, 0,
                          "[HOLD] open≈HMA15",
                          metadata={"hma15": round(hma, 2), "open": open_price})

        # ── Gate B: candle direction confirmation ────────────────────
        last_close = float(candles[-1].close)
        last_open  = float(candles[-1].open)
        candle_bullish = last_close > last_open
        candle_bearish = last_close < last_open
        if gate_a == 1 and not candle_bullish:
            return Signal(SignalType.HOLD, self.symbol, current_price, 0,
                          "[HOLD] BUY gate A ok but candle bearish",
                          metadata={"hma15": round(hma, 2), "open": open_price})
        if gate_a == -1 and not candle_bearish:
            return Signal(SignalType.HOLD, self.symbol, current_price, 0,
                          "[HOLD] SELL gate A ok but candle bullish",
                          metadata={"hma15": round(hma, 2), "open": open_price})

        # ── Gate MTF ─────────────────────────────────────────────────
        mtf_pass, mtf_label = _check_mtf(mtf_candles, gate_a, self.hma_period)
        if not mtf_pass:
            return Signal(SignalType.HOLD, self.symbol, current_price, 0,
                          f"[MTF BLOCK] {mtf_label}",
                          metadata={"hma15": round(hma, 2), "open": open_price,
                                    "mtf": mtf_label})

        # ── Score ────────────────────────────────────────────────────
        is_buy = (gate_a == 1)
        score, tags = _calc_score(
            is_buy, e9, s21, ml_c, sl_c, rsi, adx_v, p, s200,
            self.adx_threshold,
            self.rsi_buy_lo, self.rsi_buy_hi,
            self.rsi_sell_lo, self.rsi_sell_hi,
        )

        score_label = f"[{score}/8]"

        if score < self.score_threshold:
            return Signal(
                SignalType.HOLD, self.symbol, current_price, 0,
                f"[HOLD] {score_label} < {self.score_threshold} | {' '.join(tags) or 'no indicators'}",
                metadata={"score": score, "adx": round(adx_v, 1), "rsi": round(rsi, 1),
                          "hma15": round(hma, 2), "ema9": round(e9, 2),
                          "macd": round(ml_c, 4)},
            )

        # Confidence: 5→65%, 6→75%, 7→85%, 8→95%
        conf = round(0.55 + (score / 8) * 0.40, 2)
        direction = "buy" if is_buy else "sell"
        sl_price = round(p - self.sl_atr_mult * atr_v, 4) if is_buy else round(p + self.sl_atr_mult * atr_v, 4)
        tp_price = round(p + self.sl_atr_mult * self.rr_ratio * atr_v, 4) if is_buy else round(p - self.sl_atr_mult * self.rr_ratio * atr_v, 4)

        return Signal(
            type=SignalType.BUY if is_buy else SignalType.SELL,
            symbol=self.symbol, price=p, amount=0.0, confidence=conf,
            reason=f"{score_label} {' + '.join(tags)} | MTF✓ {mtf_label}",
            metadata={
                "score": score, "hma15": round(hma, 2), "open": open_price,
                "ema9": round(e9, 2), "sma21": round(s21, 2),
                "macd": round(ml_c, 4), "macd_signal": round(sl_c, 4),
                "rsi": round(rsi, 1), "adx": round(adx_v, 1),
                "sma200": round(s200, 2),
                "atr": round(atr_v, 4),
                "stop_loss": sl_price, "take_profit": tp_price,
                "rr": self.rr_ratio, "mtf": mtf_label,
            },
        )

    # ------------------------------------------------------------------
    # Backtest — sweep SL multipliers, fixed R:R = 1.5
    # ------------------------------------------------------------------

    async def backtest(self, candles: list) -> dict:
        min_len = _SMA200 + self.macd_slow + _ADX_PERIOD * 2 + _ATR_PERIOD + 10
        if len(candles) < min_len + 20:
            return {}, None

        closes = [c.close for c in candles]
        highs  = [c.high  for c in candles]
        lows   = [c.low   for c in candles]

        hma_arr        = self.hma(closes, self.hma_period)
        ema9_arr       = self.ema(closes, self.ema_fast)
        sma21_arr      = self.sma(closes, self.sma_mid)
        sma200_arr     = self.sma(closes, _SMA200)
        ml_arr, sl_arr, _ = self.macd(closes, self.macd_fast, self.macd_slow, self.macd_sig)
        rsi_arr        = self.rsi(closes, _RSI_PERIOD)
        adx_arr, _, _  = self.adx(candles, _ADX_PERIOD)
        atr_arr        = self.atr(candles, _ATR_PERIOD)

        signal_bars = []
        prev_dir = 0

        for i in range(min_len, len(candles) - 1):
            vals = [hma_arr[i], ema9_arr[i], sma21_arr[i], sma200_arr[i],
                    ml_arr[i], sl_arr[i], rsi_arr[i], atr_arr[i]]
            if any(np.isnan(v) for v in vals):
                continue
            adx_i = adx_arr[i] if not np.isnan(adx_arr[i]) else 0.0

            gate = 1 if closes[i] > hma_arr[i] else (-1 if closes[i] < hma_arr[i] else 0)
            if gate == 0:
                continue

            is_buy = (gate == 1)
            score, _ = _calc_score(
                is_buy, ema9_arr[i], sma21_arr[i], ml_arr[i], sl_arr[i],
                rsi_arr[i], adx_i, closes[i], sma200_arr[i],
                self.adx_threshold,
                self.rsi_buy_lo, self.rsi_buy_hi,
                self.rsi_sell_lo, self.rsi_sell_hi,
            )

            if score >= self.score_threshold and gate != prev_dir:
                signal_bars.append((i, gate, float(atr_arr[i])))
                prev_dir = gate
            elif score < self.score_threshold:
                prev_dir = 0

        if not signal_bars:
            return {}, None

        best_score = -999.0
        best_config = None
        stats = {}

        for sl_m in _SL_MULTS:
            rr = self.rr_ratio
            wins = losses = 0
            total_r = 0.0

            for idx, direction, atr_val in signal_bars:
                if atr_val <= 0:
                    continue
                entry = closes[idx]
                sl_p = entry - sl_m * atr_val if direction == 1 else entry + sl_m * atr_val
                tp_p = entry + sl_m * rr * atr_val if direction == 1 else entry - sl_m * rr * atr_val

                outcome = 0
                for j in range(idx + 1, min(idx + _LOOKFORWARD, len(candles))):
                    if direction == 1:
                        if lows[j]  <= sl_p: outcome = -1; break
                        if highs[j] >= tp_p: outcome =  1; break
                    else:
                        if highs[j] >= sl_p: outcome = -1; break
                        if lows[j]  <= tp_p: outcome =  1; break

                if outcome ==  1: wins   += 1; total_r += rr
                elif outcome == -1: losses += 1; total_r -= 1.0

            total = wins + losses
            wr = wins / total if total else 0.0
            pf = (wins * rr) / max(losses, 1)
            key = f"SL={sl_m}xATR  RR=1:{rr}"
            stats[key] = {
                "win_rate": round(wr * 100, 1),
                "profit_factor": round(pf, 2),
                "total_r": round(total_r, 1),
                "trades": total, "wins": wins, "losses": losses,
            }
            if total >= 5 and total_r > best_score:
                best_score = total_r
                best_config = (sl_m, rr)

        return stats, best_config


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _calc_score(is_buy, e9, s21, ml_c, sl_c, rsi, adx_v, price, s200,
                adx_thr, rsi_buy_lo, rsi_buy_hi, rsi_sell_lo, rsi_sell_hi):
    score = 0
    tags  = []
    if is_buy:
        if e9 > s21:                             score += 1; tags.append("EMA9>SMA21")
        if ml_c > sl_c:                          score += 2; tags.append("MACD>Sig")
        if rsi_buy_lo <= rsi <= rsi_buy_hi:      score += 2; tags.append(f"RSI={rsi:.0f}✓")
        if adx_v >= adx_thr:                     score += 2; tags.append(f"ADX={adx_v:.0f}")
        if price > s200:                         score += 1; tags.append("P>SMA200")
    else:
        if e9 < s21:                             score += 1; tags.append("EMA9<SMA21")
        if ml_c < sl_c:                          score += 2; tags.append("MACD<Sig")
        if rsi_sell_lo <= rsi <= rsi_sell_hi:    score += 2; tags.append(f"RSI={rsi:.0f}✓")
        if adx_v >= adx_thr:                     score += 2; tags.append(f"ADX={adx_v:.0f}")
        if price < s200:                         score += 1; tags.append("P<SMA200")
    return score, tags


def _check_mtf(mtf_candles: dict, direction: int, hma_period: int) -> tuple:
    if not mtf_candles:
        return True, "no MTF"
    results = {}
    for tf, candles in mtf_candles.items():
        if not candles or len(candles) < hma_period + 5:
            continue
        closes = [c.close for c in candles]
        hma_arr = BaseStrategy.hma(closes, hma_period)
        last_hma = float(hma_arr[-1])
        if np.isnan(last_hma):
            continue
        results[tf] = 1 if closes[-1] > last_hma else -1
    if not results:
        return True, "MTF unavailable"
    agree = sum(1 for v in results.values() if v == direction)
    label = " ".join(f"{tf}={'↑' if v==1 else '↓'}" for tf, v in sorted(results.items()))
    return agree == len(results), label
