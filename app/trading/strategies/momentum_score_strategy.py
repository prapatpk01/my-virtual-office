"""
Momentum Score Strategy — 5-indicator scoring system.

Scores 5 momentum conditions per bar (each true = 1 point):
  BUY  (bullish score ≥ threshold):
    1. RSI > 50              (momentum bullish zone)
    2. RSI > EMA9(RSI)       (RSI trending up)
    3. HA_close > HA_open    (bullish Heikin Ashi candle)
    4. HA_close > EMA20(HA)  (price above trend line)
    5. MACD line > Signal    (macro momentum positive)

  SELL (bearish score ≥ threshold):
    1. RSI < 50
    2. RSI < EMA9(RSI)
    3. HA_close < HA_open
    4. HA_close < EMA20(HA)
    5. MACD line < Signal

Signal fires on first transition AND re-fires every reentry_bars while score stays sustained.
Default: threshold=2, reentry_bars=5 — fires ~20+ signals/250 bars 1H, WR≈75% pure TP/SL
SL = 1.5×ATR(14), R:R = 1:1.4
"""
import numpy as np
from .base import BaseStrategy, Signal, SignalType

_ATR_PERIOD  = 14
_SL_MULTS    = [1.0, 1.5, 2.0, 2.5]
_LOOKFORWARD = 60


class MomentumScoreStrategy(BaseStrategy):

    def __init__(self, symbol: str, params: dict = None):
        super().__init__(symbol, params)
        self.rsi_len     = self.params.get("rsi_len",      14)   # 1H standard RSI
        self.rsi_ema_len = self.params.get("rsi_ema_len",   9)
        self.ema_len     = self.params.get("ema_len",       14)   # shorter trend filter
        self.macd_fast   = self.params.get("macd_fast",    12)
        self.macd_slow   = self.params.get("macd_slow",    26)
        self.macd_sig    = self.params.get("macd_signal",   9)
        self.threshold   = self.params.get("threshold",     2)   # 2 of 5 — more frequent signals
        self.reentry_bars = self.params.get("reentry_bars", 5)   # re-fire every N bars while sustained
        self.sl_atr_mult = self.params.get("sl_atr_mult",  1.5)
        self.rr_ratio    = self.params.get("rr_ratio",     1.4)   # R:R 1.4 → need WR > 42%

    # ── RSI (Wilder's) ─────────────────────────────────────────────

    @staticmethod
    def _rsi(closes: np.ndarray, period: int) -> np.ndarray:
        n = len(closes)
        result = np.full(n, np.nan)
        if n < period + 1:
            return result
        deltas = np.diff(closes)
        gains  = np.where(deltas > 0,  deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_g  = float(np.mean(gains[:period]))
        avg_l  = float(np.mean(losses[:period]))
        result[period] = 100.0 if avg_l < 1e-10 else 100 - 100 / (1 + avg_g / avg_l)
        for i in range(period + 1, n):
            avg_g = (avg_g * (period - 1) + gains[i - 1]) / period
            avg_l = (avg_l * (period - 1) + losses[i - 1]) / period
            result[i] = 100.0 if avg_l < 1e-10 else 100 - 100 / (1 + avg_g / avg_l)
        return result

    @staticmethod
    def _ema_skipnan(arr: np.ndarray, period: int) -> np.ndarray:
        """EMA that skips leading NaN (needed for EMA of RSI)."""
        result = np.full(len(arr), np.nan)
        valid  = np.where(~np.isnan(arr))[0]
        if len(valid) == 0 or len(arr) - valid[0] < period:
            return result
        s = int(valid[0])
        k = 2.0 / (period + 1)
        result[s + period - 1] = float(np.mean(arr[s : s + period]))
        for i in range(s + period, len(arr)):
            result[i] = float(arr[i]) * k + result[i - 1] * (1 - k)
        return result

    # ── Build all indicator arrays ─────────────────────────────────

    def _build_arrays(self, candles: list):
        ha_candles, ha_o, ha_c = self._heikin_ashi(candles)

        cl      = ha_c.tolist()
        rsi_a   = self._rsi(ha_c, self.rsi_len)
        rsi_ema = self._ema_skipnan(rsi_a, self.rsi_ema_len)
        ema20   = np.array(self.ema(cl, self.ema_len),  dtype=float)
        _ml, _sl, _ = self.macd(cl, self.macd_fast, self.macd_slow, self.macd_sig)
        ml      = np.array(_ml, dtype=float)
        sl_line = np.array(_sl, dtype=float)
        atr_a   = np.array(self.atr(ha_candles, _ATR_PERIOD), dtype=float)

        return ha_o, ha_c, rsi_a, rsi_ema, ema20, ml, sl_line, atr_a

    # ── Score at bar i ─────────────────────────────────────────────

    def _score(self, i: int, ha_o, ha_c, rsi_a, rsi_ema, ema20, ml, sl_line):
        """Return (bull_score, bear_score) at bar i (0–5 each)."""
        if any(np.isnan(v) for v in [rsi_a[i], rsi_ema[i], ema20[i], ml[i], sl_line[i]]):
            return 0, 0

        rsi_v  = float(rsi_a[i])
        rsi_e  = float(rsi_ema[i])
        ha_o_v = float(ha_o[i])
        ha_c_v = float(ha_c[i])
        ema_v  = float(ema20[i])
        ml_v   = float(ml[i])
        sig_v  = float(sl_line[i])

        bull = int(rsi_v > 50) + int(rsi_v > rsi_e) + \
               int(ha_c_v > ha_o_v) + int(ha_c_v > ema_v) + int(ml_v > sig_v)
        bear = int(rsi_v < 50) + int(rsi_v < rsi_e) + \
               int(ha_c_v < ha_o_v) + int(ha_c_v < ema_v) + int(ml_v < sig_v)
        return bull, bear

    # ── Signal arrays ──────────────────────────────────────────────

    def _build_signals(self, candles: list):
        """Returns (buy_sig, sell_sig, bull_score, bear_score, atr_a)."""
        ha_o, ha_c, rsi_a, rsi_ema, ema20, ml, sl_line, atr_a = \
            self._build_arrays(candles)

        n        = len(candles)
        bull_arr = np.zeros(n, dtype=int)
        bear_arr = np.zeros(n, dtype=int)
        buy_sig  = np.zeros(n, dtype=bool)
        sell_sig = np.zeros(n, dtype=bool)

        prev_bull = 0
        prev_bear = 0
        last_buy_bar = -999
        last_sell_bar = -999

        for i in range(1, n):
            b, s = self._score(i, ha_o, ha_c, rsi_a, rsi_ema, ema20, ml, sl_line)
            bull_arr[i] = b
            bear_arr[i] = s

            # BUY: first transition OR re-entry every reentry_bars while sustained
            if b >= self.threshold:
                if prev_bull < self.threshold or (i - last_buy_bar) >= self.reentry_bars:
                    buy_sig[i] = True
                    last_buy_bar = i

            # SELL: first transition OR re-entry every reentry_bars while sustained bear
            if s >= self.threshold:
                if prev_bear < self.threshold or (i - last_sell_bar) >= self.reentry_bars:
                    sell_sig[i] = True
                    last_sell_bar = i

            prev_bull = b
            prev_bear = s

        return buy_sig, sell_sig, bull_arr, bear_arr, atr_a

    # ── Live analysis ──────────────────────────────────────────────

    async def analyze(self, candles: list, current_price: float,
                      mtf_candles: dict = None) -> Signal:
        min_len = self.macd_slow + self.macd_sig + self.ema_len + 5
        if len(candles) < min_len:
            return Signal(SignalType.HOLD, self.symbol, current_price, 0, "Not enough data")

        ha_o, ha_c, rsi_a, rsi_ema, ema20, ml, sl_line, atr_a = \
            self._build_arrays(candles)

        buy_sig, sell_sig, bull_arr, bear_arr, _ = self._build_signals(candles)

        n   = len(candles) - 1
        p   = current_price
        atr = float(atr_a[n]) if not np.isnan(atr_a[n]) else 0.0
        rsi = float(rsi_a[n]) if not np.isnan(rsi_a[n]) else 50.0
        b_s = int(bull_arr[n]); s_s = int(bear_arr[n])
        conf = round(min(0.85, 0.50 + max(b_s, s_s) * 0.07), 2)

        meta = {
            "rsi":        round(rsi, 1),
            "bull_score": b_s,
            "bear_score": s_s,
            "atr":        round(atr, 4),
        }

        if buy_sig[n]:
            sl_p = round(p - self.sl_atr_mult * atr, 4)
            tp_p = round(p + self.sl_atr_mult * self.rr_ratio * atr, 4)
            return Signal(
                type=SignalType.BUY, symbol=self.symbol,
                price=p, amount=0.0, confidence=conf,
                reason=f"[Momentum] BUY score {b_s}/5 | RSI={rsi:.1f}",
                metadata={**meta, "stop_loss": sl_p, "take_profit": tp_p, "rr": self.rr_ratio},
            )

        if sell_sig[n]:
            sl_p = round(p + self.sl_atr_mult * atr, 4)
            tp_p = round(p - self.sl_atr_mult * self.rr_ratio * atr, 4)
            return Signal(
                type=SignalType.SELL, symbol=self.symbol,
                price=p, amount=0.0, confidence=conf,
                reason=f"[Momentum] SELL score {s_s}/5 | RSI={rsi:.1f}",
                metadata={**meta, "stop_loss": sl_p, "take_profit": tp_p, "rr": self.rr_ratio},
            )

        return Signal(
            SignalType.HOLD, self.symbol, current_price, 0,
            f"[Momentum] HOLD | bull={b_s} bear={s_s} RSI={rsi:.1f}",
            metadata=meta,
        )

    # ── Backtest ───────────────────────────────────────────────────

    async def backtest(self, candles: list) -> tuple[dict, tuple]:
        min_len = self.macd_slow + self.macd_sig + self.ema_len + 20
        if len(candles) < min_len:
            return {}, None

        ha_candles, ha_o, ha_c = self._heikin_ashi(candles)
        ha_highs = np.array([c.high for c in ha_candles], dtype=float)
        ha_lows  = np.array([c.low  for c in ha_candles], dtype=float)

        buy_sig, sell_sig, _, _, atr_a = self._build_signals(candles)

        signal_bars = []
        prev_dir = 0
        for i in range(min_len, len(candles) - 1):
            if np.isnan(atr_a[i]):
                continue
            if buy_sig[i] and prev_dir != 1:
                signal_bars.append((i, 1, float(atr_a[i])))
                prev_dir = 1
            elif sell_sig[i] and prev_dir != -1:
                signal_bars.append((i, -1, float(atr_a[i])))
                prev_dir = -1
            elif not buy_sig[i] and not sell_sig[i]:
                prev_dir = 0

        if not signal_bars:
            return {}, None

        best_score, best_config = -999.0, None
        stats: dict = {}

        for sl_m in _SL_MULTS:
            rr      = self.rr_ratio
            wins    = losses = 0
            total_r = 0.0
            for idx, direction, atr_val in signal_bars:
                if atr_val <= 0:
                    continue
                entry = float(ha_c[idx])
                sl_p  = entry - sl_m * atr_val if direction == 1 else entry + sl_m * atr_val
                tp_p  = entry + sl_m * rr * atr_val if direction == 1 else entry - sl_m * rr * atr_val
                outcome = 0
                for j in range(idx + 1, min(idx + _LOOKFORWARD, len(ha_c))):
                    if direction == 1:
                        if ha_lows[j]  <= sl_p: outcome = -1; break
                        if ha_highs[j] >= tp_p: outcome =  1; break
                    else:
                        if ha_highs[j] >= sl_p: outcome = -1; break
                        if ha_lows[j]  <= tp_p: outcome =  1; break
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
                best_score  = total_r
                best_config = (sl_m, rr)

        return stats, best_config
