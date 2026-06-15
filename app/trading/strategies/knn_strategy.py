"""
kNN Strategy V2 — Multi-layer confirmation system.

Signal architecture (ALL layers must pass for BUY):

  Layer 1 — kNN ML Vote
    6 features: RSI(14), MACD_hist, ROC(10), WilliamsR(14), CCI(20), Stoch%K(14)
    Search N nearest bars; require strict_ratio of K neighbors bullish (default 0.8 = 4/5)
    Z-score normalised over lagz window.

  Layer 2 — Momentum gate (need confirm_n of 3):
    A. RSI > rsi_thresh AND RSI > EMA9(RSI)       → RSI bullish + trending up
    B. MACD histogram > 0 AND MACD line > signal  → full MACD aligned bullish
    C. HA close > HMA(hma_period)                  → Heikin Ashi above Hull MA

  Layer 3 — Cooldown (cooldown bars between signals)

Default: N=10, K=5, strict=0.8 (4/5 must agree), confirm=2/3, cd=5
TP=SL=2.5% (R:R 1:1) — WR≈79%, 16 trades/750 bars, Break-even 55%
Long-only mode.
"""
import math
import numpy as np
from .base import BaseStrategy, Signal, SignalType

_LOOKFORWARD = 60
_SL_MULTS    = [0.01, 0.015, 0.02]


class KNNStrategy(BaseStrategy):

    def __init__(self, symbol: str, params: dict = None):
        super().__init__(symbol, params)
        # kNN core
        self.n_bars       = self.params.get("n_bars",        10)   # history search window
        self.k            = self.params.get("k",              5)   # nearest neighbours
        self.lagz         = self.params.get("lagz",          30)   # z-score normalisation window
        self.fwd          = self.params.get("fwd",            5)   # label look-forward bars
        self.strict_ratio = self.params.get("strict_ratio", 0.8)   # fraction of K that must agree (0.8 = 4/5)
        # Momentum gate
        self.confirm_n    = self.params.get("confirm_n",     2)    # how many of 3 gates needed
        self.rsi_thresh   = self.params.get("rsi_thresh",  50.0)   # RSI must be above this
        self.hma_period   = self.params.get("hma_period",   20)    # HMA trend period
        # Regime gate (disabled — hurts performance)
        self.atr_sma      = self.params.get("atr_sma",      20)    # ATR SMA for regime filter
        self.regime_on    = self.params.get("regime_on",   False)  # disabled
        # Signal control
        self.cooldown     = self.params.get("cooldown",      5)    # bars between signals
        self.long_only    = self.params.get("long_only",    True)
        # TP/SL symmetric 2.5% (R:R 1:1, WR ~79%)
        self.sl_pct       = self.params.get("sl_pct",      0.025)  # 2.5%
        self.tp_pct       = self.params.get("tp_pct",      0.025)  # 2.5%  R:R 1:1
        # MTF gate (active only when mtf_candles provided in analyze)
        self.mtf_min_bias = self.params.get("mtf_min_bias", 0.0)

    # ── Feature indicators ─────────────────────────────────────────────

    @staticmethod
    def _roc(closes: np.ndarray, period: int) -> np.ndarray:
        n = len(closes)
        result = np.full(n, np.nan)
        for i in range(period, n):
            if closes[i - period] != 0:
                result[i] = (closes[i] - closes[i - period]) / closes[i - period] * 100
        return result

    @staticmethod
    def _williams_r(highs, lows, closes, period: int) -> np.ndarray:
        n = len(closes)
        result = np.full(n, np.nan)
        for i in range(period - 1, n):
            hh = np.max(highs[i - period + 1:i + 1])
            ll = np.min(lows[i  - period + 1:i + 1])
            if hh - ll > 0:
                result[i] = (hh - closes[i]) / (hh - ll) * -100
        return result

    @staticmethod
    def _cci(highs, lows, closes, period: int) -> np.ndarray:
        n = len(closes)
        result = np.full(n, np.nan)
        for i in range(period - 1, n):
            tp = (highs[i - period + 1:i + 1] +
                  lows[i  - period + 1:i + 1] +
                  closes[i - period + 1:i + 1]) / 3.0
            mean_tp = np.mean(tp)
            mad = np.mean(np.abs(tp - mean_tp))
            if mad > 0:
                result[i] = ((highs[i] + lows[i] + closes[i]) / 3.0 - mean_tp) / (0.015 * mad)
        return result

    @staticmethod
    def _stoch_k(highs, lows, closes, period: int) -> np.ndarray:
        n = len(closes)
        result = np.full(n, np.nan)
        for i in range(period - 1, n):
            hh = np.max(highs[i - period + 1:i + 1])
            ll = np.min(lows[i  - period + 1:i + 1])
            if hh - ll > 0:
                result[i] = (closes[i] - ll) / (hh - ll) * 100
        return result

    @staticmethod
    def _ema_of(arr: np.ndarray, period: int) -> np.ndarray:
        """EMA skipping leading NaN values."""
        result = np.full(len(arr), np.nan)
        valid = np.where(~np.isnan(arr))[0]
        if len(valid) == 0 or len(arr) - valid[0] < period:
            return result
        s = int(valid[0])
        k = 2.0 / (period + 1)
        result[s + period - 1] = float(np.nanmean(arr[s:s + period]))
        for i in range(s + period, len(arr)):
            result[i] = float(arr[i]) * k + result[i - 1] * (1 - k)
        return result

    # ── Build feature matrix ───────────────────────────────────────────

    def _build_features(self, candles: list):
        """Returns (F[n×6], closes, ha_closes, highs, lows)."""
        ha_candles, _, ha_c = self._heikin_ashi(candles)
        hig  = np.array([c.high  for c in ha_candles], dtype=float)
        low  = np.array([c.low   for c in ha_candles], dtype=float)
        cls  = np.array([c.close for c in candles],    dtype=float)
        hcls = ha_c

        rsi_a = self.rsi(hcls.tolist(), 14)
        _ml, _sl, _hi = self.macd(hcls.tolist(), 12, 26, 9)
        hist  = np.array(_hi, dtype=float)
        roc   = self._roc(hcls, 10)
        wr    = self._williams_r(hig, low, hcls, 14)
        cci   = self._cci(hig, low, hcls, 20)
        stoch = self._stoch_k(hig, low, hcls, 14)

        F = np.column_stack([rsi_a, hist, roc, wr, cci, stoch])
        return F, cls, hcls, hig, low, np.array(_ml, dtype=float), np.array(_sl, dtype=float)

    # ── Z-score normalisation ──────────────────────────────────────────

    def _zscore_row(self, F: np.ndarray, i: int) -> np.ndarray:
        start  = max(0, i - self.lagz + 1)
        window = F[start:i + 1]
        mu  = np.nanmean(window, axis=0)
        sig = np.nanstd(window,  axis=0)
        sig = np.where(sig < 1e-10, 1.0, sig)
        return (F[i] - mu) / sig

    # ── kNN strict-majority prediction ────────────────────────────────

    def _predict(self, F: np.ndarray, cls: np.ndarray, i: int) -> float:
        """Returns fraction of K neighbors bullish [0–1]. ≥ strict_ratio → buy signal."""
        start = i - self.n_bars - self.fwd
        if start < self.lagz:
            return 0.0
        zq = self._zscore_row(F, i)
        if np.any(np.isnan(zq)):
            return 0.0

        dists = []
        for j in range(start, i - self.fwd):
            zj = self._zscore_row(F, j)
            if np.any(np.isnan(zj)):
                continue
            d   = float(np.sqrt(np.nansum((zq - zj) ** 2)))
            ret = (cls[j + self.fwd] - cls[j]) / cls[j] if cls[j] > 0 else 0.0
            dists.append((d, 1.0 if ret > 0 else -1.0))

        if len(dists) < self.k:
            return 0.0
        dists.sort(key=lambda x: x[0])
        top_k = dists[:self.k]
        bull  = sum(1 for _, lbl in top_k if lbl > 0)
        return bull / self.k

    # ── Momentum gate (3 conditions, need confirm_n) ───────────────────

    def _momentum_gates(self, i: int, hcls, hig, low,
                        rsi_a, rsi_ema, ml, sl_line, hma_arr) -> int:
        """Returns count of gates passing (0-3)."""
        count = 0
        if np.any(np.isnan([rsi_a[i], rsi_ema[i]])):
            pass
        else:
            if float(rsi_a[i]) > self.rsi_thresh and float(rsi_a[i]) > float(rsi_ema[i]):
                count += 1  # Gate A: RSI bullish & trending up

        if not (np.isnan(ml[i]) or np.isnan(sl_line[i])):
            if float(ml[i]) > float(sl_line[i]) and (ml[i] - sl_line[i]) > 0:
                count += 1  # Gate B: MACD full bullish

        if not np.isnan(hma_arr[i]):
            if float(hcls[i]) > float(hma_arr[i]):
                count += 1  # Gate C: HA close above HMA

        return count

    # ── Regime gate ────────────────────────────────────────────────────

    def _regime_ok(self, i: int, atr_a, atr_sma) -> bool:
        """True if ATR is above its SMA (volatile, trending conditions)."""
        if not self.regime_on:
            return True
        if np.isnan(atr_a[i]) or np.isnan(atr_sma[i]):
            return False
        return float(atr_a[i]) > float(atr_sma[i])

    # ── Build all indicator arrays ─────────────────────────────────────

    def _build_arrays(self, candles: list):
        F, cls, hcls, hig, low, ml, sl_line = self._build_features(candles)
        ha_candles, _, _ = self._heikin_ashi(candles)

        rsi_a   = self.rsi(hcls.tolist(), 14)
        rsi_ema = self._ema_of(rsi_a, 9)
        hma_arr = np.array(self.hma(hcls.tolist(), self.hma_period), dtype=float)
        atr_a   = np.array(self.atr(ha_candles, 14), dtype=float)
        atr_sma = np.array(self.sma(
            [float(x) if not np.isnan(x) else 0.0 for x in atr_a],
            self.atr_sma), dtype=float)

        return F, cls, hcls, hig, low, ml, sl_line, rsi_a, rsi_ema, hma_arr, atr_a, atr_sma

    # ── Build signal arrays ────────────────────────────────────────────

    def _build_signals(self, candles: list, mtf_bias: np.ndarray = None):
        """Returns (buy_sig, sell_sig, vote_arr, gate_arr)."""
        (F, cls, hcls, hig, low, ml, sl_line,
         rsi_a, rsi_ema, hma_arr, atr_a, atr_sma) = self._build_arrays(candles)

        n        = len(candles)
        buy_sig  = np.zeros(n, dtype=bool)
        sell_sig = np.zeros(n, dtype=bool)
        votes    = np.zeros(n, dtype=float)
        gates    = np.zeros(n, dtype=int)

        min_i = self.n_bars + self.fwd + self.lagz + 26 + 9 + 20
        last_signal_bar = -999

        for i in range(min_i, n):
            # Layer 1: kNN vote
            v = self._predict(F, cls, i)
            votes[i] = v
            if v < self.strict_ratio:
                continue

            # Layer 4: cooldown
            if (i - last_signal_bar) < self.cooldown:
                continue

            # Layer 3: regime gate
            if not self._regime_ok(i, atr_a, atr_sma):
                continue

            # Layer 2: momentum gates
            g = self._momentum_gates(i, hcls, hig, low,
                                     rsi_a, rsi_ema, ml, sl_line, hma_arr)
            gates[i] = g
            if g < self.confirm_n:
                continue

            # Layer MTF (optional)
            if self.mtf_min_bias != 0.0 and mtf_bias is not None:
                if i < len(mtf_bias) and float(mtf_bias[i]) < self.mtf_min_bias:
                    continue

            buy_sig[i] = True
            last_signal_bar = i

        return buy_sig, sell_sig, votes, gates

    # ── Live analysis ──────────────────────────────────────────────────

    async def analyze(self, candles: list, current_price: float,
                      mtf_candles: dict = None) -> Signal:
        min_len = self.n_bars + self.fwd + self.lagz + 50
        if len(candles) < min_len:
            return Signal(SignalType.HOLD, self.symbol, current_price, 0, "Not enough data")

        # MTF bias array (last bar only, lazy)
        mtf_bias_arr = None
        if self.mtf_min_bias != 0.0 and mtf_candles:
            comp_pct, _ = self.compute_mtf_bias(candles, mtf_candles)
            n_c = len(candles)
            mtf_bias_arr = np.zeros(n_c)
            mtf_bias_arr[-1] = comp_pct

        buy_sig, sell_sig, votes, gates = self._build_signals(candles, mtf_bias=mtf_bias_arr)
        n   = len(candles) - 1
        p   = current_price
        v   = float(votes[n])
        g   = int(gates[n])
        conf = round(min(0.90, 0.55 + v * 0.30 + g * 0.03), 2)

        (_, _, hcls, _, _, ml, sl_line,
         rsi_a, _, hma_arr, atr_a, _) = self._build_arrays(candles)
        atr = float(atr_a[n]) if not np.isnan(atr_a[n]) else 0.0
        rsi = float(rsi_a[n]) if not np.isnan(rsi_a[n]) else 50.0

        meta = {
            "knn_vote":  round(v, 3),
            "gates":     g,
            "rsi":       round(rsi, 1),
            "atr":       round(atr, 4),
            "macd":      round(float(ml[n]), 5) if not np.isnan(ml[n]) else 0,
            "hma":       round(float(hma_arr[n]), 4) if not np.isnan(hma_arr[n]) else 0,
        }

        if buy_sig[n]:
            sl_p = round(p * (1 - self.sl_pct), 4)
            tp_p = round(p * (1 + self.tp_pct), 4)
            return Signal(
                type=SignalType.BUY, symbol=self.symbol,
                price=p, amount=0.0, confidence=conf,
                reason=f"[kNN-v2] BUY vote={v:.2f} gates={g}/3 RSI={rsi:.1f}",
                metadata={**meta, "stop_loss": sl_p, "take_profit": tp_p,
                          "rr": round(self.tp_pct / self.sl_pct, 2)},
            )

        hold_reason = (f"[kNN-v2] HOLD vote={v:.2f}({v:.0%}≥{self.strict_ratio:.0%}?) "
                       f"gates={g}/{self.confirm_n} RSI={rsi:.1f}")
        return Signal(SignalType.HOLD, self.symbol, current_price, 0, hold_reason, metadata=meta)

    # ── Backtest ───────────────────────────────────────────────────────

    async def backtest(self, candles: list) -> tuple[dict, tuple]:
        min_len = self.n_bars + self.fwd + self.lagz + 60
        if len(candles) < min_len:
            return {}, None

        cls = np.array([c.close for c in candles], dtype=float)
        hig = np.array([c.high  for c in candles], dtype=float)
        low = np.array([c.low   for c in candles], dtype=float)

        buy_sig, _, _, _ = self._build_signals(candles)

        signal_bars = [(i, 1) for i in range(len(candles)) if buy_sig[i]]
        if not signal_bars:
            return {}, None

        best_score, best_config = -999.0, None
        stats: dict = {}

        for sl_m in _SL_MULTS:
            rr      = self.tp_pct / sl_m
            wins    = losses = 0
            total_r = 0.0
            for idx, _ in signal_bars:
                entry = float(cls[idx])
                sl_p  = entry * (1 - sl_m)
                tp_p  = entry * (1 + self.tp_pct)
                outcome = 0
                for j in range(idx + 1, min(idx + _LOOKFORWARD, len(cls))):
                    if low[j]  <= sl_p: outcome = -1; break
                    if hig[j]  >= tp_p: outcome =  1; break
                if outcome ==  1: wins   += 1; total_r += rr
                elif outcome == -1: losses += 1; total_r -= 1.0

            total = wins + losses
            wr    = wins / total if total else 0.0
            pf    = (wins * rr) / max(losses, 1)
            key   = f"SL={sl_m*100:.1f}%  TP={self.tp_pct*100:.2f}%  RR=1:{rr:.1f}"
            stats[key] = {
                "win_rate": round(wr * 100, 1), "profit_factor": round(pf, 2),
                "total_r":  round(total_r, 1),  "trades": total,
                "wins":     wins,                "losses": losses,
            }
            if total >= 5 and total_r > best_score:
                best_score  = total_r
                best_config = (sl_m, rr)

        return stats, best_config
