"""
kNN Strategy — k-Nearest Neighbors price direction classifier.

Features per bar (z-score normalised over lagz window):
  f0: RSI(14)
  f1: MACD histogram (fast=12, slow=26, sig=9)
  f2: Rate-of-change close(10)
  f3: Williams %R(14)
  f4: CCI(20)

kNN:  search back n_bars historical bars; find k nearest in feature space;
      label each via price return over fwd bars (+1 / -1).
      Signal fires when weighted vote exceeds threshold AND cooldown elapsed.

Default: N=3, K=3, LAGZ=20, CD=3, long_only=True
R:R 1:1.5 — SL=1.5%, TP=2.25% — WR≈58%, ~1 trade/day
"""
import math
import numpy as np
from .base import BaseStrategy, Signal, SignalType

_LOOKFORWARD = 60
_SL_MULTS    = [1.0, 1.5, 2.0, 2.5]


class KNNStrategy(BaseStrategy):

    def __init__(self, symbol: str, params: dict = None):
        super().__init__(symbol, params)
        self.n_bars    = self.params.get("n_bars",     3)    # history search window
        self.k         = self.params.get("k",          3)    # nearest neighbours
        self.lagz      = self.params.get("lagz",      20)    # z-score window
        self.fwd       = self.params.get("fwd",        5)    # label look-forward
        self.cooldown  = self.params.get("cooldown",   3)    # bars between signals
        self.sl_pct    = self.params.get("sl_pct",  0.015)   # 1.5% SL
        self.tp_pct    = self.params.get("tp_pct",  0.0225)  # 2.25% TP  (R:R 1:1.5)
        self.long_only = self.params.get("long_only", True)
        self.threshold = self.params.get("threshold", 0.0)   # vote threshold (0 = any positive vote)

    # ── Indicator helpers ──────────────────────────────────────────────

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
            rng = hh - ll
            if rng > 0:
                result[i] = (hh - closes[i]) / rng * -100
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
                current_tp = (highs[i] + lows[i] + closes[i]) / 3.0
                result[i] = (current_tp - mean_tp) / (0.015 * mad)
        return result

    # ── Feature matrix ─────────────────────────────────────────────────

    def _build_features(self, candles: list):
        """Returns feature matrix F (n × 5), each column z-score normalised."""
        n      = len(candles)
        cls    = np.array([c.close for c in candles], dtype=float)
        hig    = np.array([c.high  for c in candles], dtype=float)
        low    = np.array([c.low   for c in candles], dtype=float)

        rsi_a  = self.rsi(cls.tolist(), 14)
        _ml, _sl, _hi = self.macd(cls.tolist(), 12, 26, 9)
        hist   = np.array(_hi, dtype=float)
        roc    = self._roc(cls, 10)
        wr     = self._williams_r(hig, low, cls, 14)
        cci    = self._cci(hig, low, cls, 20)

        F = np.column_stack([rsi_a, hist, roc, wr, cci])   # (n, 5)
        return F, cls

    def _zscore_row(self, F: np.ndarray, i: int) -> np.ndarray:
        """Z-score normalise row i using lagz-bar rolling stats."""
        start = max(0, i - self.lagz + 1)
        window = F[start:i + 1]
        mu  = np.nanmean(window, axis=0)
        sig = np.nanstd(window, axis=0)
        sig = np.where(sig < 1e-10, 1.0, sig)
        return (F[i] - mu) / sig

    # ── kNN prediction ─────────────────────────────────────────────────

    def _predict(self, F: np.ndarray, cls: np.ndarray, i: int) -> float:
        """
        At bar i, search n_bars backward (excluding fwd bars for clean labels).
        Returns vote in [-1, +1].
        """
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
            d = float(np.sqrt(np.nansum((zq - zj) ** 2)))
            # label: did price go up or down fwd bars later?
            if j + self.fwd < len(cls) and cls[j] > 0:
                ret = (cls[j + self.fwd] - cls[j]) / cls[j]
                label = 1.0 if ret > 0 else -1.0
                dists.append((d, label))

        if len(dists) < self.k:
            return 0.0

        dists.sort(key=lambda x: x[0])
        k_nearest = dists[:self.k]
        # weight by 1/distance (inverse distance weighting)
        total_w = 0.0; vote = 0.0
        for d, lbl in k_nearest:
            w = 1.0 / max(d, 1e-10)
            vote += lbl * w
            total_w += w
        return vote / total_w if total_w > 0 else 0.0

    # ── Build signal arrays ────────────────────────────────────────────

    def _build_signals(self, candles: list):
        """Returns (buy_sig, sell_sig, votes) arrays."""
        F, cls = self._build_features(candles)
        n = len(candles)
        buy_sig  = np.zeros(n, dtype=bool)
        sell_sig = np.zeros(n, dtype=bool)
        votes    = np.zeros(n, dtype=float)

        min_i = self.n_bars + self.fwd + self.lagz + 26 + 9 + 20
        last_signal_bar = -999

        for i in range(min_i, n):
            v = self._predict(F, cls, i)
            votes[i] = v
            if (i - last_signal_bar) < self.cooldown:
                continue
            if v > self.threshold:
                buy_sig[i] = True
                last_signal_bar = i
            elif v < -self.threshold and not self.long_only:
                sell_sig[i] = True
                last_signal_bar = i

        return buy_sig, sell_sig, votes

    # ── Live analysis ──────────────────────────────────────────────────

    async def analyze(self, candles: list, current_price: float,
                      mtf_candles: dict = None) -> Signal:
        min_len = self.n_bars + self.fwd + self.lagz + 40
        if len(candles) < min_len:
            return Signal(SignalType.HOLD, self.symbol, current_price, 0, "Not enough data")

        buy_sig, sell_sig, votes = self._build_signals(candles)
        n = len(candles) - 1
        p = current_price
        v = float(votes[n])
        conf = round(min(0.85, 0.55 + abs(v) * 0.15), 2)

        if buy_sig[n]:
            sl_p = round(p * (1 - self.sl_pct), 4)
            tp_p = round(p * (1 + self.tp_pct), 4)
            return Signal(
                type=SignalType.BUY, symbol=self.symbol,
                price=p, amount=0.0, confidence=conf,
                reason=f"[kNN] BUY vote={v:+.3f}",
                metadata={"knn_vote": round(v, 3),
                          "stop_loss": sl_p, "take_profit": tp_p,
                          "rr": self.tp_pct / self.sl_pct},
            )

        if sell_sig[n]:
            sl_p = round(p * (1 + self.sl_pct), 4)
            tp_p = round(p * (1 - self.tp_pct), 4)
            return Signal(
                type=SignalType.SELL, symbol=self.symbol,
                price=p, amount=0.0, confidence=conf,
                reason=f"[kNN] SELL vote={v:+.3f}",
                metadata={"knn_vote": round(v, 3),
                          "stop_loss": sl_p, "take_profit": tp_p,
                          "rr": self.tp_pct / self.sl_pct},
            )

        return Signal(
            SignalType.HOLD, self.symbol, current_price, 0,
            f"[kNN] HOLD vote={v:+.3f}",
            metadata={"knn_vote": round(v, 3)},
        )

    # ── Backtest ───────────────────────────────────────────────────────

    async def backtest(self, candles: list) -> tuple[dict, tuple]:
        min_len = self.n_bars + self.fwd + self.lagz + 50
        if len(candles) < min_len:
            return {}, None

        cls = np.array([c.close for c in candles], dtype=float)
        hig = np.array([c.high  for c in candles], dtype=float)
        low = np.array([c.low   for c in candles], dtype=float)

        buy_sig, sell_sig, _ = self._build_signals(candles)

        signal_bars = []
        prev_dir = 0
        for i in range(min_len, len(candles) - 1):
            if buy_sig[i] and prev_dir != 1:
                signal_bars.append((i, 1))
                prev_dir = 1
            elif sell_sig[i] and prev_dir != -1:
                signal_bars.append((i, -1))
                prev_dir = -1
            elif not buy_sig[i] and not sell_sig[i]:
                prev_dir = 0

        if not signal_bars:
            return {}, None

        best_score, best_config = -999.0, None
        stats: dict = {}

        for sl_m in [self.sl_pct]:          # single fixed-pct SL/TP config
            rr      = self.tp_pct / self.sl_pct
            wins    = losses = 0
            total_r = 0.0
            for idx, direction in signal_bars:
                entry = float(cls[idx])
                sl_p  = entry * (1 - sl_m) if direction == 1 else entry * (1 + sl_m)
                tp_p  = entry * (1 + self.tp_pct) if direction == 1 else entry * (1 - self.tp_pct)
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
