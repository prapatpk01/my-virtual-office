"""
MACD + EMA Voting Strategy — 2/3 confirmation + ATR-based SL/TP.

Three voters:
  A) price vs HMA15
  B) EMA9  vs SMA21  (cross detection)
  C) MACD  vs Signal (cross detection)

Signal when 2/3 agree. SL/TP computed from ATR.
Backtest finds the optimal SL multiplier + R:R ratio automatically.
MTF gate (D): 1H + 4H HMA15 must agree with A's direction.
"""
import logging
import numpy as np
from .base import BaseStrategy, Signal, SignalType

logger = logging.getLogger("macd_ema_strategy")

# Parameter search space for backtest
_SL_MULTS  = [1.0, 1.5, 2.0, 2.5]
_RR_RATIOS = [1.2]
_ATR_PERIOD = 14
_LOOKFORWARD = 60  # max candles to look for SL/TP hit


class MACDEMAStrategy(BaseStrategy):

    def __init__(self, symbol: str, params: dict = None):
        super().__init__(symbol, params)
        self.hma_period  = self.params.get("hma_period",    15)
        self.ema_fast    = self.params.get("ema_fast",       9)
        self.sma_slow    = self.params.get("sma_slow",      21)
        self.macd_fast   = self.params.get("macd_fast",     12)
        self.macd_slow   = self.params.get("macd_slow",     26)
        self.macd_sig    = self.params.get("macd_signal",    9)
        self.adx_threshold = self.params.get("adx_threshold", 15)
        # SL/TP — updated by backtest at startup
        self.sl_atr_mult = self.params.get("sl_atr_mult",  1.5)
        self.rr_ratio    = self.params.get("rr_ratio",     1.2)

    # ------------------------------------------------------------------
    # Live signal
    # ------------------------------------------------------------------

    async def analyze(self, candles: list, current_price: float,
                      mtf_candles: dict = None) -> Signal:
        min_len = self.macd_slow + self.macd_sig + self.hma_period + 5
        if len(candles) < min_len:
            return Signal(SignalType.HOLD, self.symbol, current_price, 0, "Not enough data")

        closes = [c.close for c in candles]
        open_price = float(candles[-1].open)   # use candle open as primary HMA15 gate

        hma15      = self.hma(closes, self.hma_period)
        ema9       = self.ema(closes, self.ema_fast)
        sma21      = self.sma(closes, self.sma_slow)
        macd_line, signal_line, _ = self.macd(
            closes, self.macd_fast, self.macd_slow, self.macd_sig
        )
        atr_arr        = self.atr(candles, _ATR_PERIOD)
        adx_arr, _, _  = self.adx(candles, 14)

        p     = current_price
        hma   = float(hma15[-1])
        e9_c  = float(ema9[-1]);        e9_p  = float(ema9[-2])
        s21_c = float(sma21[-1]);       s21_p = float(sma21[-2])
        ml_c  = float(macd_line[-1]);   ml_p  = float(macd_line[-2])
        sl_c  = float(signal_line[-1]); sl_p  = float(signal_line[-2])
        current_atr = float(atr_arr[-1])
        current_adx = float(adx_arr[-1]) if not np.isnan(adx_arr[-1]) else 0.0

        if any(np.isnan(v) for v in [hma, e9_c, s21_c, ml_c, sl_c, current_atr]):
            return Signal(SignalType.HOLD, self.symbol, current_price, 0, "Indicator NaN")

        # Voter A uses candle OPEN price (core gate — must pass to get any signal)
        vote_a = 1 if open_price > hma else (-1 if open_price < hma else 0)
        vote_b = 1 if e9_c > s21_c else (-1 if e9_c < s21_c else 0)
        vote_c = 1 if ml_c > sl_c  else (-1 if ml_c < sl_c  else 0)

        # A is CORE: signal only when A passes + at least one of B/C agrees
        b_cross_up   = e9_p <= s21_p and e9_c > s21_c
        b_cross_down = e9_p >= s21_p and e9_c < s21_c
        c_cross_up   = ml_p <= sl_p  and ml_c > sl_c
        c_cross_down = ml_p >= sl_p  and ml_c < sl_c

        buy_votes  = sum(1 for v in [vote_b, vote_c] if v ==  1)
        sell_votes = sum(1 for v in [vote_b, vote_c] if v == -1)

        # ── ADX gate — filter sideways markets ────────────────────────
        if current_adx < self.adx_threshold:
            return Signal(
                SignalType.HOLD, self.symbol, current_price, 0,
                f"[ADX BLOCK] ADX={current_adx:.1f} < {self.adx_threshold}",
                metadata={"hma15": hma, "open": open_price, "adx": round(current_adx, 1)},
            )

        # ── MTF bias (D) — final gate, must agree with A ──────────────
        mtf_pass, mtf_label = _check_mtf(mtf_candles, vote_a, self.hma_period)

        if vote_a == 1 and buy_votes >= 1:
            if not mtf_pass:
                return Signal(
                    SignalType.HOLD, self.symbol, current_price, 0,
                    f"[MTF BLOCK] A=BUY B/C ok | {mtf_label}",
                    metadata={"hma15": hma, "open": open_price, "mtf": mtf_label},
                )
            total_buy = 1 + buy_votes
            reason, conf = _build_reason(
                "buy", total_buy, vote_a, vote_b, vote_c,
                b_cross_up, b_cross_down, c_cross_up, c_cross_down,
            )
            sl_price = round(p - self.sl_atr_mult * current_atr, 4)
            tp_price = round(p + self.sl_atr_mult * self.rr_ratio * current_atr, 4)
            return Signal(
                type=SignalType.BUY, symbol=self.symbol,
                price=p, amount=0.0, confidence=conf,
                reason=f"{reason} | MTF✓ {mtf_label}",
                metadata={
                    "hma15": hma, "open": open_price,
                    "ema9": e9_c, "sma21": s21_c,
                    "macd": ml_c, "macd_signal": sl_c,
                    "atr": round(current_atr, 4),
                    "adx": round(current_adx, 1),
                    "stop_loss": sl_price,
                    "take_profit": tp_price,
                    "rr": self.rr_ratio,
                    "votes": total_buy,
                    "mtf": mtf_label,
                },
            )

        if vote_a == -1 and sell_votes >= 1:
            if not mtf_pass:
                return Signal(
                    SignalType.HOLD, self.symbol, current_price, 0,
                    f"[MTF BLOCK] A=SELL B/C ok | {mtf_label}",
                    metadata={"hma15": hma, "open": open_price, "mtf": mtf_label},
                )
            total_sell = 1 + sell_votes
            reason, conf = _build_reason(
                "sell", total_sell, vote_a, vote_b, vote_c,
                b_cross_up, b_cross_down, c_cross_up, c_cross_down,
            )
            sl_price = round(p + self.sl_atr_mult * current_atr, 4)
            tp_price = round(p - self.sl_atr_mult * self.rr_ratio * current_atr, 4)
            return Signal(
                type=SignalType.SELL, symbol=self.symbol,
                price=p, amount=0.0, confidence=conf,
                reason=f"{reason} | MTF✓ {mtf_label}",
                metadata={
                    "hma15": hma, "open": open_price,
                    "ema9": e9_c, "sma21": s21_c,
                    "macd": ml_c, "macd_signal": sl_c,
                    "atr": round(current_atr, 4),
                    "adx": round(current_adx, 1),
                    "stop_loss": sl_price,
                    "take_profit": tp_price,
                    "rr": self.rr_ratio,
                    "votes": total_sell,
                    "mtf": mtf_label,
                },
            )

        # HOLD
        a_reason = (f"open>HMA15" if vote_a == 1
                    else f"open<HMA15" if vote_a == -1
                    else "open≈HMA15")
        votes_str = (
            f"A={'B' if vote_a==1 else 'S' if vote_a==-1 else '-'} "
            f"B={'B' if vote_b==1 else 'S' if vote_b==-1 else '-'} "
            f"C={'B' if vote_c==1 else 'S' if vote_c==-1 else '-'}"
        )
        return Signal(
            SignalType.HOLD, self.symbol, current_price, 0,
            f"[HOLD] {votes_str} | {a_reason}",
            metadata={
                "hma15": hma, "open": open_price, "ema9": e9_c, "sma21": s21_c,
                "macd": ml_c, "macd_signal": sl_c,
                "buy_votes": buy_votes, "sell_votes": sell_votes,
            },
        )

    # ------------------------------------------------------------------
    # Backtest — find optimal SL multiplier + R:R ratio
    # ------------------------------------------------------------------

    async def backtest(self, candles: list) -> dict:
        """
        Simulate 2/3 voting signals on historical candles.
        Tests all combinations of SL multipliers and R:R ratios.
        Returns stats dict and the best (sl_mult, rr) tuple.
        """
        min_len = self.macd_slow + self.macd_sig + self.hma_period + _ATR_PERIOD + 5
        if len(candles) < min_len + 20:
            return {}, None

        closes = [c.close for c in candles]
        highs  = [c.high  for c in candles]
        lows   = [c.low   for c in candles]

        hma_arr  = self.hma(closes, self.hma_period)
        ema9_arr = self.ema(closes, self.ema_fast)
        sma21_arr= self.sma(closes, self.sma_slow)
        ml_arr, sl_arr, _ = self.macd(closes, self.macd_fast, self.macd_slow, self.macd_sig)
        atr_arr  = self.atr(candles, _ATR_PERIOD)

        # Collect all signal bars (direction + atr)
        signal_bars: list[tuple[int, int, float]] = []  # (idx, direction +1/-1, atr)
        prev_dir = 0  # avoid consecutive same-direction entries

        for i in range(min_len, len(candles) - 1):
            if any(np.isnan(v) for v in [
                hma_arr[i], ema9_arr[i], sma21_arr[i], ml_arr[i], sl_arr[i], atr_arr[i]
            ]):
                continue
            va, vb, vc = _votes(
                closes[i], hma_arr[i],
                ema9_arr[i], sma21_arr[i],
                ml_arr[i], sl_arr[i],
            )
            bv = sum(1 for v in [va, vb, vc] if v ==  1)
            sv = sum(1 for v in [va, vb, vc] if v == -1)

            if bv >= 2 and prev_dir != 1:
                signal_bars.append((i, 1, float(atr_arr[i])))
                prev_dir = 1
            elif sv >= 2 and prev_dir != -1:
                signal_bars.append((i, -1, float(atr_arr[i])))
                prev_dir = -1
            else:
                prev_dir = 0

        if not signal_bars:
            return {}, None

        # Test each parameter combination
        best_score = -999.0
        best_config = None
        stats: dict[str, dict] = {}

        for sl_m in _SL_MULTS:
            for rr in _RR_RATIOS:
                wins = losses = 0
                total_r = 0.0

                for idx, direction, atr_val in signal_bars:
                    if atr_val <= 0:
                        continue
                    entry = closes[idx]
                    if direction == 1:
                        sl_p = entry - sl_m * atr_val
                        tp_p = entry + sl_m * rr * atr_val
                    else:
                        sl_p = entry + sl_m * atr_val
                        tp_p = entry - sl_m * rr * atr_val

                    outcome = 0
                    for j in range(idx + 1, min(idx + _LOOKFORWARD, len(candles))):
                        if direction == 1:
                            if lows[j] <= sl_p:
                                outcome = -1; break
                            if highs[j] >= tp_p:
                                outcome = 1; break
                        else:
                            if highs[j] >= sl_p:
                                outcome = -1; break
                            if lows[j] <= tp_p:
                                outcome = 1; break

                    if outcome == 1:
                        wins += 1; total_r += rr
                    elif outcome == -1:
                        losses += 1; total_r -= 1.0

                total = wins + losses
                wr = wins / total if total else 0.0
                pf = (wins * rr) / max(losses, 1)
                key = f"SL={sl_m}xATR  RR=1:{rr}"
                stats[key] = {
                    "win_rate": round(wr * 100, 1),
                    "profit_factor": round(pf, 2),
                    "total_r": round(total_r, 1),
                    "trades": total,
                    "wins": wins,
                    "losses": losses,
                }

                if total >= 5 and total_r > best_score:
                    best_score = total_r
                    best_config = (sl_m, rr)

        return stats, best_config


# ------------------------------------------------------------------
# Module-level helpers (keep analyze() readable)
# ------------------------------------------------------------------

def _votes(price, hma, e9, s21, ml, sl):
    va = 1 if price > hma  else (-1 if price < hma  else 0)
    vb = 1 if e9    > s21  else (-1 if e9    < s21  else 0)
    vc = 1 if ml    > sl   else (-1 if ml    < sl   else 0)
    return va, vb, vc


def _build_reason(direction, votes, va, vb, vc, b_up, b_dn, c_up, c_dn):
    is_buy = direction == "buy"
    target = 1 if is_buy else -1
    tags = []
    if va == target:
        tags.append(f"HMA15={'above' if is_buy else 'below'}")
    if vb == target:
        cross = " ✚cross" if (b_up if is_buy else b_dn) else ""
        tags.append(f"EMA9{'>' if is_buy else '<'}SMA21{cross}")
    if vc == target:
        cross = " ✚cross" if (c_up if is_buy else c_dn) else ""
        tags.append(f"MACD{'>' if is_buy else '<'}Sig{cross}")

    conf = 0.85 if votes == 3 else 0.65
    has_cross = (is_buy and (b_up or c_up)) or (not is_buy and (b_dn or c_dn))
    if has_cross:
        conf = min(conf + 0.10, 0.95)
    return f"[{votes}/3] {' + '.join(tags)}", conf


def _check_mtf(mtf_candles: dict, direction: int, hma_period: int) -> tuple:
    """Check that 1H and 4H both agree with direction (+1=bull, -1=bear).
    Returns (passed: bool, label: str).
    Passes through (True) when no MTF data is available.
    """
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
    label = " ".join(
        f"{tf}={'↑' if v == 1 else '↓'}" for tf, v in sorted(results.items())
    )
    return agree == len(results), label
