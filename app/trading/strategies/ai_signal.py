"""AI Signal strategy — uses Claude to analyze price action and news."""
import os
import json
from typing import Any
import numpy as np
from .base import BaseStrategy, Signal, SignalType


class AISignalStrategy(BaseStrategy):
    """
    Uses Claude claude-sonnet-4-6 to analyze recent OHLCV data and produce
    a trading signal with reasoning.
    Requires ANTHROPIC_API_KEY in environment.
    """

    def __init__(self, symbol: str, params: dict = None):
        super().__init__(symbol, params)
        self.lookback = self.params.get("lookback", 20)
        self.position_pct = self.params.get("position_pct", 0.05)
        self.buy_threshold = float(self.params.get("buy_threshold", 2.0))
        self.sell_threshold = float(self.params.get("sell_threshold", -2.0))
        self._client = None
        self.history_context = ""  # optional: plain-text digest from LearningAnalysis

    def set_history_context(self, text: str):
        """
        Inject a short digest of historical win-rate/recommendations
        (see trading.learning_analysis.LearningAnalysis.context_for_ai) so the
        AI reasons using real track record, not just the current candle window.
        """
        self.history_context = text or ""

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
        return self._client

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _clip(value: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, value))

    @classmethod
    def _normalize_component_scores(cls, analysis: dict) -> dict:
        keys = ("trend", "momentum", "volatility", "volume", "risk")
        scores = {}
        for k in keys:
            scores[k] = cls._clip(cls._safe_float(analysis.get(f"{k}_score", 0.0), 0.0), -2.0, 2.0)
        return scores

    def _derive_signal_from_score(self, total_score: float) -> SignalType:
        if total_score >= self.buy_threshold:
            return SignalType.BUY
        if total_score <= self.sell_threshold:
            return SignalType.SELL
        return SignalType.HOLD

    async def analyze(self, candles: list, current_price: float,
                      mtf_candles: dict = None) -> Signal:
        recent = candles[-self.lookback:]
        if len(recent) < 5:
            return Signal(SignalType.HOLD, self.symbol, current_price, 0, "Not enough data for AI")

        candle_data = [
            {"open": c.open, "high": c.high, "low": c.low, "close": c.close, "volume": c.volume}
            for c in recent
        ]

        # Compute basic indicators to give Claude more context
        closes = [c.close for c in recent]
        rsi_vals = self.rsi(closes)
        curr_rsi = float(rsi_vals[-1]) if not np.isnan(rsi_vals[-1]) else None
        rsi_str = f"{curr_rsi:.1f}" if curr_rsi is not None else "N/A"
        ema_fast = self.ema(closes, 9)
        ema_slow = self.ema(closes, 21)
        macd_line, macd_signal, macd_hist = self.macd(closes)
        atr_vals = self.atr(recent, 14)

        curr_ema_fast = float(ema_fast[-1]) if not np.isnan(ema_fast[-1]) else None
        curr_ema_slow = float(ema_slow[-1]) if not np.isnan(ema_slow[-1]) else None
        curr_macd = float(macd_line[-1]) if not np.isnan(macd_line[-1]) else None
        curr_macd_signal = float(macd_signal[-1]) if not np.isnan(macd_signal[-1]) else None
        curr_macd_hist = float(macd_hist[-1]) if not np.isnan(macd_hist[-1]) else None
        curr_atr = float(atr_vals[-1]) if not np.isnan(atr_vals[-1]) else None

        prev_close = closes[-2] if len(closes) >= 2 else closes[-1]
        price_change_pct = ((closes[-1] - prev_close) / prev_close * 100.0) if prev_close != 0 else 0.0
        returns = np.diff(np.array(closes, dtype=float)) / np.array(closes[:-1], dtype=float)
        volatility_pct = float(np.std(returns) * 100.0) if len(returns) > 0 else 0.0
        avg_vol_5 = float(np.mean([c.volume for c in recent[-5:]])) if len(recent) >= 5 else float(recent[-1].volume)
        vol_ratio = (recent[-1].volume / avg_vol_5) if avg_vol_5 else 1.0

        history_section = (
            f"\n{self.history_context}\n" if self.history_context else ""
        )

        prompt = f"""You are a professional quantitative trader.
Analyze the following {self.lookback} recent OHLCV candles for {self.symbol} using systematic thinking.
Evaluate in strict order: trend -> momentum -> volatility -> volume -> risk.
Each component score must be in range -2.0 to +2.0.
Then calculate total_score = trend + momentum + volatility + volume + risk.
Decision rule:
- total_score >= {self.buy_threshold:.1f}: buy
- total_score <= {self.sell_threshold:.1f}: sell
- otherwise: hold

Current price: {current_price}
RSI(14): {rsi_str}
EMA(9): {curr_ema_fast if curr_ema_fast is not None else "N/A"}
EMA(21): {curr_ema_slow if curr_ema_slow is not None else "N/A"}
MACD: {curr_macd if curr_macd is not None else "N/A"}
MACD signal: {curr_macd_signal if curr_macd_signal is not None else "N/A"}
MACD histogram: {curr_macd_hist if curr_macd_hist is not None else "N/A"}
ATR(14): {curr_atr if curr_atr is not None else "N/A"}
Price change (last candle): {price_change_pct:.2f}%
Return volatility: {volatility_pct:.2f}%
Volume ratio (last/avg5): {vol_ratio:.2f}
{history_section}
Recent candles (oldest to newest):
{json.dumps(candle_data, indent=2)}

Respond ONLY with a JSON object in this exact format:
{{
  "signal": "buy" | "sell" | "hold",
  "confidence": 0.0-1.0,
  "reason": "brief explanation (max 100 chars)",
  "systematic_thinking": {{
    "trend_score": -2.0 to 2.0,
    "momentum_score": -2.0 to 2.0,
    "volatility_score": -2.0 to 2.0,
    "volume_score": -2.0 to 2.0,
    "risk_score": -2.0 to 2.0,
    "total_score": -10.0 to 10.0,
    "decision_basis": "short summary"
  }}
}}"""

        try:
            client = self._get_client()
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=250,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()
            # Extract JSON even if wrapped in markdown
            if "```" in text:
                text = text.split("```")[1].lstrip("json").strip()
            parsed = json.loads(text)
            llm_signal_raw = str(parsed.get("signal", "hold")).lower()
            try:
                llm_signal = SignalType(llm_signal_raw)
            except ValueError:
                llm_signal = SignalType.HOLD
            confidence = self._clip(self._safe_float(parsed.get("confidence", 0.5), 0.5), 0.0, 1.0)
            reason = str(parsed.get("reason", "AI analysis"))[:100]

            analysis = parsed.get("systematic_thinking", {}) or {}
            scores = self._normalize_component_scores(analysis)
            computed_total = sum(scores.values())
            sig_type = self._derive_signal_from_score(computed_total)
            llm_total = self._safe_float(analysis.get("total_score", computed_total), computed_total)
            decision_basis = str(analysis.get("decision_basis", ""))[:180]

            return Signal(
                type=sig_type,
                symbol=self.symbol,
                price=current_price,
                amount=self.position_pct if sig_type != SignalType.HOLD else 0,
                reason=f"[AI] {reason}",
                confidence=confidence,
                metadata={
                    "rsi": curr_rsi,
                    "systematic_thinking": {
                        **scores,
                        "total_score": computed_total,
                        "llm_total_score": llm_total,
                        "llm_signal": llm_signal.value,
                        "decision_basis": decision_basis,
                    },
                    "ai_raw": parsed,
                },
            )
        except Exception as e:
            return Signal(
                SignalType.HOLD, self.symbol, current_price, 0,
                f"AI error: {str(e)[:60]}",
            )
