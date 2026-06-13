"""AI Signal strategy — uses Claude to analyze price action and news."""
import os
import json
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
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
        return self._client

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
        curr_rsi = float(rsi_vals[-1]) if not __import__("numpy").isnan(rsi_vals[-1]) else None

        prompt = f"""You are a professional quantitative trader. Analyze the following {self.lookback} recent OHLCV candles for {self.symbol} and provide a trading recommendation.

Current price: {current_price}
RSI(14): {curr_rsi:.1f if curr_rsi else 'N/A'}

Recent candles (oldest to newest):
{json.dumps(candle_data, indent=2)}

Respond ONLY with a JSON object in this exact format:
{{
  "signal": "buy" | "sell" | "hold",
  "confidence": 0.0-1.0,
  "reason": "brief explanation (max 100 chars)"
}}"""

        try:
            client = self._get_client()
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=150,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()
            # Extract JSON even if wrapped in markdown
            if "```" in text:
                text = text.split("```")[1].lstrip("json").strip()
            parsed = json.loads(text)
            sig_type = SignalType(parsed.get("signal", "hold").lower())
            confidence = float(parsed.get("confidence", 0.5))
            reason = parsed.get("reason", "AI analysis")

            return Signal(
                type=sig_type,
                symbol=self.symbol,
                price=current_price,
                amount=self.position_pct if sig_type != SignalType.HOLD else 0,
                reason=f"[AI] {reason}",
                confidence=confidence,
                metadata={"rsi": curr_rsi, "ai_raw": parsed},
            )
        except Exception as e:
            return Signal(
                SignalType.HOLD, self.symbol, current_price, 0,
                f"AI error: {str(e)[:60]}",
            )
