"""ClaudeAnalyzer — AI confirmation gate for trade signals using Claude API."""
import json
import logging
import math
import os
from typing import Optional

logger = logging.getLogger("claude_analyzer")

_DEFAULT_MODEL = "claude-haiku-4-5"


class ClaudeAnalyzer:
    """
    Asks Claude to confirm or reject a BUY signal before the bot executes.

    Set ANTHROPIC_API_KEY in environment to enable.
    Set CLAUDE_MODEL to override the model (default: claude-haiku-4-5).
    Set USE_AI_GATE=true in environment to activate the gate in the bot.

    Returns (confirm: bool, reason: str).
    If ANTHROPIC_API_KEY is not set, always returns (True, "AI gate skipped").
    """

    def __init__(self, model: Optional[str] = None):
        self.model = model or os.getenv("CLAUDE_MODEL", _DEFAULT_MODEL)
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError:
                logger.error("anthropic package not installed — run: pip install anthropic")
                return None
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            if not api_key:
                return None
            self._client = anthropic.AsyncAnthropic(api_key=api_key)
        return self._client

    async def confirm_buy(
        self,
        symbol: str,
        price: float,
        strategy_name: str,
        signal_reason: str,
        indicators: dict,
        recent_candles: list,
        mtf_bias: str = "unknown",
        volume_ratio: float = 1.0,
        recent_outcomes: list = None,
    ) -> tuple[bool, str]:
        """
        Ask Claude to confirm or reject a BUY signal using 5-dimension scoring.

        indicators keys:
            rsi, hma_period, price_above_hma, ema5, sma9,
            ema5_above_sma9, atr, sl_price, tp_price, rr

        recent_candles: list of dicts {open, high, low, close, volume}, newest last.
        mtf_bias: "bullish" | "bearish" | "neutral" | "unknown"
        volume_ratio: current bar volume / 20-bar average
        recent_outcomes: list of {"type": "tp"|"sl"|"timeout", "pnl_pct": float}

        Returns (confirm: bool, reason: str).
        CONFIRM if avg score >= 3.5 AND no dimension scores 1.
        """
        client = self._get_client()
        if client is None:
            return True, "AI gate skipped — ANTHROPIC_API_KEY not set"

        rsi = indicators.get("rsi")
        hma_period = indicators.get("hma_period", "?")
        above_hma = indicators.get("price_above_hma")
        ema5 = indicators.get("ema5")
        sma9 = indicators.get("sma9")
        ema_cross = indicators.get("ema5_above_sma9")
        atr = indicators.get("atr")
        sl_price = indicators.get("sl_price")
        tp_price = indicators.get("tp_price")
        rr = indicators.get("rr", 1.2)

        def _fmt(v, decimals=2):
            if v is None or (isinstance(v, float) and math.isnan(v)):
                return "N/A"
            return f"{v:.{decimals}f}"

        sl_pct = abs(price - sl_price) / price * 100 if sl_price else 0
        tp_pct = abs(tp_price - price) / price * 100 if tp_price else 0

        candle_lines = "\n".join(
            f"  {i+1:2d}. O={c['open']:.2f} H={c['high']:.2f} "
            f"L={c['low']:.2f} C={c['close']:.2f} V={c['volume']:.0f}"
            for i, c in enumerate(recent_candles[-20:])
        )

        # Recent outcomes context
        if recent_outcomes:
            wins = sum(1 for o in recent_outcomes if o.get("type") == "tp")
            losses = sum(1 for o in recent_outcomes if o.get("type") == "sl")
            timeouts = sum(1 for o in recent_outcomes if o.get("type") == "timeout")
            avg_pnl = sum(o.get("pnl_pct", 0) for o in recent_outcomes) / len(recent_outcomes)
            outcomes_line = (f"Last {len(recent_outcomes)} trades: "
                             f"{wins}TP / {losses}SL / {timeouts}timeout  "
                             f"avg_pnl={avg_pnl:+.2f}%")
        else:
            outcomes_line = "No recent trade history yet."

        # Volume context
        if volume_ratio >= 1.5:
            vol_context = f"{volume_ratio:.1f}x avg (HIGH — confirms breakout)"
        elif volume_ratio >= 1.1:
            vol_context = f"{volume_ratio:.1f}x avg (above average)"
        elif volume_ratio >= 0.7:
            vol_context = f"{volume_ratio:.1f}x avg (normal)"
        else:
            vol_context = f"{volume_ratio:.1f}x avg (LOW — weak signal)"

        prompt = f"""You are a crypto trading risk analyst. A strategy has triggered a BUY signal on {symbol}.
Score each dimension 1-5, then decide whether to CONFIRM or REJECT.

## Signal Details
- Symbol: {symbol}
- Current price: ${price:,.2f}
- Strategy: {strategy_name}
- Signal reason: {signal_reason}
- Stop Loss: ${_fmt(sl_price)} ({sl_pct:.2f}% below entry)
- Take Profit: ${_fmt(tp_price)} ({tp_pct:.2f}% above entry)
- Risk/Reward: 1:{rr}

## Market Context
- Higher-timeframe trend bias: {mtf_bias.upper()}
- Volume (current vs 20-bar avg): {vol_context}
- Recent strategy performance: {outcomes_line}

## Technical Indicators (1H timeframe)
- RSI(14): {_fmt(rsi, 1)}
- Price above HMA({hma_period}): {"YES" if above_hma else "NO" if above_hma is not None else "N/A"}
- EMA(5): {_fmt(ema5)}
- SMA(9): {_fmt(sma9)}
- EMA5 > SMA9 (short-term momentum): {"YES" if ema_cross else "NO" if ema_cross is not None else "N/A"}
- ATR(14): {_fmt(atr, 4)}

## Recent 1H Candles (oldest → newest, last 20)
{candle_lines}

## Scoring Guide
5=Strongly favorable  4=Favorable  3=Neutral  2=Unfavorable  1=Strongly unfavorable

## Your Task
Respond ONLY with valid JSON (no markdown, no prose):
{{
  "confirm": true or false,
  "confidence": 0.0 to 1.0,
  "scores": {{
    "trend": 1-5,
    "momentum": 1-5,
    "timing": 1-5,
    "price_action": 1-5,
    "risk_reward": 1-5
  }},
  "reason": "explanation under 120 chars"
}}

CONFIRM if: avg score >= 3.5 AND no single dimension = 1.
trend: Is price in clear uptrend? HTF={mtf_bias.upper()} matters most — BEARISH HTF = score 1-2.
momentum: RSI 45-70, EMA5>SMA9, fresh cross = high score; RSI>75 overbought = score 1-2.
timing: Not chasing (price not extended from HMA), volume supports entry?
price_action: Higher lows, bullish candle bodies, no large bearish wicks?
risk_reward: SL/TP placement reasonable for current ATR and market conditions?"""

        try:
            response = await client.messages.create(
                model=self.model,
                max_tokens=400,
                messages=[{"role": "user", "content": prompt}],
            )

            # Extract text block (skip thinking blocks)
            text = ""
            for block in response.content:
                if hasattr(block, "type") and block.type == "text":
                    text = block.text.strip()
                    break

            if not text:
                logger.warning("[Claude] Empty text response for %s %s", strategy_name, symbol)
                return True, "AI returned empty response — defaulting to confirm"

            # Strip markdown code fences if Claude wrapped the JSON
            if "```" in text:
                for part in text.split("```"):
                    part = part.strip().lstrip("json").strip()
                    if part.startswith("{"):
                        text = part
                        break

            parsed = json.loads(text)
            confirm = bool(parsed.get("confirm", True))
            confidence = float(parsed.get("confidence", 0.5))
            reason = str(parsed.get("reason", "AI analysis"))

            # Programmatic score validation — override Claude's confirm if scores fail
            scores = parsed.get("scores", {})
            if scores and len(scores) == 5:
                avg_score = sum(scores.values()) / 5
                min_score = min(scores.values())
                if min_score <= 1 or avg_score < 3.5:
                    confirm = False
                    reason = f"score override avg={avg_score:.1f} min={min_score} | {reason}"

            logger.info(
                "[Claude] %s %s → confirm=%s conf=%.2f | %s",
                strategy_name, symbol, confirm, confidence, reason,
            )
            return confirm, f"[AI {confidence:.0%}] {reason}"

        except json.JSONDecodeError as e:
            logger.warning("[Claude] JSON parse error: %s | raw: %.200s", e, text)
            return True, "AI parse error — defaulting to confirm"
        except Exception as e:
            logger.error("[Claude] Error for %s %s: %s", strategy_name, symbol, e)
            return True, f"AI error ({type(e).__name__}) — defaulting to confirm"
