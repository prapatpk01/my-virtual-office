"""
ClaudeAnalyzer v2 — Gate mode + Chief mode.

Gate mode  (USE_AI_GATE=true):
    Claude confirms/rejects a strategy BUY signal before execution.
    Model: claude-haiku-4-5  (fast, cheap, ~$0.001/call)

Chief mode (USE_AI_CHIEF=true):
    Claude runs independently every tick as the main decision-maker.
    Strategies provide "opinions" as context — Claude can BUY even if no
    strategy triggered, or HOLD even if all strategies say BUY.
    Model: claude-opus-4-8 + adaptive thinking + streaming (~$0.05-0.15/call)
"""
import json
import logging
import math
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger("claude_analyzer")

_GATE_MODEL  = os.getenv("CLAUDE_MODEL",       "claude-haiku-4-5")
_CHIEF_MODEL = os.getenv("CLAUDE_CHIEF_MODEL", "claude-opus-4-8")


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class ChiefDecision:
    action: str          # "BUY" | "HOLD"
    confidence: float    # 0-100
    reasoning: str
    confluence: list = field(default_factory=list)
    divergence: list = field(default_factory=list)
    risk_flags: list = field(default_factory=list)


# ── Indicator helpers (numpy only, no pandas) ─────────────────────────────────

def _s(v) -> Optional[float]:
    """Safe float round — returns None for NaN/None."""
    try:
        f = float(v)
        return None if math.isnan(f) else round(f, 4)
    except (TypeError, ValueError):
        return None


def _extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model response, tolerating ```json fences.

    Uses brace-matching rather than lstrip('json') (which strips stray j/s/o/n chars).
    """
    if not text:
        raise json.JSONDecodeError("empty", "", 0)
    t = text.strip()
    # Remove code fences if present
    if "```" in t:
        for seg in t.split("```"):
            seg = seg.strip()
            if seg.startswith("json"):
                seg = seg[4:].strip()
            if seg.startswith("{"):
                t = seg
                break
    # Brace-match from first { to its matching }
    start = t.find("{")
    if start == -1:
        raise json.JSONDecodeError("no object", t, 0)
    depth = 0
    for i in range(start, len(t)):
        if t[i] == "{":
            depth += 1
        elif t[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(t[start:i + 1])
    return json.loads(t[start:])


def _extract_tf_metrics(candles: list) -> dict:
    """RSI, MACD hist, EMA20/50, volume ratio for one TF (uses confirmed closed bar)."""
    from .strategies.base import BaseStrategy
    if len(candles) < 60:
        return {}
    closes  = np.array([c.close  for c in candles], dtype=float)
    volumes = np.array([c.volume for c in candles], dtype=float)

    rsi14    = BaseStrategy.rsi(closes.tolist(), 14)
    _, _, mh = BaseStrategy.macd(closes.tolist(), 12, 26, 9)
    ema20    = BaseStrategy.ema(closes.tolist(), 20)
    ema50    = BaseStrategy.ema(closes.tolist(), 50)
    vol_ma   = BaseStrategy.sma(volumes.tolist(), 20)

    i = -2  # second-to-last = confirmed closed candle
    return {
        "close":          _s(closes[i]),
        "rsi":            _s(rsi14[i]),
        "rsi_prev":       _s(rsi14[i - 1]),
        "macd_hist":      _s(mh[i]),
        "macd_hist_prev": _s(mh[i - 1]),
        "ema_fast":       _s(ema20[i]),
        "ema_slow":       _s(ema50[i]),
        "volume_ratio":   _s(float(volumes[i]) / max(float(vol_ma[i] or 1e-9), 1e-9)),
    }


def _momentum_score(m: dict) -> Optional[float]:
    """Compute -1.0 (strong down) to +1.0 (strong up) per TF. Same formula as v2.1."""
    parts = []
    rsi = m.get("rsi")
    if rsi is not None:
        parts.append((rsi - 50) / 50)

    macd = m.get("macd_hist")
    macd_p = m.get("macd_hist_prev")
    if macd is not None:
        c = 1.0 if macd > 0 else (-1.0 if macd < 0 else 0.0)
        if macd_p is not None:
            c *= 1.2 if abs(macd) > abs(macd_p) else 0.8
        parts.append(max(-1.0, min(1.0, c)))

    ef = m.get("ema_fast")
    es = m.get("ema_slow")
    if ef is not None and es is not None:
        parts.append(1.0 if ef > es else (-1.0 if ef < es else 0.0))

    return round(sum(parts) / len(parts), 3) if parts else None


def _rsi_divergence(candles: list, lookback: int = 30, swing_lb: int = 3) -> dict:
    """Detect bullish/bearish RSI divergence via swing-point comparison."""
    from .strategies.base import BaseStrategy
    if len(candles) < lookback + 10:
        return {"bullish": False, "bearish": False}

    sub    = candles[-lookback:]
    highs  = np.array([c.high  for c in sub], dtype=float)
    lows   = np.array([c.low   for c in sub], dtype=float)
    closes = np.array([c.close for c in sub], dtype=float)
    rsi    = BaseStrategy.rsi(closes.tolist(), 14)
    n      = len(lows)

    sl_idx = [i for i in range(swing_lb, n - swing_lb)
              if lows[i] == lows[max(0, i - swing_lb):i + swing_lb + 1].min()]
    sh_idx = [i for i in range(swing_lb, n - swing_lb)
              if highs[i] == highs[max(0, i - swing_lb):i + swing_lb + 1].max()]

    result = {"bullish": False, "bearish": False}
    if len(sl_idx) >= 2:
        i1, i2 = sl_idx[-2], sl_idx[-1]
        if (lows[i2] < lows[i1] and
                not np.isnan(rsi[i2]) and not np.isnan(rsi[i1]) and
                rsi[i2] > rsi[i1]):
            result["bullish"] = True
    if len(sh_idx) >= 2:
        i1, i2 = sh_idx[-2], sh_idx[-1]
        if (highs[i2] > highs[i1] and
                not np.isnan(rsi[i2]) and not np.isnan(rsi[i1]) and
                rsi[i2] < rsi[i1]):
            result["bearish"] = True
    return result


def _swing_levels(candles: list, lookback: int = 5, window: int = 100) -> dict:
    """Find nearest swing high/low levels as support/resistance."""
    sub = candles[-min(window, len(candles)):]
    n   = len(sub)
    highs   = np.array([c.high  for c in sub], dtype=float)
    lows    = np.array([c.low   for c in sub], dtype=float)
    current = float(sub[-1].close)

    s_highs, s_lows = [], []
    for i in range(lookback, n - lookback):
        if highs[i] >= np.max(highs[max(0, i - lookback):i + lookback + 1]) - 1e-8:
            s_highs.append(round(float(highs[i]), 2))
        if lows[i] <= np.min(lows[max(0, i - lookback):i + lookback + 1]) + 1e-8:
            s_lows.append(round(float(lows[i]), 2))

    return {
        "resistance": sorted(set(h for h in s_highs if h > current))[:3],
        "support":    sorted(set(l for l in s_lows  if l < current), reverse=True)[:3],
    }


def build_snapshot(symbol: str, candles_by_tf: dict, strategy_opinions: list) -> dict:
    """
    Build multi-TF snapshot for the Chief prompt.

    candles_by_tf: {"1h": [...], "4h": [...], "1d": [...]}
    strategy_opinions: [{"name": str, "signal": "BUY"|"HOLD", "reason": str}]
    """
    tf_metrics = {}
    momentum   = {}
    for tf, candles in candles_by_tf.items():
        if candles and len(candles) >= 60:
            m = _extract_tf_metrics(candles)
            tf_metrics[tf] = m
            momentum[tf]   = _momentum_score(m)
        else:
            tf_metrics[tf] = None
            momentum[tf]   = None

    valid = [s for s in momentum.values() if s is not None]
    alignment = {
        "bullish_tfs": sum(1 for s in valid if s >  0.15),
        "bearish_tfs": sum(1 for s in valid if s < -0.15),
        "neutral_tfs": sum(1 for s in valid if -0.15 <= s <= 0.15),
        "total":       len(valid),
    }

    candles_1h = candles_by_tf.get("1h", [])
    divergence = (_rsi_divergence(candles_1h) if len(candles_1h) >= 40
                  else {"bullish": False, "bearish": False})
    levels     = (_swing_levels(candles_1h)   if len(candles_1h) >= 20
                  else {"resistance": [], "support": []})

    return {
        "symbol":            symbol,
        "timeframe_metrics": tf_metrics,
        "momentum_scores":   momentum,
        "alignment":         alignment,
        "rsi_divergence_1h": divergence,
        "swing_levels_1h":   levels,
        "strategy_opinions": strategy_opinions,
    }


# ── News ──────────────────────────────────────────────────────────────────────

async def fetch_news(symbol: str, max_items: int = 6) -> list[str]:
    """CryptoPanic headlines. Returns [] if CRYPTOPANIC_API_KEY not set."""
    api_key = os.environ.get("CRYPTOPANIC_API_KEY", "")
    if not api_key:
        return []
    currency = symbol.split("/")[0]
    try:
        import aiohttp
        async with aiohttp.ClientSession() as sess:
            async with sess.get(
                "https://cryptopanic.com/api/v1/posts/",
                params={"auth_token": api_key, "currencies": currency, "public": "true"},
                timeout=aiohttp.ClientTimeout(total=8),
            ) as r:
                data = await r.json()
        snippets = []
        for post in (data.get("results") or [])[:max_items]:
            title = post.get("title", "")
            v     = post.get("votes", {})
            tag   = ("bullish lean" if v.get("positive", 0) > v.get("negative", 0) else
                     "bearish lean" if v.get("negative", 0) > v.get("positive", 0) else
                     "neutral")
            snippets.append(f"{title} [sentiment: {tag}]")
        return snippets
    except Exception as e:
        logger.debug("News fetch failed: %s", e)
        return []


# ── System prompts ────────────────────────────────────────────────────────────

_CHIEF_SYSTEM = """คุณเป็น Chief Trading Analyst สำหรับระบบ swing trade BTC/USDT ใน live market

หน้าที่: วิเคราะห์ข้อมูล multi-timeframe แล้วตัดสินใจ BUY หรือ HOLD โดยอิสระ
ทีมงาน 3 strategy (SwingReversal, CPKRegime, HybridSwing) ให้ความเห็นเป็น input เท่านั้น
คุณสามารถ BUY ได้แม้ไม่มี strategy ใดเห็นด้วย แต่ต้องมีเหตุผลแข็งแกร่งมาก

ข้อมูลที่ได้รับ:
- momentum_scores: -1.0 (ขาลงแรง) ถึง +1.0 (ขาขึ้นแรง) ต่อ TF — คำนวณจากโค้ด ไม่ใช่การเดา
- alignment: สรุปกี่ TF เป็นขาขึ้น/ขาลง
- rsi_divergence_1h: ตรวจจาก swing points จริง bullish=True หมายถึงสัญญาณกลับตัวขึ้น
- swing_levels_1h: แนวรับ/ต้านจาก price history จริง
- strategy_opinions: ความเห็น 3 strategy

กฎสำคัญ:
1. TF ยาว (4h, 1d) สำคัญกว่า TF สั้น (1h)
2. ถ้า 4h และ 1d momentum < -0.3 ทั้งคู่ → HOLD ยกเว้นมีเหตุผลพิเศษมาก
3. ถ้าราคาใกล้ resistance (< 1% ห่าง) และไม่มี bullish divergence → เสี่ยง
4. ถ้าข้อมูล TF ใดเป็น null → ลด confidence และบอกตรงๆ
5. ห้ามอ้างมั่นใจสูงโดยไม่มีหลักฐานใน data ที่ให้มา

ตอบกลับเป็น JSON เท่านั้น ห้ามมีข้อความอื่น:
{
  "action": "BUY" หรือ "HOLD",
  "confidence": ตัวเลข 0-100,
  "reasoning": "เหตุผล 2-3 ประโยคภาษาไทย",
  "confluence": ["1h: ...", "4h: ..."],
  "divergence": ["1d: ..."],
  "risk_flags": ["ความเสี่ยง..."]
}"""

_GATE_SYSTEM = """You are a crypto trading risk analyst. A strategy triggered a BUY signal.
Score each dimension 1-5 then decide CONFIRM or REJECT.
CONFIRM if: avg score >= 3.5 AND no dimension = 1.
trend: uptrend? HTF bearish = 1-2. momentum: RSI 45-70, fresh MACD = high; RSI>75 = 1-2.
timing: not chasing, volume supports? price_action: bullish candles, no big bearish wicks?
risk_reward: SL/TP reasonable for ATR?
Reply ONLY valid JSON (no markdown):
{"confirm":true/false,"confidence":0.0-1.0,"scores":{"trend":1-5,"momentum":1-5,"timing":1-5,"price_action":1-5,"risk_reward":1-5},"reason":"<120 chars"}"""


# ── Main class ────────────────────────────────────────────────────────────────

class ClaudeAnalyzer:
    """
    Dual-mode Claude integration.
    Instantiate with chief_mode=True for USE_AI_CHIEF behaviour.
    """

    def __init__(self, model: Optional[str] = None, chief_mode: bool = False):
        self.chief_mode = chief_mode
        self.model      = model or (_CHIEF_MODEL if chief_mode else _GATE_MODEL)
        self._client    = None

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError:
                logger.error("anthropic not installed — pip install anthropic")
                return None
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            if not api_key:
                return None
            self._client = anthropic.AsyncAnthropic(api_key=api_key)
        return self._client

    # ── Chief mode ────────────────────────────────────────────────────────

    async def chief_decide(self, snapshot: dict, news: list[str]) -> ChiefDecision:
        """
        Chief mode: Claude independently decides BUY or HOLD.
        Uses claude-opus-4-8 + adaptive thinking + streaming.
        """
        client = self._get_client()
        if client is None:
            return ChiefDecision("HOLD", 0, "Chief: ANTHROPIC_API_KEY not set")

        news_text = ("\n".join(f"- {n}" for n in news)
                     if news else "(ไม่มีข่าว — วิเคราะห์จาก technical เท่านั้น)")

        user_msg = (
            f"ข้อมูลตลาด {snapshot['symbol']}:\n"
            f"```json\n{json.dumps(snapshot, ensure_ascii=False, indent=2)}\n```\n\n"
            f"ข่าว/sentiment:\n{news_text}\n\n"
            f"ตัดสินใจ: BUY หรือ HOLD?"
        )

        try:
            async with client.messages.stream(
                model=self.model,
                max_tokens=8000,   # adaptive thinking shares this budget — keep generous
                thinking={"type": "adaptive"},
                system=_CHIEF_SYSTEM,
                messages=[{"role": "user", "content": user_msg}],
            ) as stream:
                msg = await stream.get_final_message()

            if getattr(msg, "stop_reason", None) == "max_tokens":
                logger.warning("[Chief] response truncated (max_tokens) — HOLD")
                return ChiefDecision("HOLD", 0, "Chief: response truncated — HOLD")

            text = ""
            for block in msg.content:
                if hasattr(block, "type") and block.type == "text":
                    text = block.text.strip()
                    break

            if not text:
                return ChiefDecision("HOLD", 0, "Chief: empty response — HOLD")

            parsed = _extract_json(text)
            action = "BUY" if str(parsed.get("action", "HOLD")).upper() == "BUY" else "HOLD"
            conf   = max(0.0, min(100.0, float(parsed.get("confidence", 0))))
            reason = str(parsed.get("reasoning", ""))

            decision = ChiefDecision(
                action=action, confidence=conf, reasoning=reason,
                confluence=list(parsed.get("confluence", [])),
                divergence=list(parsed.get("divergence", [])),
                risk_flags=list(parsed.get("risk_flags", [])),
            )
            logger.info("[Chief] %s → %s conf=%.0f | %s",
                        snapshot["symbol"], action, conf, reason[:80])
            return decision

        except json.JSONDecodeError as e:
            logger.warning("[Chief] JSON parse error: %s", e)
            return ChiefDecision("HOLD", 0, "Chief: parse error — HOLD")
        except Exception as e:
            logger.error("[Chief] API error: %s", e)
            return ChiefDecision("HOLD", 0, f"Chief: {type(e).__name__} — HOLD")

    # ── Gate mode ─────────────────────────────────────────────────────────

    async def confirm_buy(
        self,
        symbol: str, price: float, strategy_name: str,
        signal_reason: str, indicators: dict, recent_candles: list,
        mtf_bias: str = "unknown", volume_ratio: float = 1.0,
        recent_outcomes: list = None,
    ) -> tuple[bool, str]:
        """Gate mode: confirm or reject a strategy BUY signal. Uses Haiku.

        Note: this gate fails OPEN — on missing key, parse error, or API failure it
        returns confirm=True so AI downtime never blocks an otherwise-valid strategy
        signal. (Chief mode fails CLOSED — it returns HOLD on any failure.)
        """
        client = self._get_client()
        if client is None:
            logger.warning("[Gate] ANTHROPIC_API_KEY not set — confirming without AI review")
            return True, "AI gate skipped — ANTHROPIC_API_KEY not set"

        sl_p = indicators.get("sl_price")
        tp_p = indicators.get("tp_price")
        rsi  = indicators.get("rsi")
        atr  = indicators.get("atr")
        rr   = indicators.get("rr", 1.2)

        def _fmt(v, d=2):
            if v is None or (isinstance(v, float) and math.isnan(v)):
                return "N/A"
            return f"{v:.{d}f}"

        sl_pct = abs(price - sl_p) / price * 100 if sl_p else 0
        tp_pct = abs(tp_p - price) / price * 100 if tp_p else 0

        candle_lines = "\n".join(
            f"  {i+1:2d}. O={c['open']:.2f} H={c['high']:.2f} "
            f"L={c['low']:.2f} C={c['close']:.2f} V={c['volume']:.0f}"
            for i, c in enumerate(recent_candles[-20:])
        )

        if recent_outcomes:
            wins   = sum(1 for o in recent_outcomes if o.get("type") == "tp")
            losses = sum(1 for o in recent_outcomes if o.get("type") == "sl")
            avg_pnl = sum(o.get("pnl_pct", 0) for o in recent_outcomes) / len(recent_outcomes)
            outcomes_line = f"Last {len(recent_outcomes)} trades: {wins}TP/{losses}SL avg={avg_pnl:+.2f}%"
        else:
            outcomes_line = "No recent trade history."

        vol_ctx = f"{volume_ratio:.1f}x avg ({'HIGH' if volume_ratio>=1.5 else 'above avg' if volume_ratio>=1.1 else 'normal' if volume_ratio>=0.7 else 'LOW'})"

        prompt = (
            f"BUY {symbol} @ ${price:,.2f} from {strategy_name}\n"
            f"SL=${_fmt(sl_p)} ({sl_pct:.2f}% below)  TP=${_fmt(tp_p)} ({tp_pct:.2f}% above)  R:R=1:{rr}\n"
            f"HTF: {mtf_bias.upper()} | Vol: {vol_ctx} | RSI: {_fmt(rsi,1)} | ATR: {_fmt(atr,4)}\n"
            f"Recent: {outcomes_line}\nSignal: {signal_reason}\n\nLast 20 candles:\n{candle_lines}"
        )

        try:
            response = await client.messages.create(
                model=self.model,
                max_tokens=400,
                system=_GATE_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            text = ""
            for block in response.content:
                if hasattr(block, "type") and block.type == "text":
                    text = block.text.strip()
                    break

            parsed  = _extract_json(text)
            confirm = bool(parsed.get("confirm", True))
            conf    = float(parsed.get("confidence", 0.5))
            reason  = str(parsed.get("reason", ""))

            scores = parsed.get("scores", {})
            if scores and len(scores) == 5:
                # Coerce to float — a model returning "4" as a string must not
                # crash sum()/min() and fall through to the fail-open default.
                try:
                    vals = [float(v) for v in scores.values()]
                    avg_score = sum(vals) / 5
                    min_score = min(vals)
                    if min_score <= 1 or avg_score < 3.5:
                        confirm = False
                        reason  = f"score override avg={avg_score:.1f} min={min_score:.0f} | {reason}"
                except (TypeError, ValueError):
                    logger.warning("[Gate] non-numeric scores: %s", scores)

            logger.info("[Gate] %s %s → confirm=%s conf=%.2f | %s",
                        strategy_name, symbol, confirm, conf, reason)
            return confirm, f"[AI {conf:.0%}] {reason}"

        except json.JSONDecodeError as e:
            logger.warning("[Gate] JSON parse error: %s", e)
            return True, "AI parse error — defaulting confirm"
        except Exception as e:
            logger.error("[Gate] Error: %s", e)
            return True, f"AI error — defaulting confirm"
