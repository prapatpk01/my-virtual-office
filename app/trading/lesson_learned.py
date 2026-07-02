"""
Lesson-Learned Tracker — per-trade entry/exit forensics + rolling round review.

Every closed trade is recorded with its ENTRY context (which entry path fired,
confidence score/level, market regime, MTF bias, ADX) and its EXIT logic
(stop_loss / breakeven / take_profit2 / health_weak / guard detail).

Rounds: every `round_size` (default 5) closed trades = 1 round. After each
round a summary is sent to Telegram. Alerts fire when:
  - losses in the round >= round_loss_alert (default 3), or
  - consecutive losses reach consec_loss_alert (default 3, immediate).
Alerts include a rule-based analysis of the losing trades (dominant side /
symbol / regime / entry path / exit reason) so the failure pattern is visible
at a glance.

Adaptive caution mode (OPT-IN, LESSON_ADAPTIVE=true): after an alert, request
a temporary min_score bump (+LESSON_SCORE_BUMP, default 10) for the next
LESSON_CAUTION_TRADES closed trades (default 10). The bot applies/removes the
bump; entering/leaving caution is notified. Default OFF — it changes live
trading behaviour and has not been backtested.

State persists to a JSON file so restarts keep the rolling history.
No new Railway variables are required — every knob has a default.
"""
from __future__ import annotations

import json
import logging
import os
import time

logger = logging.getLogger("lessons")

TERMINAL_REASONS = {"stop_loss", "breakeven", "take_profit2", "health_weak"}

def _env_f(k: str, d: float) -> float:
    try: return float(os.environ.get(k, d))
    except (TypeError, ValueError): return d

def _env_i(k: str, d: int) -> int:
    try: return int(os.environ.get(k, d))
    except (TypeError, ValueError): return d

def _env_b(k: str, d: bool) -> bool:
    v = os.environ.get(k)
    return d if v is None else v.strip().lower() in ("1", "true", "yes", "on")


class LessonTracker:
    def __init__(self, telegram=None, state_path: str | None = None):
        self.telegram          = telegram
        self.state_path        = state_path or os.environ.get("LESSON_STATE_PATH", "lessons_state.json")
        self.round_size        = _env_i("LESSON_ROUND_SIZE", 5)
        self.round_loss_alert  = _env_i("LESSON_ROUND_LOSS_ALERT", 3)
        self.consec_loss_alert = _env_i("LESSON_CONSEC_ALERT", 3)
        self.adaptive          = _env_b("LESSON_ADAPTIVE", False)
        self.score_bump        = _env_f("LESSON_SCORE_BUMP", 10.0)
        self.caution_trades    = _env_i("LESSON_CAUTION_TRADES", 10)

        self._pending: dict[tuple, dict] = {}   # (symbol, slot) -> entry snapshot + partial pnl
        self._history: list[dict] = []          # closed trades, oldest→newest (tail kept)
        self._consec_losses = 0
        self._caution_left  = 0                 # closed trades remaining in caution mode
        self._load()

    # ── recording ──────────────────────────────────────────────────────────

    def record_open(self, symbol: str, slot: str, side: str, meta: dict) -> None:
        self._pending[(symbol, slot)] = {
            "symbol": symbol, "slot": slot, "side": side,
            "opened_at": int(time.time()),
            "entry_path": meta.get("entry_path", "pullback"),
            "score":      float(meta.get("confidence_score", 0) or 0),
            "level":      meta.get("confidence_level", ""),
            "regime":     meta.get("regime", ""),
            "bias":       float(meta.get("mtf_bias", 0) or 0),
            "adx":        float(meta.get("adx15", 0) or 0),
            "one_r":      float(meta.get("one_r", 0) or 0),
            "partial_pnl": 0.0,
        }

    def record_partial(self, symbol: str, slot: str, pnl: float) -> None:
        p = self._pending.get((symbol, slot))
        if p is not None:
            p["partial_pnl"] += pnl

    def record_close(self, symbol: str, slot: str, reason: str,
                     pnl: float, price: float, detail: str = "") -> None:
        p = self._pending.pop((symbol, slot), None) or {
            "symbol": symbol, "slot": slot, "side": "?", "opened_at": 0,
            "entry_path": "?", "score": 0.0, "level": "?", "regime": "?",
            "bias": 0.0, "adx": 0.0, "one_r": 0.0, "partial_pnl": 0.0,
        }
        total = pnl + p["partial_pnl"]
        trade = {
            "ts": int(time.time()), "symbol": symbol, "slot": slot,
            "side": p["side"], "entry_path": p["entry_path"],
            "score": p["score"], "level": p["level"], "regime": p["regime"],
            "bias": p["bias"], "adx": p["adx"],
            "exit": reason, "detail": detail,
            "pnl": round(total, 4), "win": total > 0,
        }
        self._history.append(trade)
        self._history = self._history[-200:]
        self._consec_losses = 0 if trade["win"] else self._consec_losses + 1
        if self._caution_left > 0:
            self._caution_left -= 1
        self._save()
        self._check_alerts(trade)

    # ── adaptive caution (bot polls score_bump_active after each close) ─────

    def score_bump_active(self) -> float:
        return self.score_bump if (self.adaptive and self._caution_left > 0) else 0.0

    # ── alerts & analysis ───────────────────────────────────────────────────

    def _check_alerts(self, last_trade: dict) -> None:
        alerts: list[str] = []

        if self._consec_losses == self.consec_loss_alert:
            recent_losses = [t for t in self._history[-self._consec_losses:]]
            alerts.append(
                f"🔻 *แพ้ {self._consec_losses} ครั้งติด*\n" + self._analyze(recent_losses))

        n = len(self._history)
        if n and n % self.round_size == 0:
            rnd    = self._history[-self.round_size:]
            losses = [t for t in rnd if not t["win"]]
            pnl    = sum(t["pnl"] for t in rnd)
            header = (f"📒 *Round #{n // self.round_size}* (เทรดที่ {n - self.round_size + 1}-{n})\n"
                      f"W/L: {self.round_size - len(losses)}/{len(losses)}  PnL: `{pnl:+.2f}$`")
            lines  = [f"{'✅' if t['win'] else '❌'} {t['symbol'].split('/')[0]} "
                      f"{t['side']} {t['entry_path']} sc{t['score']:.0f} → {t['exit']} `{t['pnl']:+.2f}$`"
                      for t in rnd]
            body = header + "\n" + "\n".join(lines)
            if len(losses) >= self.round_loss_alert:
                body += f"\n\n⚠️ *แพ้ {len(losses)}/{self.round_size} ใน round นี้*\n" + self._analyze(losses)
                alerts.append(body)
            else:
                self._notify(body)   # normal round summary, no alert flavour

        if alerts:
            for a in alerts:
                self._notify(a)
            if self.adaptive and self._caution_left <= 0:
                self._caution_left = self.caution_trades
                self._notify(
                    f"🛡 *Caution mode ON* — ยก min_score +{self.score_bump:.0f} "
                    f"อีก {self.caution_trades} เทรด (LESSON_ADAPTIVE)")

    def _analyze(self, losses: list[dict]) -> str:
        """Rule-based pattern summary of losing trades."""
        if not losses:
            return "—"
        def top(key):
            counts: dict[str, int] = {}
            for t in losses:
                v = str(t.get(key, "?"))
                counts[v] = counts.get(v, 0) + 1
            k, c = max(counts.items(), key=lambda kv: kv[1])
            return k, c
        side, sc   = top("side")
        sym, syc   = top("symbol")
        exi, exc   = top("exit")
        pat, pac   = top("entry_path")
        reg, rgc   = top("regime")
        n = len(losses)
        avg_score  = sum(t["score"] for t in losses) / n
        lines = ["🔍 *วิเคราะห์ไม้ที่แพ้:*"]
        for t in losses:
            d = f" ({t['detail']})" if t.get("detail") else ""
            lines.append(f"• {t['symbol'].split('/')[0]} {t['side']} เข้า:{t['entry_path']} "
                         f"sc{t['score']:.0f} {t['regime'] or '?'} → ออก:{t['exit']}{d} `{t['pnl']:+.2f}$`")
        pattern = []
        if sc  == n: pattern.append(f"แพ้ฝั่ง {side} ทั้งหมด")
        if syc == n: pattern.append(f"แพ้ที่ {sym.split('/')[0]} ทั้งหมด")
        if exc >= max(2, n - 1): pattern.append(f"ออกทาง {exi} เป็นหลัก ({exc}/{n})")
        if pac == n and pat != "?": pattern.append(f"เข้าแบบ {pat} ทั้งหมด")
        if rgc == n and reg not in ("", "?"): pattern.append(f"regime {reg} ทั้งหมด")
        pattern.append(f"score เฉลี่ยไม้แพ้ {avg_score:.0f}")
        lines.append("📌 " + " | ".join(pattern))
        return "\n".join(lines)

    # ── plumbing ────────────────────────────────────────────────────────────

    def _notify(self, text: str) -> None:
        logger.info("[LESSON] %s", text.replace("\n", " | "))
        if self.telegram:
            try:
                self.telegram.notify(text)
            except Exception as e:
                logger.warning("[LESSON] telegram notify failed: %s", e)

    def _save(self) -> None:
        try:
            with open(self.state_path, "w") as f:
                json.dump({"history": self._history[-200:],
                           "consec": self._consec_losses,
                           "caution_left": self._caution_left}, f)
        except Exception as e:
            logger.warning("[LESSON] state save failed: %s", e)

    def _load(self) -> None:
        try:
            if os.path.exists(self.state_path):
                with open(self.state_path) as f:
                    st = json.load(f)
                self._history       = st.get("history", [])
                self._consec_losses = int(st.get("consec", 0))
                self._caution_left  = int(st.get("caution_left", 0))
                logger.info("[LESSON] restored %d trades, consec=%d",
                            len(self._history), self._consec_losses)
        except Exception as e:
            logger.warning("[LESSON] state load failed (starting fresh): %s", e)
