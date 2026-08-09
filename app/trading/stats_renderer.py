"""Rich Telegram /stats renderer for paper/internal mode.

OKX live mode keeps using adaptive_stats.render_adaptive_stats(). This module
only replaces TelegramNotifier._render_stats for the internal/paper fallback so
/stats remains useful before live OKX position history exists.
"""
from __future__ import annotations

import time
from collections import defaultdict

SEP = "—" * 16


def _fmt_price(v) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):,.4f}"
    except (TypeError, ValueError):
        return "—"


def _short_symbol(symbol: str) -> str:
    return str(symbol or "?").split("/")[0]


def _short_strategy(name: str) -> str:
    text = str(name or "unknown")
    # Keep the useful strategy family while removing the repeated symbol suffix.
    if "(" in text:
        text = text.split("(", 1)[0]
    return text.replace(":L", "").replace(":S", "")


def _age_str(ts_ms) -> str:
    try:
        age = max(0, int(time.time() - float(ts_ms) / 1000.0))
    except (TypeError, ValueError):
        return "—"
    if age < 3600:
        return f"{max(1, age // 60)}m ago"
    if age < 86400:
        return f"{age / 3600:.1f}h ago"
    return f"{age / 86400:.1f}d ago"


def render_internal_stats(notifier, s: dict) -> str:
    """Render a detailed paper/internal /stats report.

    Data comes from SignalState.summary() plus the bot state callback already
    attached to TelegramNotifier. No exchange history is invented in paper mode.
    """
    state = notifier.get_state_fn() if getattr(notifier, "get_state_fn", None) else {}
    positions = list(state.get("positions", []) or [])

    bal = s.get("balance")
    balance_line = f"💰 Balance: `${bal:,.2f}`" if bal is not None else "💰 Balance: `—`"
    mode = "📄 PAPER" if state.get("paper", True) else "⚠️ INTERNAL FALLBACK"

    trades = int(s.get("trades", 0) or 0)
    wins = int(s.get("wins", 0) or 0)
    losses = int(s.get("losses", 0) or 0)
    wr = float(s.get("win_rate", 0) or 0)
    net = float(s.get("total_pnl_usd", 0) or 0)
    ret = float(s.get("return_pct", 0) or 0)
    pf = s.get("profit_factor")
    total_r = float(s.get("total_r", 0) or 0)
    streak = int(s.get("streak", 0) or 0)
    pending = int(s.get("pending", 0) or 0)
    total_signals = int(s.get("total_signals", 0) or 0)
    sig_day = float(s.get("signals_per_day", 0) or 0)
    paper_balance = s.get("paper_balance")
    start_balance = s.get("start_balance")

    net_sign = "+" if net >= 0 else "-"
    ret_sign = "+" if ret >= 0 else ""
    streak_str = f"W{streak}" if streak > 0 else f"L{abs(streak)}" if streak < 0 else "—"
    pf_str = "—" if pf is None else ("∞" if float(pf) >= 999 else f"{float(pf):.2f}")

    lines = [
        f"📊 *Trading Stats — {mode}*",
        balance_line,
        f"📌 Open positions: `{len(positions)}`",
        "",
        SEP,
        "PERFORMANCE",
        SEP,
        f"Closed trades : `{trades}`  (`{wins}W / {losses}L`)",
        f"Win rate      : `{wr:.1f}%`",
        f"Net P&L       : `{net_sign}${abs(net):,.2f}`",
        f"Return        : `{ret_sign}{ret:.2f}%`",
        f"Total R       : `{total_r:+.2f}R`",
        f"Profit factor : `{pf_str}`",
        f"Streak        : `{streak_str}`",
    ]

    if paper_balance is not None and start_balance is not None:
        lines.append(
            f"Journal model : `${float(start_balance):,.2f}` → `${float(paper_balance):,.2f}`"
        )

    lines += [
        "",
        SEP,
        "ACTIVITY",
        SEP,
        f"Signals fired : `{total_signals}`",
        f"Signals/day   : `{sig_day:.1f}`",
        f"Pending       : `{pending}`",
    ]

    if positions:
        lines += ["", SEP, "OPEN POSITIONS", SEP]
        for p in positions:
            side = str(p.get("side", "?")).lower()
            arrow = "🟢" if side == "long" else "🔴" if side == "short" else "⚪️"
            sym = _short_symbol(p.get("symbol"))
            strat = _short_strategy(p.get("strategy"))
            entry = float(p.get("entry", 0) or 0)
            amount = float(p.get("amount", 0) or 0)
            sl = p.get("stop_loss")
            tp = p.get("take_profit")
            notional = entry * amount
            lines.append(
                f"{arrow} `{sym}` {side.upper()} · `{strat}`\n"
                f"   Entry `{_fmt_price(entry)}` | Size `{amount:.6g}` | ≈`${notional:,.2f}`\n"
                f"   SL `{_fmt_price(sl)}` | TP `{_fmt_price(tp)}`"
            )

    breakdown = s.get("strategy_breakdown", {}) or {}
    if breakdown:
        lines += ["", SEP, "BY STRATEGY · 7D", SEP]
        for name, d in sorted(breakdown.items()):
            closed = int(d.get("wins", 0) or 0) + int(d.get("losses", 0) or 0)
            wr_v = d.get("win_rate")
            wr_s = "—" if wr_v is None else f"{float(wr_v):.1f}%"
            lines.append(
                f"`{_short_strategy(name)}` signals `{int(d.get('signals', 0) or 0)}` · "
                f"closed `{closed}` · WR `{wr_s}`"
            )

    # Build a useful per-symbol view from the local outcomes journal.
    recent = list(s.get("recent", []) or [])
    if recent:
        per_symbol = defaultdict(lambda: {"trades": 0, "wins": 0, "net": 0.0, "r": 0.0})
        for t in recent:
            sym = _short_symbol(t.get("symbol"))
            d = per_symbol[sym]
            d["trades"] += 1
            d["wins"] += 1 if t.get("won", False) else 0
            d["net"] += float(t.get("pnl_usd", 0) or 0)
            d["r"] += float(t.get("pnl_r", 0) or 0)
        lines += ["", SEP, "BY SYMBOL · RECENT JOURNAL", SEP]
        for sym, d in sorted(per_symbol.items(), key=lambda kv: -kv[1]["net"]):
            local_wr = d["wins"] / d["trades"] * 100 if d["trades"] else 0
            sign = "+" if d["net"] >= 0 else "-"
            lines.append(
                f"`{sym:<5}` {d['trades']} trades · `{local_wr:.0f}%WR` · "
                f"`{sign}${abs(d['net']):,.2f}` · `{d['r']:+.2f}R`"
            )

        lines += ["", SEP, "LAST 5 CLOSED", SEP]
        for i, t in enumerate(reversed(recent[-5:]), 1):
            won = bool(t.get("won", False))
            pnl = float(t.get("pnl_usd", 0) or 0)
            pnl_r = float(t.get("pnl_r", 0) or 0)
            sign = "+" if pnl >= 0 else "-"
            side = str(t.get("side", "?")).upper()
            reason = str(t.get("reason_label") or t.get("reason") or "closed")
            lines.append(
                f"{i}. {'✅' if won else '❌'} `{_short_symbol(t.get('symbol'))}` {side} "
                f"`{sign}${abs(pnl):,.2f}` `{pnl_r:+.2f}R` · {_age_str(t.get('ts'))}\n"
                f"   {_short_strategy(t.get('strategy'))} · {reason}"
            )

    lines += [
        "",
        "_Paper/internal stats use the local journal; live OKX post-fee history is shown automatically in LIVE mode._",
    ]
    return "\n".join(lines)
