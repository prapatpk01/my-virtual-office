"""Adaptive /stats — built from OKX positions-history so the numbers match the
OKX app exactly.

Source-of-truth rules (see the product spec):
  * Trades / Win-rate / PnL come from OKX positions-history rows. ONE row = one
    round-trip position (open -> fully closed; any T1/T2 partials are already
    collapsed by OKX). We use OKX's own `realized_pnl` (= pnl - fees - funding)
    verbatim — never recompute PnL.
  * TP1/TP2/SL breakdown needs a target concept OKX doesn't have, so it comes
    from the bot's LOCAL journal (signal_state outcomes). IRON RULE: the
    denominator is the OKX trade count, not the journal count — trades with no
    journal match are reported as "Untracked", never silently dropped.
  * OVERALL resets monthly (current UTC month); BY SYMBOL is all-time since the
    configured start date.

Rendered as PLAIN TEXT (no Markdown/parse_mode) with emoji + "――――――" rules,
because Telegram would otherwise show raw * and _ characters.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

RULE = "――――――――――――――"


def _sym_short(symbol: str) -> str:
    return (symbol or "").split("/")[0].split("-")[0].upper()


def since_ts_for(date_str: str = "2026-07-16") -> int:
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _month_bounds(now_ms: int):
    """Return (cur_start_ms, cur_label, prev_start_ms, prev_label) in UTC."""
    now = datetime.fromtimestamp(now_ms / 1000, tz=timezone.utc)
    cur_start = datetime(now.year, now.month, 1, tzinfo=timezone.utc)
    if now.month == 1:
        prev_start = datetime(now.year - 1, 12, 1, tzinfo=timezone.utc)
    else:
        prev_start = datetime(now.year, now.month - 1, 1, tzinfo=timezone.utc)
    return (int(cur_start.timestamp() * 1000), cur_start.strftime("%B %Y"),
            int(prev_start.timestamp() * 1000), prev_start.strftime("%B %Y"))


def _age_str(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 3600:
        return f"{seconds // 60}m ago"
    if seconds < 86400:
        return f"{seconds / 3600:.1f}h ago"
    return f"{seconds / 86400:.1f}d ago"


def _classify(outcome: dict) -> str:
    """Map a journal outcome to TP1 / TP2 / SL for the breakdown. Best-effort:
    a losing close is SL; a winner that ran well past the first target is TP2,
    a smaller winner is TP1."""
    reason = (outcome.get("reason") or "").lower()
    won = bool(outcome.get("won"))
    pnl_r = outcome.get("pnl_r") or 0.0
    if not won or "stop_loss" in reason or "hard_sl" in reason:
        return "sl"
    if "tp2" in reason or "take_profit" in reason or "hard_tp" in reason or pnl_r >= 1.2:
        return "tp2"
    return "tp1"  # any smaller win / trailed-BE / 0.8R lock / early trend-exit in profit


def _match_journal(okx_trade: dict, journal: list[dict], used: set) -> dict | None:
    """Find the journal outcome for an OKX position by symbol + close time
    proximity (OKX close_ts vs journal ts, within 15 min)."""
    short = _sym_short(okx_trade["symbol"])
    ct = okx_trade["close_ts"]
    best, best_dt = None, 15 * 60_000 + 1
    for i, o in enumerate(journal):
        if i in used or _sym_short(o.get("symbol", "")) != short:
            continue
        dt = abs(int(o.get("ts", 0)) - ct)
        if dt < best_dt:
            best, best_dt, best_i = o, dt, i
    if best is not None:
        used.add(best_i)
    return best


def build_adaptive_stats(okx_positions: list[dict], journal_outcomes: list[dict],
                         open_positions_detail: list[dict] | None,
                         balance: float | None, since_ms: int,
                         now_ms: int | None = None) -> dict:
    now_ms = now_ms or int(time.time() * 1000)
    cur_start, cur_label, prev_start, prev_label = _month_bounds(now_ms)

    trades = [t for t in okx_positions if t.get("close_ts", 0) >= since_ms]
    trades.sort(key=lambda t: t.get("close_ts", 0))

    # ── OVERALL (current month) ───────────────────────────────────────────
    month = [t for t in trades if t["close_ts"] >= cur_start]
    n = len(month)
    wins = sum(1 for t in month if t["realized_pnl"] > 0)
    losses = n - wins
    net = round(sum(t["realized_pnl"] for t in month), 2)
    prev_net = round(sum(t["realized_pnl"] for t in trades
                         if prev_start <= t["close_ts"] < cur_start), 2)

    # TP1/TP2/SL from journal, denominator = OKX month trade count
    used: set = set()
    tp1 = tp2 = sl = tracked = 0
    for t in month:
        o = _match_journal(t, journal_outcomes, used)
        if o is None:
            continue
        tracked += 1
        cat = _classify(o)
        tp1 += cat == "tp1"; tp2 += cat == "tp2"; sl += cat == "sl"
    untracked = n - tracked

    # ── BY SYMBOL (all-time since since_ms) ───────────────────────────────
    per: dict[str, dict] = {}
    for t in trades:
        d = per.setdefault(_sym_short(t["symbol"]), {"trades": 0, "wins": 0, "net": 0.0})
        d["trades"] += 1
        d["wins"] += 1 if t["realized_pnl"] > 0 else 0
        d["net"] += t["realized_pnl"]
    total_n = len(trades)
    total_wins = sum(1 for t in trades if t["realized_pnl"] > 0)
    total_net = round(sum(t["realized_pnl"] for t in trades), 2)

    # ── LAST 5 (by close time, across symbols) ────────────────────────────
    last5 = [{
        "short": _sym_short(t["symbol"]), "side": t.get("side", ""),
        "net": round(t["realized_pnl"], 2), "won": t["realized_pnl"] > 0,
        "age_s": max(0, int((now_ms - t["close_ts"]) / 1000)),
    } for t in reversed(trades[-5:])]

    open_list = []
    for p in (open_positions_detail or []):
        entry = p.get("entry_price") or 0.0
        mark = p.get("mark_price") or entry
        amt = p.get("amount") or 0.0
        upnl = p.get("unrealized_pnl")
        if upnl is None and entry:
            sign = 1 if p.get("side") == "long" else -1
            upnl = sign * (mark - entry) * amt
        open_list.append({
            "short": _sym_short(p.get("symbol", "")), "side": p.get("side", ""),
            "amount": amt, "entry": entry,
            "notional": round(p.get("notional") or (amt * mark), 2),
            "upnl": round(upnl, 2) if upnl is not None else None,
        })

    return {
        "source": "okx",
        "balance": balance,
        "month_label": cur_label, "prev_month_label": prev_label,
        "trades": n, "wins": wins, "losses": losses,
        "win_rate": round(wins / n * 100) if n else 0,
        "net_pnl": net, "prev_month_pnl": prev_net,
        "tp1": tp1, "tp2": tp2, "sl": sl, "untracked": untracked,
        "per_symbol": {k: {"trades": v["trades"],
                           "win_rate": round(v["wins"] / v["trades"] * 100) if v["trades"] else 0,
                           "net": round(v["net"], 2)} for k, v in per.items()},
        "total": {"trades": total_n,
                  "win_rate": round(total_wins / total_n * 100) if total_n else 0,
                  "net": total_net},
        "open_list": open_list,
        "last5": last5,
    }


def _money(v: float) -> str:
    return f"{'+' if v >= 0 else '-'}${abs(v):,.2f}"


def render_adaptive_stats(s: dict) -> str:
    """Plain-text render (NO markdown)."""
    if s.get("source") != "okx":
        # Fallback block
        lines = ["⚠️ OKX history unavailable — showing local journal",
                 f"💰 Balance: ${s.get('balance', 0) or 0:,.2f}" if s.get("balance") is not None
                 else "💰 Balance: —",
                 f"Trades: {s.get('trades', 0)}  WR: {s.get('win_rate', 0)}%  "
                 f"Net: {_money(s.get('net_pnl', 0.0))}"]
        return "\n".join(lines)

    L = []
    bal = s.get("balance")
    L.append(f"💰 Balance: ${bal:,.2f}" if bal is not None else "💰 Balance: —")
    op = s.get("open_list", [])
    L.append(f"📌 Open positions: {len(op)}")
    for p in op:
        arrow = "🟢" if p["side"] == "long" else "🔴"
        upnl = "" if p["upnl"] is None else f"  uPnL {_money(p['upnl'])}"
        L.append(f"   {arrow} {p['short']} {p['side'].upper()} {p['amount']:g} "
                 f"@ {p['entry']:,.4f} (≈${p['notional']:,.0f}){upnl}")

    n = s["trades"]
    L += ["", RULE, f"📊 OVERALL — {s['month_label']}", RULE,
          f"Trades   : {n}  ({s['wins']}W / {s['losses']}L)",
          f"Win rate : {s['win_rate']}%"]
    d = n or 1
    L.append(f"TP1 hit  : {s['tp1']}/{n} ({s['tp1']*100//d}%)   "
             f"TP2 hit : {s['tp2']}/{n} ({s['tp2']*100//d}%)   "
             f"SL only : {s['sl']}/{n} ({s['sl']*100//d}%)")
    if s.get("untracked"):
        L.append(f"Untracked: {s['untracked']}/{n} (closed while bot was offline — target unknown)")
    L.append(f"Net PnL  : {_money(s['net_pnl'])}  (post-fee, from OKX)")
    L.append(f"{s['prev_month_label']} PnL : {_money(s['prev_month_pnl'])}")

    L += ["", RULE, "📈 BY SYMBOL (all-time)", RULE]
    per = s.get("per_symbol", {})
    for sym, v in sorted(per.items(), key=lambda kv: -kv[1]["net"]):
        L.append(f"{sym:<5} {v['trades']} trades  {v['win_rate']}%WR  {_money(v['net'])}")
    tot = s["total"]
    L.append(RULE[:12])
    L.append(f"TOTAL {tot['trades']} trades  {tot['win_rate']}%WR  {_money(tot['net'])}")

    L += ["", RULE, "🕒 LAST 5 TRADES", RULE]
    for i, t in enumerate(s.get("last5", []), 1):
        e = "✅" if t["won"] else "❌"
        L.append(f"{i}. {e} {t['short']} {t['side'].upper()} {_money(t['net'])} — {_age_str(t['age_s'])}")
    return "\n".join(L)
