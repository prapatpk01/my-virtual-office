"""Build /stats from OKX order history — the real, post-fee source of truth.

Groups filled orders (opens + reduce-only closes) into round-trip trades per
symbol, using OKX's own realized `pnl` and `fee` per order so the numbers match
the OKX app exactly:

    trade net PnL = sum(realized pnl of the trade's reduce fills)
                  - sum(fees of every fill in the trade, open + closes)

A trade spans from the position going from flat -> open to back to flat.
Partial closes within a trade are what make TP1/TP2 detection possible.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone


def _sym_short(symbol: str) -> str:
    # "SOL/USDT:USDT" -> "SOL"
    return symbol.split("/")[0].split("-")[0].upper()


def group_trades(orders: list[dict]) -> list[dict]:
    """orders: normalized fills for ONE symbol, chronological. Returns a list of
    round-trip trades, each: {symbol, side, open_ts, close_ts, net_pnl,
    partials(int), fees, realized, won}."""
    trades: list[dict] = []
    pos = 0.0            # signed position size
    cur: dict | None = None
    for o in orders:
        amt = o["amount"] or 0.0
        signed = amt if o["side"] == "buy" else -amt
        is_reduce = o["reduce_only"] or (o["pnl"] not in (None, 0.0))
        if cur is None and not is_reduce:
            # opening a new trade
            cur = {"symbol": o["symbol"], "side": "long" if signed > 0 else "short",
                   "open_ts": o["ts"], "close_ts": o["ts"], "realized": 0.0,
                   "fees": 0.0, "partials": 0, "reduces": 0}
        if cur is not None:
            cur["fees"] += o["fee"] or 0.0
            if is_reduce:
                cur["realized"] += (o["pnl"] or 0.0)
                cur["reduces"] += 1
                cur["close_ts"] = o["ts"]
            pos += signed
            # position back to (near) flat -> trade complete
            if abs(pos) < 1e-9:
                cur["partials"] = max(0, cur["reduces"] - 1)
                cur["net_pnl"] = round(cur["realized"] - cur["fees"], 6)
                cur["won"] = cur["net_pnl"] > 0
                trades.append(cur)
                cur = None
                pos = 0.0
    return trades


def build_stats(orders_by_symbol: dict[str, list[dict]], balance: float | None,
                open_positions: int, since_ms: int,
                open_positions_detail: list[dict] | None = None) -> dict:
    """orders_by_symbol: {symbol: [normalized fills chronological]}.
    open_positions_detail: live position dicts (symbol/side/amount/entry_price/
    mark_price/unrealized_pnl) to LIST under an OPEN POSITIONS section.
    Returns the dict the Telegram /stats renderer consumes."""
    all_trades: list[dict] = []
    per_symbol: dict[str, dict] = {}
    for symbol, orders in orders_by_symbol.items():
        trades = [t for t in group_trades(orders) if t["close_ts"] >= since_ms]
        short = _sym_short(symbol)
        d = per_symbol.setdefault(short, {"trades": 0, "wins": 0, "net": 0.0})
        for t in trades:
            d["trades"] += 1
            d["wins"] += 1 if t["won"] else 0
            d["net"] += t["net_pnl"]
            t["short"] = short
            all_trades.append(t)

    all_trades.sort(key=lambda t: t["close_ts"])
    n = len(all_trades)
    wins = sum(1 for t in all_trades if t["won"])
    losses = n - wins
    net_total = round(sum(t["net_pnl"] for t in all_trades), 2)

    # TP1/TP2/SL among trades that ran the 2-TP structure (had >=1 partial)
    with_partial = [t for t in all_trades if t["partials"] >= 1]
    tp1_hits = len(with_partial)                                   # took a partial = TP1 fired
    tp2_hits = sum(1 for t in with_partial if t["won"] and t["reduces"] >= 2 and t["realized"] > 0)
    sl_only = sum(1 for t in all_trades if t["partials"] == 0 and not t["won"])
    denom = max(1, len(with_partial))

    open_list = []
    for p in (open_positions_detail or []):
        entry = p.get("entry_price") or 0.0
        mark = p.get("mark_price") or entry
        amt = p.get("amount") or 0.0
        upnl = p.get("unrealized_pnl")
        notional = p.get("notional") or (amt * mark)
        if upnl is None and entry:
            # derive uPnL if the exchange didn't hand it back
            sign = 1 if p.get("side") == "long" else -1
            upnl = sign * (mark - entry) * amt
        open_list.append({
            "short": _sym_short(p.get("symbol", "")),
            "side": p.get("side", ""),
            "amount": amt,
            "entry": entry,
            "mark": mark,
            "notional": round(notional, 2),
            "upnl": round(upnl, 4) if upnl is not None else None,
        })

    return {
        "source": "okx",
        "balance": balance,
        "open_positions": open_positions,
        "open_list": open_list,
        "trades": n, "wins": wins, "losses": losses,
        "win_rate": round(wins / n * 100, 1) if n else 0.0,
        "net_pnl": net_total,
        "tp1_hits": tp1_hits, "tp2_hits": tp2_hits, "sl_only": sl_only,
        "partial_denom": len(with_partial),
        "per_symbol": {k: {"trades": v["trades"],
                           "win_rate": round(v["wins"] / v["trades"] * 100, 0) if v["trades"] else 0,
                           "net": round(v["net"], 2)}
                       for k, v in per_symbol.items()},
        "last_trades": [
            {"short": t["short"], "side": t["side"], "net": round(t["net_pnl"], 2),
             "won": t["won"], "age_s": max(0, int(time.time() - t["close_ts"] / 1000))}
            for t in reversed(all_trades[-5:])
        ],
    }


def since_ts_for(date_str: str = "2026-07-16") -> int:
    """ms timestamp for the stats start date (default 16 Jul 2026 UTC)."""
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)
