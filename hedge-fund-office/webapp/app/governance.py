"""
Governance engine — Sentinel Global Fund rules.
คำนวณ NAV, sleeve allocation, trim-zone alerts (Rule #3), drawdown vs peak,
และ 9-gate pre-trade checklist logic.
"""

# Rule #3 — Position Balance Framework
POSITION_BASE = 20.0      # ≤20% = optimal
POSITION_WATCH = 22.0     # 20-22% = watch
POSITION_TRIM = 23.0      # 23-25% = trim to 18-19%
POSITION_EMERGENCY = 25.0  # >25% = trim now
TRIM_TARGET = 19.0        # เป้าหลัง trim

# Sleeve targets
SLEEVE_TARGET = {"Growth": 55.0, "Income": 30.0, "Cash": 13.0}

# Drawdown alert levels (vs peak)
DRAWDOWN_LEVELS = [
    (25.0, "Critical", "Growth→25%, cash→35%, full team review"),
    (18.0, "Red", "Growth→40%, cash→20%, income+defensive only"),
    (12.0, "Orange", "Growth −10%, cash→10%, pause new entries"),
    (8.0, "Yellow", "Review all positions; report within 24h"),
]


def position_zone(pct: float) -> tuple[str, str]:
    """คืน (zone, action) ตาม Rule #3"""
    if pct > POSITION_EMERGENCY:
        return "EMERGENCY", "🚨 Trim ทันที"
    if pct >= POSITION_TRIM:
        return "TRIM", f"🔴 Trim → target {TRIM_TARGET:.0f}% (หา replacement ก่อน)"
    if pct >= POSITION_WATCH:
        return "WATCH", "⚠️ ประชุม: trim หรือ watch"
    return "BASE", "✅ Optimal"


def trim_to_target(shares: float, price: float, nav: float, target_pct: float = TRIM_TARGET) -> dict:
    """คำนวณจำนวนหุ้นที่ต้องขายเพื่อลง target %"""
    if nav <= 0 or price <= 0:
        return {"sell_shares": 0, "sell_value": 0.0}
    target_value = nav * target_pct / 100.0
    cur_value = shares * price
    sell_value = max(0.0, cur_value - target_value)
    return {"sell_shares": round(sell_value / price, 2), "sell_value": round(sell_value, 2)}


def build_portfolio(holdings, prices: dict) -> dict:
    """
    holdings: list ของ ORM Holding
    prices: dict {ticker: market record}
    คืน snapshot: rows, nav, sleeve alloc, alerts
    """
    rows, nav, book = [], 0.0, 0.0
    sleeve_val = {"Growth": 0.0, "Income": 0.0, "Cash": 0.0}

    for h in holdings:
        pr = prices.get(h.ticker.upper(), {})
        price = pr.get("price")
        value = (price or h.cost_basis) * h.shares
        cost = h.cost_basis * h.shares
        nav += value
        book += cost
        sleeve_val[h.sleeve if h.sleeve in sleeve_val else "Growth"] += value
        pl = value - cost
        pl_pct = (pl / cost * 100) if cost else 0.0
        rows.append({
            "ticker": h.ticker, "shares": h.shares, "cost_basis": h.cost_basis,
            "sleeve": h.sleeve, "price": price, "value": round(value, 2),
            "book": round(cost, 2), "pl": round(pl, 2), "pl_pct": round(pl_pct, 2),
            "change_pct": pr.get("change_pct"), "rsi14": pr.get("rsi14"),
            "macd": (pr.get("macd") or {}).get("status"),
            "price_status": pr.get("status", "OK"), "asof": pr.get("asof"),
        })

    # คำนวณ % + zone หลังรู้ NAV
    for r in rows:
        r["pct"] = round(r["value"] / nav * 100, 2) if nav else 0.0
        r["zone"], r["zone_action"] = position_zone(r["pct"])
        if r["zone"] in ("TRIM", "EMERGENCY") and r["price"]:
            r["trim"] = trim_to_target(r["shares"], r["price"], nav)
    rows.sort(key=lambda x: x["value"], reverse=True)

    sleeve = {k: {"value": round(v, 2), "pct": round(v / nav * 100, 2) if nav else 0.0,
                  "target": SLEEVE_TARGET[k], "drift": round((v / nav * 100 if nav else 0) - SLEEVE_TARGET[k], 2)}
              for k, v in sleeve_val.items()}

    # blended dividend yield ประมาณจากราคา (ต้องมี yield table แยก — ที่นี่คำนวณจาก note ถ้ามี)
    alerts = []
    for r in rows:
        if r["zone"] in ("TRIM", "EMERGENCY"):
            t = r.get("trim", {})
            alerts.append({"level": r["zone"], "ticker": r["ticker"],
                           "msg": f"{r['ticker']} {r['pct']:.2f}% — {r['zone_action']}"
                                  + (f" (ขาย ~{t.get('sell_shares')} หุ้น ≈ ${t.get('sell_value')})" if t else "")})
        elif r["zone"] == "WATCH":
            alerts.append({"level": "WATCH", "ticker": r["ticker"],
                           "msg": f"{r['ticker']} {r['pct']:.2f}% — {r['zone_action']}"})

    return {"rows": rows, "nav": round(nav, 2), "book": round(book, 2),
            "pl": round(nav - book, 2), "pl_pct": round((nav - book) / book * 100, 2) if book else 0.0,
            "sleeve": sleeve, "alerts": alerts}


def drawdown_status(nav: float, peak: float) -> dict:
    if peak <= 0:
        return {"dd_pct": 0.0, "level": "—", "action": "", "peak": peak}
    dd = (nav / peak - 1) * 100
    level, action = "SAFE", "✅ ปกติ"
    for thr, lvl, act in DRAWDOWN_LEVELS:
        if -dd >= thr:
            level, action = lvl, act
            break
    return {"dd_pct": round(dd, 2), "level": level, "action": action, "peak": round(peak, 2)}


# 9-Gate pre-trade checklist (Section 8 ของ investment-system)
GATES = [
    ("regime_ts", "Regime timestamp [V] ≤ 24 ชม."),
    ("regime_score", "Regime score ≥ 40 (Neutral+)"),
    ("momentum", "Momentum score ≥ 58/100"),
    ("soft_block", "Soft-block check (ถ้า apply)"),
    ("position_cap", "Position ≤ 20% NAV (Rule #3)"),
    ("atr_stop", "ATR stop ระบุแล้ว (Rule #4)"),
    ("data_quality", "DQS ≥ 70% + ทุก key data มี [V/E/U]"),
    ("stagger", "Stagger rule (ถ้าใกล้ Tier-1 event)"),
    ("cio_signoff", "James Hartwell (CIO) sign-off"),
]
