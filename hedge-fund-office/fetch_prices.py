#!/usr/bin/env python3
"""
Sentinel Global Fund — Price & Market Data Fetcher
===================================================
ดึงราคาหุ้น/ETF และ indicator พื้นฐานจากแหล่งฟรี (yfinance เป็นหลัก)
สำหรับใช้กับ INVESTMENT-SYSTEM skill + PORTFOLIO-SNAPSHOT

Owner : Nina Okonkwo (Data & Source Engineer) + Leo Tanaka (Real-time Data)
Rule  : ทุก data point ต้องมี source + timestamp (Governance Rule #5)
        DATA UNAVAILABLE = ห้ามเดา (เขียน None)

Usage:
    python3 fetch_prices.py                      # ดึงทั้งพอร์ต (holdings.json)
    python3 fetch_prices.py GPIQ SPMO VOO        # ดึงเฉพาะที่ระบุ
    python3 fetch_prices.py --watchlist          # ดึง watchlist ด้วย
    python3 fetch_prices.py --json               # output เป็น JSON

หมายเหตุ network:
    ต้องรันในสภาพแวดล้อมที่เข้าถึง Yahoo Finance ได้ (เครื่องคุณ / live fund env).
    ใน sandbox ของ Claude Code บน web, host การเงินถูก block โดย egress policy —
    ให้ใช้ WebSearch/screenshot แทน (Leo Tanaka parse).
"""

import sys
import json
import os
from datetime import datetime, timezone

# ── ค่าคงที่กองทุน ────────────────────────────────────────────────────────
PORTFOLIO_TICKERS = ["GPIQ", "SPMO", "VOO", "BALI", "SCHD", "SGOV", "O", "JAAA", "HSBC"]
BENCHMARK = "SPY"
WATCHLIST = ["AVDV", "DFIV", "MAIN", "QDVO", "RKLB", "CRWV", "UAL", "BKNG", "CAT", "UNH"]

CA_BUNDLE = "/root/.ccr/ca-bundle.crt"  # ให้ requests/yfinance เชื่อ proxy CA ถ้ามี


def _setup_ca():
    """ชี้ CA bundle ให้ requests/yfinance ถ้าอยู่หลัง agent proxy"""
    if os.path.exists(CA_BUNDLE):
        os.environ.setdefault("REQUESTS_CA_BUNDLE", CA_BUNDLE)
        os.environ.setdefault("SSL_CERT_FILE", CA_BUNDLE)


def calc_rsi(closes, period=14):
    """RSI(14) — Wilder's smoothing. คืน None ถ้าข้อมูลไม่พอ"""
    if len(closes) < period + 1:
        return None
    gains, losses = [], []
    for i in range(1, len(closes)):
        ch = closes[i] - closes[i - 1]
        gains.append(max(ch, 0))
        losses.append(max(-ch, 0))
    avg_g = sum(gains[:period]) / period
    avg_l = sum(losses[:period]) / period
    for i in range(period, len(gains)):
        avg_g = (avg_g * (period - 1) + gains[i]) / period
        avg_l = (avg_l * (period - 1) + losses[i]) / period
    if avg_l == 0:
        return 100.0
    rs = avg_g / avg_l
    return round(100 - (100 / (1 + rs)), 1)


def _ema(vals, period):
    k = 2 / (period + 1)
    e = vals[0]
    for v in vals[1:]:
        e = v * k + e * (1 - k)
    return e


def calc_macd(closes, fast=12, slow=26, signal=9):
    """MACD(12,26,9). คืน dict {macd, signal, hist, status} หรือ None"""
    if len(closes) < slow + signal:
        return None
    macd_line = []
    for i in range(slow, len(closes) + 1):
        window = closes[:i]
        macd_line.append(_ema(window[-fast:], fast) - _ema(window[-slow:], slow))
    sig = _ema(macd_line[-signal:], signal)
    macd = macd_line[-1]
    hist = macd - sig
    return {
        "macd": round(macd, 3),
        "signal": round(sig, 3),
        "hist": round(hist, 3),
        "status": "BULLISH" if macd > sig and macd > 0 else
                  "BEARISH" if macd < sig else "NEUTRAL",
    }


def fetch_ticker(symbol):
    """ดึงราคา + technicals ของ 1 symbol ผ่าน yfinance"""
    import yfinance as yf
    rec = {
        "symbol": symbol, "price": None, "prev_close": None, "change_pct": None,
        "rsi14": None, "macd": None, "vol": None, "avg_vol_20d": None,
        "source": "yfinance/Yahoo", "asof": None, "status": "OK",
    }
    try:
        t = yf.Ticker(symbol)
        hist = t.history(period="3mo")  # ~63 แท่ง พอสำหรับ MACD+RSI
        if hist.empty:
            rec["status"] = "DATA UNAVAILABLE"
            return rec
        closes = [float(c) for c in hist["Close"].tolist()]
        vols = [float(v) for v in hist["Volume"].tolist()]
        rec["price"] = round(closes[-1], 2)
        rec["prev_close"] = round(closes[-2], 2) if len(closes) > 1 else None
        if rec["prev_close"]:
            rec["change_pct"] = round((closes[-1] / closes[-2] - 1) * 100, 2)
        rec["rsi14"] = calc_rsi(closes)
        rec["macd"] = calc_macd(closes)
        rec["vol"] = int(vols[-1])
        rec["avg_vol_20d"] = int(sum(vols[-20:]) / 20) if len(vols) >= 20 else None
        rec["asof"] = str(hist.index[-1].date())
    except Exception as e:
        rec["status"] = f"ERROR: {type(e).__name__}: {e}"
    return rec


def main():
    _setup_ca()
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}

    tickers = args if args else PORTFOLIO_TICKERS[:]
    if "--watchlist" in flags:
        tickers += WATCHLIST
    tickers = [BENCHMARK] + tickers  # ใส่ benchmark เสมอ

    ts = datetime.now(timezone.utc).isoformat()
    results = [fetch_ticker(s) for s in dict.fromkeys(tickers)]  # dedupe, keep order

    if "--json" in flags:
        print(json.dumps({"fetched_at_utc": ts, "data": results}, indent=2))
        return

    print(f"\n  SENTINEL GLOBAL FUND — Price Feed  |  {ts}")
    print(f"  Source: yfinance/Yahoo Finance\n")
    print(f"  {'Ticker':<7}{'Price':>10}{'Chg%':>8}{'RSI14':>7}{'MACD':>9}{'AsOf':>13}")
    print("  " + "─" * 56)
    for r in results:
        if r["status"] != "OK":
            print(f"  {r['symbol']:<7}{'—':>10}{'':>8}{'':>7}{'':>9}  {r['status']}")
            continue
        macd = r["macd"]["status"][:4] if r["macd"] else "—"
        rsi = r["rsi14"] if r["rsi14"] is not None else "—"
        chg = f"{r['change_pct']:+.2f}" if r["change_pct"] is not None else "—"
        print(f"  {r['symbol']:<7}{r['price']:>10}{chg:>8}{str(rsi):>7}{macd:>9}{r['asof']:>13}")
    print()


if __name__ == "__main__":
    main()
