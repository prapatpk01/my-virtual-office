"""
Market data — yfinance fetch with in-memory TTL cache.
บน Railway (egress เปิด) ดึงได้จริง | ใน sandbox ที่ block Yahoo จะคืน status error
ทุก record มี source + asof (Governance Rule #5). ห้ามเดา — ค่าที่หาไม่ได้ = None.
"""
import os
import time
from datetime import datetime, timezone

CA_BUNDLE = "/root/.ccr/ca-bundle.crt"
if os.path.exists(CA_BUNDLE):
    os.environ.setdefault("REQUESTS_CA_BUNDLE", CA_BUNDLE)
    os.environ.setdefault("SSL_CERT_FILE", CA_BUNDLE)

_CACHE: dict[str, tuple[float, dict]] = {}
_TTL = 300  # 5 นาที


def _rsi(closes, period=14):
    if len(closes) < period + 1:
        return None
    gains, losses = [], []
    for i in range(1, len(closes)):
        ch = closes[i] - closes[i - 1]
        gains.append(max(ch, 0.0))
        losses.append(max(-ch, 0.0))
    ag = sum(gains[:period]) / period
    al = sum(losses[:period]) / period
    for i in range(period, len(gains)):
        ag = (ag * (period - 1) + gains[i]) / period
        al = (al * (period - 1) + losses[i]) / period
    if al == 0:
        return 100.0
    return round(100 - 100 / (1 + ag / al), 1)


def _ema(vals, period):
    k = 2 / (period + 1)
    e = vals[0]
    for v in vals[1:]:
        e = v * k + e * (1 - k)
    return e


def _macd(closes, fast=12, slow=26, signal=9):
    if len(closes) < slow + signal:
        return None
    line = [_ema(closes[:i][-fast:], fast) - _ema(closes[:i][-slow:], slow)
            for i in range(slow, len(closes) + 1)]
    sig = _ema(line[-signal:], signal)
    macd = line[-1]
    return {
        "macd": round(macd, 3), "signal": round(sig, 3), "hist": round(macd - sig, 3),
        "status": "BULLISH" if macd > sig and macd > 0 else
                  "BEARISH" if macd < sig else "NEUTRAL",
    }


def fetch(symbol: str) -> dict:
    symbol = symbol.upper()
    now = time.time()
    if symbol in _CACHE and now - _CACHE[symbol][0] < _TTL:
        return _CACHE[symbol][1]

    rec = {"symbol": symbol, "price": None, "prev_close": None, "change_pct": None,
           "rsi14": None, "macd": None, "source": "yfinance", "asof": None, "status": "OK"}
    try:
        import yfinance as yf
        hist = yf.Ticker(symbol).history(period="3mo")
        if hist.empty:
            rec["status"] = "DATA UNAVAILABLE"
        else:
            closes = [float(c) for c in hist["Close"].tolist()]
            rec["price"] = round(closes[-1], 2)
            if len(closes) > 1:
                rec["prev_close"] = round(closes[-2], 2)
                rec["change_pct"] = round((closes[-1] / closes[-2] - 1) * 100, 2)
            rec["rsi14"] = _rsi(closes)
            rec["macd"] = _macd(closes)
            rec["asof"] = str(hist.index[-1].date())
    except Exception as e:
        rec["status"] = f"ERROR: {type(e).__name__}"
    _CACHE[symbol] = (now, rec)
    return rec


def fetch_many(symbols: list[str]) -> dict[str, dict]:
    return {s.upper(): fetch(s) for s in dict.fromkeys(s.upper() for s in symbols)}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
