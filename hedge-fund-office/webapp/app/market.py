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


# ── Dividend / company info (cache 1 ชม.) ──────────────────────────────────
_INFO_CACHE: dict[str, tuple[float, dict]] = {}
_INFO_TTL = 3600


def _epoch_to_date(v) -> str | None:
    try:
        return datetime.fromtimestamp(int(v), tz=timezone.utc).strftime("%Y-%m-%d")
    except (TypeError, ValueError, OSError):
        return None


def fetch_info(symbol: str) -> dict:
    """dividendRate ($/หุ้น/ปี), yield, ex-div, earnings date — Rule #5: หาไม่ได้ = None"""
    symbol = symbol.upper()
    now = time.time()
    if symbol in _INFO_CACHE and now - _INFO_CACHE[symbol][0] < _INFO_TTL:
        return _INFO_CACHE[symbol][1]

    rec = {"symbol": symbol, "dividend_rate": None, "dividend_yield": None,
           "ex_dividend_date": None, "earnings_date": None, "name": None,
           "source": "yfinance", "status": "OK"}
    try:
        import yfinance as yf
        t = yf.Ticker(symbol)
        info = t.info or {}
        rec["name"] = info.get("shortName") or info.get("longName")
        rate = info.get("dividendRate") or info.get("trailingAnnualDividendRate")
        rec["dividend_rate"] = round(float(rate), 4) if rate else None
        y = info.get("dividendYield") or info.get("trailingAnnualDividendYield")
        if y:
            y = float(y)
            rec["dividend_yield"] = round(y if y > 1 else y * 100, 2)  # normalize เป็น %
        rec["ex_dividend_date"] = _epoch_to_date(info.get("exDividendDate"))
        try:
            ed = t.calendar.get("Earnings Date") if isinstance(t.calendar, dict) else None
            if ed:
                rec["earnings_date"] = str(ed[0] if isinstance(ed, (list, tuple)) else ed)[:10]
        except Exception:
            pass
    except Exception as e:
        rec["status"] = f"ERROR: {type(e).__name__}"
    _INFO_CACHE[symbol] = (now, rec)
    return rec


def fetch_info_many(symbols: list[str]) -> dict[str, dict]:
    return {s.upper(): fetch_info(s) for s in dict.fromkeys(s.upper() for s in symbols)}


# ── News (cache 15 นาที) ────────────────────────────────────────────────────
_NEWS_CACHE: dict[str, tuple[float, list]] = {}
_NEWS_TTL = 900


def fetch_news(symbols: list[str], per_symbol: int = 5) -> list[dict]:
    """ข่าวจาก yfinance ต่อ ticker — รวม, dedupe, เรียงใหม่สุดก่อน"""
    items, seen = [], set()
    for symbol in dict.fromkeys(s.upper() for s in symbols):
        now = time.time()
        if symbol in _NEWS_CACHE and now - _NEWS_CACHE[symbol][0] < _NEWS_TTL:
            arts = _NEWS_CACHE[symbol][1]
        else:
            arts = []
            try:
                import yfinance as yf
                for a in (yf.Ticker(symbol).news or [])[:per_symbol]:
                    c = a.get("content", a)  # yfinance >=0.2.5x ห่อใน content
                    title = c.get("title")
                    if not title:
                        continue
                    link = (c.get("canonicalUrl") or {}).get("url") or c.get("link") or a.get("link")
                    pub = (c.get("provider") or {}).get("displayName") or a.get("publisher")
                    ts = c.get("pubDate") or ""
                    if not ts and a.get("providerPublishTime"):
                        ts = _epoch_to_date(a["providerPublishTime"]) or ""
                    arts.append({"symbol": symbol, "title": title, "link": link,
                                 "publisher": pub or "—", "published": str(ts)[:16].replace("T", " ")})
            except Exception:
                pass
            _NEWS_CACHE[symbol] = (now, arts)
        for a in arts:
            if a["title"] not in seen:
                seen.add(a["title"])
                items.append(a)
    items.sort(key=lambda x: x["published"], reverse=True)
    return items


# ── Multi-source market news (RSS ทางการ — Bloomberg/CNN/CNBC/MarketWatch) ──
_UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                     "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"}
RSS_SOURCES = [
    ("Bloomberg Markets",   "https://feeds.bloomberg.com/markets/news.rss"),
    ("Bloomberg Economics", "https://feeds.bloomberg.com/economics/news.rss"),
    ("CNN Business",        "http://rss.cnn.com/rss/money_latest.rss"),
    ("CNBC Markets",        "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664"),
    ("MarketWatch",         "https://feeds.content.dowjones.io/public/rss/mw_topstories"),
]
_RSS_CACHE: dict[str, tuple[float, list]] = {}
_RSS_TTL = 900


def _fetch_rss(source: str, url: str, limit: int = 8) -> list[dict]:
    """ดึง+parse RSS ด้วย stdlib — fail เงียบต่อ feed (แหล่งที่ล่มแค่หายไป ไม่พังหน้า)"""
    now = time.time()
    if url in _RSS_CACHE and now - _RSS_CACHE[url][0] < _RSS_TTL:
        return _RSS_CACHE[url][1]
    items = []
    try:
        import urllib.request
        import xml.etree.ElementTree as ET
        req = urllib.request.Request(url, headers=_UA)
        with urllib.request.urlopen(req, timeout=10) as r:
            root = ET.fromstring(r.read())
        for it in root.iter("item"):
            title = (it.findtext("title") or "").strip()
            if not title:
                continue
            pub = (it.findtext("pubDate") or "")[:25]
            items.append({"source": source, "title": title,
                          "link": (it.findtext("link") or "").strip(), "published": pub})
            if len(items) >= limit:
                break
    except Exception:
        pass
    _RSS_CACHE[url] = (now, items)
    return items


def fetch_market_news() -> list[dict]:
    """ข่าวตลาดรวมหลายสำนัก — แหล่งไหนล่มก็ข้าม (Rule #5: มีเท่าไหร่แสดงเท่านั้น)"""
    out, seen = [], set()
    for source, url in RSS_SOURCES:
        for a in _fetch_rss(source, url):
            if a["title"] not in seen:
                seen.add(a["title"])
                out.append(a)
    return out


# ── CNN Fear & Greed Index (ใช้ใน contrarian framework §7) ──────────────────
_FG_CACHE: tuple[float, dict] | None = None


def fetch_fear_greed() -> dict:
    """F&G score 0-100 + rating — พร้อม action ตามกฎกองทุน"""
    global _FG_CACHE
    now = time.time()
    if _FG_CACHE and now - _FG_CACHE[0] < 1800:
        return _FG_CACHE[1]
    rec = {"score": None, "rating": None, "action": None,
           "source": "CNN Fear & Greed", "status": "OK"}
    try:
        import json as _json
        import urllib.request
        hdrs = dict(_UA, Accept="application/json",
                    Referer="https://edition.cnn.com/markets/fear-and-greed")
        req = urllib.request.Request(
            "https://production.dataviz.cnn.io/index/fearandgreed/graphdata", headers=hdrs)
        with urllib.request.urlopen(req, timeout=10) as r:
            data = _json.loads(r.read())
        fg = data.get("fear_and_greed", {})
        score = fg.get("score")
        if score is not None:
            score = round(float(score))
            rec["score"], rec["rating"] = score, (fg.get("rating") or "").title()
            rec["action"] = (
                "🟢 ALL-IN ZONE — extreme fear (รอ stabilize)" if score < 15 else
                "🟢 เริ่ม scale in 25-30%" if score < 25 else
                "⚪ Neutral — ถือตามแผน" if score < 65 else
                "🟡 เริ่ม TRIM ตัวที่กำไร" if score < 75 else
                "🟠 TRIM 30-50% satellite" if score <= 85 else
                "🔴 EUPHORIA — take profit เชิงรุก")
    except Exception as e:
        rec["status"] = f"ERROR: {type(e).__name__}"
    _FG_CACHE = (now, rec)
    return rec


# ── Research links ต่อ ticker (แหล่งที่ไม่มี API — ให้ทีมกดเข้าไปค้น) ─────────
def research_links(symbol: str) -> list[dict]:
    s = symbol.upper()
    return [
        {"name": "StockAnalysis", "url": f"https://stockanalysis.com/stocks/{s}/"},
        {"name": "TradingView",   "url": f"https://www.tradingview.com/symbols/{s}/"},
        {"name": "Yahoo Finance", "url": f"https://finance.yahoo.com/quote/{s}"},
        {"name": "Seeking Alpha", "url": f"https://seekingalpha.com/symbol/{s}"},
        {"name": "Finviz",        "url": f"https://finviz.com/quote.ashx?t={s}"},
    ]


# ── ปฏิทินเศรษฐกิจ H2-2026 (ตารางประกาศทางการ — static, ระบุแหล่ง) ─────────
ECON_CALENDAR_2026H2 = [
    {"date": "2026-07-03", "event": "Nonfarm Payrolls (มิ.ย.)", "kind": "NFP",  "source": "BLS schedule"},
    {"date": "2026-07-14", "event": "CPI (มิ.ย.)",              "kind": "CPI",  "source": "BLS schedule"},
    {"date": "2026-07-28", "event": "FOMC Meeting วันที่ 1",     "kind": "FOMC", "source": "Federal Reserve calendar"},
    {"date": "2026-07-29", "event": "FOMC Statement + แถลงข่าว", "kind": "FOMC", "source": "Federal Reserve calendar"},
    {"date": "2026-08-07", "event": "Nonfarm Payrolls (ก.ค.)",  "kind": "NFP",  "source": "BLS schedule"},
    {"date": "2026-08-12", "event": "CPI (ก.ค.)",               "kind": "CPI",  "source": "BLS schedule"},
    {"date": "2026-09-04", "event": "Nonfarm Payrolls (ส.ค.)",  "kind": "NFP",  "source": "BLS schedule"},
    {"date": "2026-09-11", "event": "CPI (ส.ค.)",               "kind": "CPI",  "source": "BLS schedule"},
    {"date": "2026-09-15", "event": "FOMC Meeting วันที่ 1",     "kind": "FOMC", "source": "Federal Reserve calendar"},
    {"date": "2026-09-16", "event": "FOMC Statement + Dot Plot", "kind": "FOMC", "source": "Federal Reserve calendar"},
    {"date": "2026-10-02", "event": "Nonfarm Payrolls (ก.ย.)",  "kind": "NFP",  "source": "BLS schedule"},
    {"date": "2026-10-13", "event": "CPI (ก.ย.)",               "kind": "CPI",  "source": "BLS schedule"},
    {"date": "2026-10-27", "event": "FOMC Meeting วันที่ 1",     "kind": "FOMC", "source": "Federal Reserve calendar"},
    {"date": "2026-10-28", "event": "FOMC Statement + แถลงข่าว", "kind": "FOMC", "source": "Federal Reserve calendar"},
    {"date": "2026-11-06", "event": "Nonfarm Payrolls (ต.ค.)",  "kind": "NFP",  "source": "BLS schedule"},
    {"date": "2026-11-12", "event": "CPI (ต.ค.)",               "kind": "CPI",  "source": "BLS schedule"},
    {"date": "2026-12-04", "event": "Nonfarm Payrolls (พ.ย.)",  "kind": "NFP",  "source": "BLS schedule"},
    {"date": "2026-12-10", "event": "CPI (พ.ย.)",               "kind": "CPI",  "source": "BLS schedule"},
    {"date": "2026-12-08", "event": "FOMC Meeting วันที่ 1",     "kind": "FOMC", "source": "Federal Reserve calendar"},
    {"date": "2026-12-09", "event": "FOMC Statement + Dot Plot", "kind": "FOMC", "source": "Federal Reserve calendar"},
]


def build_calendar(symbols: list[str]) -> list[dict]:
    """รวม: econ (static) + earnings + ex-dividend (สดจาก yfinance) เรียงตามวันที่"""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    events = [dict(e, symbol=None) for e in ECON_CALENDAR_2026H2 if e["date"] >= today]
    for sym in dict.fromkeys(s.upper() for s in symbols):
        info = fetch_info(sym)
        if info.get("earnings_date") and info["earnings_date"] >= today:
            events.append({"date": info["earnings_date"], "event": f"{sym} Earnings",
                           "kind": "EARNINGS", "symbol": sym, "source": "yfinance"})
        if info.get("ex_dividend_date") and info["ex_dividend_date"] >= today:
            events.append({"date": info["ex_dividend_date"], "event": f"{sym} XD วันขึ้นเครื่องหมาย",
                           "kind": "XD", "symbol": sym, "source": "yfinance"})
    events.sort(key=lambda e: e["date"])
    return events


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
