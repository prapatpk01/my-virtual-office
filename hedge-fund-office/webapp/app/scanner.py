"""
Institutional High-Beta Momentum Swing Scanner v6.1 — quant implementation
7-phase framework คำนวณจาก OHLCV จริง (yfinance) — ไม่ใช่ web search
Rule #5: คำนวณไม่ได้ = DATA UNAVAILABLE, ไม่เดา
Rule #6: ไม่แสดง Win Probability (ไม่มี backtest รองรับ)
"""
import time
from datetime import datetime, timezone

# ── Universe ตามธีม Phase 3E (sector leaders — แก้ไขเพิ่มได้) ────────────────
SCAN_UNIVERSE = {
    "AI Infrastructure":  ["NVDA", "AVGO", "AMD", "MRVL", "SMCI", "VRT", "ANET", "COHR"],
    "Semiconductors":     ["TSM", "MU", "LRCX", "AMAT", "KLAC", "ASML", "ARM"],
    "Defense Tech":       ["KTOS", "AVAV", "LHX", "RTX", "PLTR", "BWXT"],
    "Quantum":            ["IONQ", "RGTI", "QBTS"],
    "Cybersecurity":      ["CRWD", "PANW", "ZS", "NET", "S", "FTNT"],
    "Cloud/DataCenter":   ["MSFT", "ORCL", "DDOG", "SNOW", "NBIS", "CRWV"],
    "Nuclear/Energy":     ["CCJ", "CEG", "VST", "OKLO", "SMR", "GEV"],
    "Robotics/Autonomy":  ["TSLA", "SYM", "ROK", "TER"],
    "AI Software":        ["META", "GOOGL", "NOW", "APP", "SOUN"],
    "Space/Aero":         ["RKLB", "ASTS", "LUNR", "RDW", "HEI"],
}
THEME_PRIORITY = list(SCAN_UNIVERSE.keys())
TICKER_THEME = {t: th for th, ts in SCAN_UNIVERSE.items() for t in ts}

_LAST: dict | None = None   # cache ผลสแกนล่าสุด (in-memory)


def _ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def _rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, 1e-12)
    return 100 - 100 / (1 + rs)


def _metrics(df, spy_ret20, qqq_ret20, spy_close):
    """คำนวณทุก indicator ต่อ 1 ตัว — คืน dict หรือ None ถ้าข้อมูลไม่พอ"""
    df = df.dropna()
    if len(df) < 60:
        return None
    close, high, low, vol = df["Close"], df["High"], df["Low"], df["Volume"]
    price = float(close.iloc[-1])
    if price <= 0:
        return None

    ema10, ema20 = _ema(close, 10), _ema(close, 20)
    sma50 = close.rolling(50).mean()
    rsi = float(_rsi(close).iloc[-1])
    macd_line = _ema(close, 12) - _ema(close, 26)
    macd_sig = _ema(macd_line, 9)
    hist = macd_line - macd_sig
    tr = (high - low).combine(abs(high - close.shift()), max).combine(abs(low - close.shift()), max)
    atr_pct = float(tr.rolling(14).mean().iloc[-1] / price * 100)
    obv = ((close.diff() > 0).astype(int) * 2 - 1).mul(vol).cumsum()
    ret20 = float(close.iloc[-1] / close.iloc[-21] - 1) * 100 if len(close) > 21 else None

    # beta คำนวณจาก daily returns เทียบ SPY (6 เดือน)
    beta = None
    try:
        r = close.pct_change().dropna().tail(120)
        rs = spy_close.pct_change().dropna().tail(120)
        j = r.align(rs, join="inner")
        if len(j[0]) > 60 and float(j[1].var()) > 0:
            beta = round(float(j[0].cov(j[1]) / j[1].var()), 2)
    except Exception:
        pass

    v20 = float(vol.rolling(20).mean().iloc[-1])
    swing_lo = float(low.tail(40).min())
    swing_hi = float(high.tail(40).max())

    return {
        "price": round(price, 2), "rsi": round(rsi, 1),
        "macd_pos": bool(macd_line.iloc[-1] > 0),
        "hist_expanding": bool(hist.iloc[-1] > hist.iloc[-3]),
        "ema10": round(float(ema10.iloc[-1]), 2), "ema20": round(float(ema20.iloc[-1]), 2),
        "sma50": round(float(sma50.iloc[-1]), 2),
        "aligned": bool(ema10.iloc[-1] > ema20.iloc[-1] > sma50.iloc[-1]),
        "above_all": bool(price > ema10.iloc[-1] and price > ema20.iloc[-1] and price > sma50.iloc[-1]),
        "hh_hl": bool(high.tail(10).max() > high.tail(20).head(10).max()
                      and low.tail(10).min() > low.tail(20).head(10).min()),
        "atr_pct": round(atr_pct, 2), "beta": beta,
        "ret20": round(ret20, 2) if ret20 is not None else None,
        "vs_spy": round(ret20 - spy_ret20, 2) if ret20 is not None and spy_ret20 is not None else None,
        "vs_qqq": round(ret20 - qqq_ret20, 2) if ret20 is not None and qqq_ret20 is not None else None,
        "vol_x20d": round(float(vol.iloc[-1]) / v20, 2) if v20 > 0 else None,
        "vol5_x20": round(float(vol.rolling(5).mean().iloc[-1]) / v20, 2) if v20 > 0 else None,
        "obv_rising": bool(obv.iloc[-1] > obv.iloc[-10]),
        "dollar_vol_m": round(v20 * price / 1e6, 1),
        "gain5d": round(float(close.iloc[-1] / close.iloc[-6] - 1) * 100, 2) if len(close) > 6 else 0.0,
        "ext_pct": round((price / float(ema10.iloc[-1]) - 1) * 100, 2),
        "swing_lo": swing_lo, "swing_hi": swing_hi,
    }


def _score(m, theme_rank):
    """Phase 7 — weighted score /100 + breakdown"""
    a = (10 if 55 <= m["rsi"] <= 78 else 0) + (8 if m["macd_pos"] else 0) \
        + (7 if m["hist_expanding"] else 0) \
        + (5 if (m["vs_spy"] or 0) > 0 else 0) + (5 if (m["vs_qqq"] or 0) > 0 else 0)
    b = (10 if (m["vol_x20d"] or 0) >= 1.5 else 5 if (m["vol_x20d"] or 0) >= 1.0 else 0) \
        + (8 if (m["vol5_x20"] or 0) > 1.0 else 0) + (7 if m["obv_rising"] else 0)
    c = (8 if m["above_all"] else 0) + (4 if m["aligned"] else 0) + (3 if m["hh_hl"] else 0)
    d = (4 if (m["beta"] or 0) > 1.3 else 0) + (3 if m["atr_pct"] > 3 else 0) \
        + (3 if m["dollar_vol_m"] > 50 else 0)
    e = 5 if theme_rank < 3 else 3
    f = 6 if (m["gain5d"] > 5 and m["obv_rising"]) else 0   # momentum-event proxy (ไม่มี earnings feed)
    return {"A_momentum": a, "B_volume": b, "C_structure": c,
            "D_beta": d, "E_sector": e, "F_catalyst": f,
            "total": a + b + c + d + e + f}


def _trade(m):
    """Phase 6 — entry/stop/target/R:R จากโครงสร้างราคาจริง"""
    entry_lo, entry_hi = m["ema10"], round(m["ema10"] * 1.03, 2)
    entry_mid = round((entry_lo + entry_hi) / 2, 2)
    stop = min(m["ema20"], round(m["swing_lo"] * 1.0, 2))
    stop = round(stop, 2)
    target = round(m["swing_lo"] + 1.618 * (m["swing_hi"] - m["swing_lo"]), 2)
    upside = round((target / m["price"] - 1) * 100, 1)
    risk = entry_mid - stop
    rr = round((target - entry_mid) / risk, 1) if risk > 0 else None
    return {"entry_lo": entry_lo, "entry_hi": entry_hi, "stop": stop,
            "target": target, "upside_pct": upside, "rr": rr}


def run_scan(fetch_info=None) -> dict:
    """รัน 7-phase scan ทั้ง universe — คืน dict พร้อม regime + top picks + เหตุผล reject"""
    global _LAST
    t0 = time.time()
    result = {"asof": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "regime": {"status": "DATA UNAVAILABLE"}, "picks": [], "rejected": {},
              "scanned": 0, "paused": False, "took_s": 0}
    try:
        import yfinance as yf
        bench = yf.download(["SPY", "QQQ", "RSP", "^VIX"], period="6mo",
                            group_by="ticker", progress=False, threads=True)

        def _closes(sym):
            try:
                s = bench[sym]["Close"].dropna()
                return s if len(s) >= 22 else None
            except (KeyError, TypeError):
                return None

        spy_c, qqq_c, rsp_c = _closes("SPY"), _closes("QQQ"), _closes("RSP")
        vix_c = _closes("^VIX")
        if spy_c is None or qqq_c is None or rsp_c is None or vix_c is None:
            result["error"] = ("ดึงข้อมูล benchmark (SPY/QQQ/RSP/VIX) ไม่ได้ — "
                               "Yahoo อาจ rate-limit ชั่วคราว ลองใหม่ใน 1-2 นาที")
            result["took_s"] = round(time.time() - t0, 1)
            _LAST = result
            return result
        vix = float(vix_c.iloc[-1])

        spy_p, spy_e20 = float(spy_c.iloc[-1]), float(_ema(spy_c, 20).iloc[-1])
        qqq_p, qqq_e20 = float(qqq_c.iloc[-1]), float(_ema(qqq_c, 20).iloc[-1])
        spy20 = float(spy_c.iloc[-1] / spy_c.iloc[-21] - 1) * 100
        qqq20 = float(qqq_c.iloc[-1] / qqq_c.iloc[-21] - 1) * 100
        rsp20 = float(rsp_c.iloc[-1] / rsp_c.iloc[-21] - 1) * 100

        # Phase 2 — regime score
        score = 50
        score += 15 if spy_p > spy_e20 else -15
        score += 15 if qqq_p > qqq_e20 else -15
        score += 15 if vix < 15 else (8 if vix < 20 else (-10 if vix < 25 else -20))
        score += 5 if rsp20 > spy20 else -5
        score = max(0, min(100, score))
        cls = ("Strong Risk-On" if score >= 80 else "Risk-On" if score >= 60 else
               "Neutral" if score >= 40 else "Risk-Off" if score >= 20 else "Defensive")
        strict = (spy_p < spy_e20) or (qqq_p < qqq_e20) or (vix > 20)
        result["regime"] = {
            "status": "OK", "score": score, "classification": cls, "vix": round(vix, 2),
            "spy": round(spy_p, 2), "spy_e20": round(spy_e20, 2), "spy_above": spy_p > spy_e20,
            "qqq": round(qqq_p, 2), "qqq_e20": round(qqq_e20, 2), "qqq_above": qqq_p > qqq_e20,
            "breadth": "RSP นำ SPY (กว้าง)" if rsp20 > spy20 else "SPY นำ RSP (แคบ — mega-cap นำ)",
            "strict": strict,
        }
        if vix > 30:
            result["paused"] = True   # Phase 2: dislocation — หยุด scanner
            result["took_s"] = round(time.time() - t0, 1)
            _LAST = result
            return result

        # Phase 1+3 — ดึง universe แล้วคำนวณ
        tickers = [t for ts in SCAN_UNIVERSE.values() for t in ts]
        data = yf.download(tickers, period="6mo", group_by="ticker",
                           progress=False, threads=True)
        rejected: dict[str, int] = {}
        cands = []
        for t in tickers:
            try:
                df = data[t]
            except KeyError:
                rejected["no data"] = rejected.get("no data", 0) + 1
                continue
            m = _metrics(df, spy20, qqq20, spy_c)
            result["scanned"] += 1
            if m is None:
                rejected["no data"] = rejected.get("no data", 0) + 1
                continue
            # hard filters (Phase 3/4)
            if m["rsi"] > 80:
                rejected["RSI>80 parabolic"] = rejected.get("RSI>80 parabolic", 0) + 1
                continue
            if not m["above_all"]:
                rejected["ต่ำกว่า 10/20EMA หรือ 50SMA"] = rejected.get("ต่ำกว่า 10/20EMA หรือ 50SMA", 0) + 1
                continue
            if m["ext_pct"] > 10:
                rejected["EXTENDED >10% เหนือ 10EMA"] = rejected.get("EXTENDED >10% เหนือ 10EMA", 0) + 1
                continue
            if m["gain5d"] > 20:
                rejected["+20% ใน 5 วัน ยังไม่พัก"] = rejected.get("+20% ใน 5 วัน ยังไม่พัก", 0) + 1
                continue
            if strict and (m["vs_spy"] or 0) <= 0:
                rejected["regime เข้ม — RS ไม่ผ่าน"] = rejected.get("regime เข้ม — RS ไม่ผ่าน", 0) + 1
                continue
            theme = TICKER_THEME.get(t, "—")
            sc = _score(m, THEME_PRIORITY.index(theme) if theme in THEME_PRIORITY else 9)
            tr = _trade(m)
            if tr["rr"] is None or tr["rr"] < 3:
                rejected["R:R < 1:3"] = rejected.get("R:R < 1:3", 0) + 1
                continue
            if tr["upside_pct"] < 10:
                rejected["upside < 10%"] = rejected.get("upside < 10%", 0) + 1
                continue
            cands.append({"ticker": t, "theme": theme, "m": m, "score": sc, "trade": tr})

        cands.sort(key=lambda x: x["score"]["total"], reverse=True)

        # Phase 5 — earnings filter เฉพาะ top 8 (ประหยัด request)
        finals = []
        for c in cands[:8]:
            if fetch_info:
                info = fetch_info(c["ticker"])
                ed = info.get("earnings_date")
                c["earnings_date"] = ed or "DATA UNAVAILABLE"
                if ed:
                    try:
                        days = (datetime.strptime(ed, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                                - datetime.now(timezone.utc)).days
                        if 0 <= days <= 7:
                            rejected["earnings ภายใน 5 วันทำการ"] = rejected.get("earnings ภายใน 5 วันทำการ", 0) + 1
                            continue
                    except ValueError:
                        pass
            finals.append(c)
            if len(finals) == 5:
                break
        result["picks"] = finals
        result["rejected"] = rejected
    except Exception as e:
        result["error"] = (f"scan ไม่สำเร็จ ({type(e).__name__}) — Yahoo อาจ rate-limit "
                           "หรือ network ขัดข้อง ลองใหม่ใน 1-2 นาที")
    result["took_s"] = round(time.time() - t0, 1)
    _LAST = result
    return result


def last_scan() -> dict | None:
    return _LAST
