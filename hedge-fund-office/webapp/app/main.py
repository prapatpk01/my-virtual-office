"""
Sentinel Global Fund — Web App
FastAPI + Postgres (Railway) / SQLite (dev). Shared-password team access.
Features: dashboard (prices/NAV/P&L/allocation), holdings CRUD + trade log,
governance gates, team log + watchlist.
"""
import os

from fastapi import FastAPI, Request, Form, Depends
from fastapi.responses import RedirectResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from . import db, market, governance, auth, scanner

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FUND_NAME = os.environ.get("FUND_NAME", "Sentinel Global Fund")

app = FastAPI(title=FUND_NAME)
app.add_middleware(SessionMiddleware, secret_key=os.environ.get("SECRET_KEY", "dev-secret-change-me"))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))


@app.on_event("startup")
def _startup():
    db.init_db()


def get_db():
    s = db.SessionLocal()
    try:
        yield s
    finally:
        s.close()


# ── Health / Auth ─────────────────────────────────────────────────────────
@app.get("/healthz")
def healthz():
    return {"ok": True, "fund": FUND_NAME}


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse(request, "login.html", {"fund": FUND_NAME, "err": None})


@app.post("/login")
def login(request: Request, name: str = Form(""), password: str = Form("")):
    if auth.check_password(password):
        request.session["user"] = (name or "team").strip()[:48]
        return RedirectResponse("/", status_code=302)
    return templates.TemplateResponse(
        request, "login.html", {"fund": FUND_NAME, "err": "รหัสผ่านไม่ถูกต้อง"}, status_code=401)


@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login", status_code=302)


# ── Dashboard ─────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request, sess=Depends(get_db)):
    if (r := auth.require_login(request)):
        return r
    holdings = sess.query(db.Holding).all()
    prices = market.fetch_many([h.ticker for h in holdings] + ["SPY", "^VIX"])
    pf = governance.build_portfolio(holdings, prices)

    vix = prices.get("^VIX", {})
    v = vix.get("price")
    vix_action = None if v is None else (
        "🔴 Complacent — trim satellite กำไร, ยก stop" if v < 15 else
        "🟡 Neutral — ถือ, deploy เฉพาะ high RS" if v < 20 else
        "⚠️ ดีดแรง — ดูทิศทางก่อน ห้าม FOMO" if v < 25 else
        "🟠 เริ่มมองหาการลงทุน — เตรียม limit" if v < 30 else
        "🟢 SPECIAL — เข้าซื้อเชิงรุก (ผ่าน guardrail ก่อน)")
    fg = market.fetch_fear_greed()

    peak = float(db.get_meta(sess, "peak_nav", "0") or 0)
    if pf["nav"] > peak:
        db.set_meta(sess, "peak_nav", str(pf["nav"]))
        peak = pf["nav"]
    dd = governance.drawdown_status(pf["nav"], peak)

    # บันทึก NAV เฉพาะวันที่ราคาสดครบทุกตัว — กันค่า fallback (ราคาทุน) ปนใน chart
    if pf["rows"] and all(r["price"] is not None for r in pf["rows"]):
        db.record_nav_snapshot(sess, pf["nav"], pf["book"])
    snaps = sess.query(db.NavSnapshot).order_by(db.NavSnapshot.day).all()
    nav_history = [{"day": s.day, "nav": s.nav, "book": s.book} for s in snaps]

    return templates.TemplateResponse(request, "dashboard.html", {
        "fund": FUND_NAME, "user": auth.current_user(request),
        "pf": pf, "dd": dd, "spy": prices.get("SPY", {}), "asof": market.now_iso(),
        "nav_history": nav_history,
        "vix": vix, "vix_action": vix_action, "fg": fg,
    })


# ── Holdings ──────────────────────────────────────────────────────────────
@app.get("/holdings", response_class=HTMLResponse)
def holdings_page(request: Request, sess=Depends(get_db)):
    if (r := auth.require_login(request)):
        return r
    holdings = sess.query(db.Holding).order_by(db.Holding.ticker).all()
    trades = sess.query(db.Trade).order_by(db.Trade.created_at.desc()).limit(30).all()
    return templates.TemplateResponse(request, "holdings.html", {
        "fund": FUND_NAME, "user": auth.current_user(request),
        "holdings": holdings, "trades": trades, "sleeves": ["Growth", "Income", "Cash"],
    })


@app.post("/holdings/save")
def holdings_save(request: Request, ticker: str = Form(...), shares: float = Form(...),
                  cost_basis: float = Form(...), sleeve: str = Form("Growth"),
                  note: str = Form(""), sess=Depends(get_db)):
    if (r := auth.require_login(request)):
        return r
    ticker = ticker.strip().upper()
    h = sess.query(db.Holding).filter_by(ticker=ticker).first()
    if h:
        h.shares, h.cost_basis, h.sleeve, h.note = shares, cost_basis, sleeve, note
    else:
        sess.add(db.Holding(ticker=ticker, shares=shares, cost_basis=cost_basis, sleeve=sleeve, note=note))
    sess.commit()
    return RedirectResponse("/holdings", status_code=302)


@app.post("/holdings/delete")
def holdings_delete(request: Request, ticker: str = Form(...), sess=Depends(get_db)):
    if (r := auth.require_login(request)):
        return r
    h = sess.query(db.Holding).filter_by(ticker=ticker.strip().upper()).first()
    if h:
        sess.delete(h)
        sess.commit()
    return RedirectResponse("/holdings", status_code=302)


@app.post("/trades/add")
def trade_add(request: Request, ticker: str = Form(...), action: str = Form(...),
              shares: float = Form(...), price: float = Form(...), notes: str = Form(""),
              apply_to_holding: str = Form(""), sess=Depends(get_db)):
    if (r := auth.require_login(request)):
        return r
    ticker = ticker.strip().upper()
    user = auth.current_user(request)
    sess.add(db.Trade(ticker=ticker, action=action.upper(), shares=shares, price=price,
                      notes=notes, created_by=user))
    # ปรับ holding อัตโนมัติถ้าเลือก
    if apply_to_holding:
        h = sess.query(db.Holding).filter_by(ticker=ticker).first()
        if h:
            if action.upper() == "BUY":
                total_cost = h.shares * h.cost_basis + shares * price
                h.shares += shares
                h.cost_basis = total_cost / h.shares if h.shares else price
            else:  # SELL / TRIM
                h.shares = max(0.0, h.shares - shares)
    sess.commit()
    return RedirectResponse("/holdings", status_code=302)


# ── Governance gates ──────────────────────────────────────────────────────
@app.get("/gates", response_class=HTMLResponse)
def gates_page(request: Request, sess=Depends(get_db)):
    if (r := auth.require_login(request)):
        return r
    holdings = sess.query(db.Holding).all()
    prices = market.fetch_many([h.ticker for h in holdings]) if holdings else {}
    pf = governance.build_portfolio(holdings, prices)
    peak = float(db.get_meta(sess, "peak_nav", "0") or 0)
    dd = governance.drawdown_status(pf["nav"], peak)
    return templates.TemplateResponse(request, "gates.html", {
        "fund": FUND_NAME, "user": auth.current_user(request),
        "gates": governance.GATES, "pf": pf, "dd": dd,
    })


# ── Team log + watchlist ──────────────────────────────────────────────────
@app.get("/team", response_class=HTMLResponse)
def team_page(request: Request, sess=Depends(get_db)):
    if (r := auth.require_login(request)):
        return r
    logs = sess.query(db.TeamLog).order_by(db.TeamLog.created_at.desc()).limit(50).all()
    watch = sess.query(db.Watch).order_by(db.Watch.created_at.desc()).all()
    return templates.TemplateResponse(request, "team.html", {
        "fund": FUND_NAME, "user": auth.current_user(request),
        "logs": logs, "watch": watch, "categories": ["note", "meeting", "decision"],
    })


@app.post("/team/log")
def team_log(request: Request, category: str = Form("note"), title: str = Form(...),
             body: str = Form(""), sess=Depends(get_db)):
    if (r := auth.require_login(request)):
        return r
    sess.add(db.TeamLog(author=auth.current_user(request), category=category, title=title, body=body))
    sess.commit()
    return RedirectResponse("/team", status_code=302)


@app.post("/team/watch")
def team_watch(request: Request, ticker: str = Form(...), theme: str = Form(""),
               catalyst: str = Form(""), catalyst_date: str = Form(""), sess=Depends(get_db)):
    if (r := auth.require_login(request)):
        return r
    sess.add(db.Watch(ticker=ticker.strip().upper(), theme=theme, catalyst=catalyst,
                      catalyst_date=catalyst_date, added_by=auth.current_user(request)))
    sess.commit()
    return RedirectResponse("/team", status_code=302)


@app.post("/team/watch/delete")
def team_watch_delete(request: Request, id: int = Form(...), sess=Depends(get_db)):
    if (r := auth.require_login(request)):
        return r
    w = sess.query(db.Watch).get(id)
    if w:
        sess.delete(w)
        sess.commit()
    return RedirectResponse("/team", status_code=302)


# ── Income / Dividends ────────────────────────────────────────────────────
@app.get("/income", response_class=HTMLResponse)
def income_page(request: Request, sess=Depends(get_db)):
    if (r := auth.require_login(request)):
        return r
    holdings = sess.query(db.Holding).all()
    prices = market.fetch_many([h.ticker for h in holdings]) if holdings else {}
    pf = governance.build_portfolio(holdings, prices)
    infos = market.fetch_info_many([h.ticker for h in holdings]) if holdings else {}
    inc = governance.income_summary(holdings, infos, pf["nav"])
    logs = sess.query(db.DividendLog).order_by(db.DividendLog.pay_date.desc()).limit(60).all()
    received = governance.dividends_by_month(logs)
    return templates.TemplateResponse(request, "income.html", {
        "fund": FUND_NAME, "user": auth.current_user(request),
        "inc": inc, "logs": logs, "received": received, "nav": pf["nav"],
    })


@app.post("/income/log")
def income_log(request: Request, ticker: str = Form(...), amount: float = Form(...),
               pay_date: str = Form(...), note: str = Form(""), sess=Depends(get_db)):
    if (r := auth.require_login(request)):
        return r
    sess.add(db.DividendLog(ticker=ticker.strip().upper(), amount=amount, pay_date=pay_date,
                            note=note, created_by=auth.current_user(request)))
    sess.commit()
    return RedirectResponse("/income", status_code=302)


@app.post("/income/log/delete")
def income_log_delete(request: Request, id: int = Form(...), sess=Depends(get_db)):
    if (r := auth.require_login(request)):
        return r
    row = sess.query(db.DividendLog).get(id)
    if row:
        sess.delete(row)
        sess.commit()
    return RedirectResponse("/income", status_code=302)


# ── News ──────────────────────────────────────────────────────────────────
@app.get("/news", response_class=HTMLResponse)
def news_page(request: Request, sess=Depends(get_db)):
    if (r := auth.require_login(request)):
        return r
    holdings = sess.query(db.Holding).all()
    watch = sess.query(db.Watch).all()
    symbols = list(dict.fromkeys([h.ticker.upper() for h in holdings] + [w.ticker.upper() for w in watch]))
    news = market.fetch_news(symbols) if symbols else []
    market_news = market.fetch_market_news()
    links = {s: market.research_links(s) for s in symbols}
    return templates.TemplateResponse(request, "news.html", {
        "fund": FUND_NAME, "user": auth.current_user(request),
        "news": news, "market_news": market_news, "links": links,
        "n_symbols": len(symbols), "asof": market.now_iso(),
    })


# ── Calendar ──────────────────────────────────────────────────────────────
@app.get("/calendar", response_class=HTMLResponse)
def calendar_page(request: Request, sess=Depends(get_db)):
    if (r := auth.require_login(request)):
        return r
    holdings = sess.query(db.Holding).all()
    watch = sess.query(db.Watch).all()
    symbols = [h.ticker for h in holdings] + [w.ticker for w in watch]
    events = market.build_calendar(symbols)
    return templates.TemplateResponse(request, "calendar.html", {
        "fund": FUND_NAME, "user": auth.current_user(request),
        "events": events, "asof": market.now_iso(),
    })


# ── Scanner (Momentum Swing Scanner v6.1) ─────────────────────────────────
@app.get("/scanner", response_class=HTMLResponse)
def scanner_page(request: Request):
    if (r := auth.require_login(request)):
        return r
    return templates.TemplateResponse(request, "scanner.html", {
        "fund": FUND_NAME, "user": auth.current_user(request),
        "res": scanner.last_scan(), "universe": scanner.SCAN_UNIVERSE,
    })


@app.post("/scanner/run")
def scanner_run(request: Request, sess=Depends(get_db)):
    if (r := auth.require_login(request)):
        return r
    res = scanner.run_scan(fetch_info=market.fetch_info)
    if res.get("picks"):
        top = ", ".join(f"{p['ticker']} ({p['score']['total']})" for p in res["picks"])
        sess.add(db.TeamLog(author="Maya Chen (Scanner)", category="note",
                            title=f"Momentum scan — top {len(res['picks'])}",
                            body=f"Regime {res['regime'].get('score')}/100 {res['regime'].get('classification')} · {top}"))
        sess.commit()
    return RedirectResponse("/scanner", status_code=302)


# ── JSON API ──────────────────────────────────────────────────────────────
# รองรับ 2 แบบ: session (login แล้ว) หรือ ?key=API_KEY (AI team — GET เท่านั้น)
def _api_ok(request: Request, key: str | None) -> bool:
    return bool(auth.current_user(request)) or auth.check_api_key(key)


def _deny():
    return JSONResponse({"error": "unauthorized — ใส่ ?key=API_KEY (ตั้งค่า API_KEY ใน Railway Variables ก่อน)"},
                        status_code=401)


@app.get("/api/portfolio")
def api_portfolio(request: Request, key: str | None = None, sess=Depends(get_db)):
    if not _api_ok(request, key):
        return _deny()
    holdings = sess.query(db.Holding).all()
    prices = market.fetch_many([h.ticker for h in holdings]) if holdings else {}
    pf = governance.build_portfolio(holdings, prices)
    pf["fetched_at"] = market.now_iso()
    return pf


@app.get("/api/prices")
def api_prices(request: Request, symbols: str = "SPY", key: str | None = None):
    if not _api_ok(request, key):
        return _deny()
    return {"fetched_at": market.now_iso(), "data": market.fetch_many(symbols.split(","))}


@app.get("/api/summary")
def api_summary(request: Request, key: str | None = None, sess=Depends(get_db)):
    """ภาพรวมกองทุนใน 1 call — NAV, sentiment, alerts, drawdown (เหมาะกับ AI team)"""
    if not _api_ok(request, key):
        return _deny()
    holdings = sess.query(db.Holding).all()
    prices = market.fetch_many([h.ticker for h in holdings] + ["SPY", "^VIX"])
    pf = governance.build_portfolio(holdings, prices)
    peak = float(db.get_meta(sess, "peak_nav", "0") or 0)
    return {
        "fetched_at": market.now_iso(),
        "nav": pf["nav"], "book": pf["book"], "pl": pf["pl"], "pl_pct": pf["pl_pct"],
        "positions": len(pf["rows"]), "sleeve": pf["sleeve"], "alerts": pf["alerts"],
        "drawdown": governance.drawdown_status(pf["nav"], peak),
        "spy": prices.get("SPY"), "vix": prices.get("^VIX"),
        "fear_greed": market.fetch_fear_greed(),
        "holdings": [{"ticker": r["ticker"], "sleeve": r["sleeve"], "price": r["price"],
                      "value": r["value"], "pct_nav": r["pct"], "pl_pct": r["pl_pct"],
                      "zone": r["zone"], "rsi14": r["rsi14"], "macd": r["macd"]} for r in pf["rows"]],
    }


@app.get("/api/income")
def api_income(request: Request, key: str | None = None, sess=Depends(get_db)):
    if not _api_ok(request, key):
        return _deny()
    holdings = sess.query(db.Holding).all()
    prices = market.fetch_many([h.ticker for h in holdings]) if holdings else {}
    pf = governance.build_portfolio(holdings, prices)
    infos = market.fetch_info_many([h.ticker for h in holdings]) if holdings else {}
    logs = sess.query(db.DividendLog).order_by(db.DividendLog.pay_date.desc()).limit(120).all()
    return {"fetched_at": market.now_iso(),
            "estimate": governance.income_summary(holdings, infos, pf["nav"]),
            "received": governance.dividends_by_month(logs)}


@app.get("/api/calendar")
def api_calendar(request: Request, key: str | None = None, sess=Depends(get_db)):
    if not _api_ok(request, key):
        return _deny()
    holdings = sess.query(db.Holding).all()
    watch = sess.query(db.Watch).all()
    symbols = [h.ticker for h in holdings] + [w.ticker for w in watch]
    return {"fetched_at": market.now_iso(), "events": market.build_calendar(symbols)}


@app.get("/api/news")
def api_news(request: Request, key: str | None = None, sess=Depends(get_db)):
    if not _api_ok(request, key):
        return _deny()
    holdings = sess.query(db.Holding).all()
    watch = sess.query(db.Watch).all()
    symbols = [h.ticker for h in holdings] + [w.ticker for w in watch]
    return {"fetched_at": market.now_iso(),
            "market_news": market.fetch_market_news(),
            "ticker_news": market.fetch_news(symbols) if symbols else []}


@app.get("/api/scanner")
def api_scanner(request: Request, key: str | None = None, run: int = 0, sess=Depends(get_db)):
    """ผล scan ล่าสุด — ?run=1 เพื่อรันใหม่ (ใช้เวลา 20-40 วิ)"""
    if not _api_ok(request, key):
        return _deny()
    if run:
        res = scanner.run_scan(fetch_info=market.fetch_info)
        if res.get("picks"):
            top = ", ".join(f"{p['ticker']} ({p['score']['total']})" for p in res["picks"])
            sess.add(db.TeamLog(author="Maya Chen (Scanner)", category="note",
                                title=f"Momentum scan (API) — top {len(res['picks'])}",
                                body=f"Regime {res['regime'].get('score')}/100 · {top}"))
            sess.commit()
        return res
    return scanner.last_scan() or {"note": "ยังไม่เคยรัน — เรียก /api/scanner?run=1&key=..."}


@app.get("/api/log")
def api_log(request: Request, title: str = "", body: str = "", author: str = "AI Team",
            category: str = "note", key: str | None = None, sess=Depends(get_db)):
    """ให้ AI team เขียน team log ผ่าน GET (ข้อจำกัด: web_fetch ส่ง POST ไม่ได้)"""
    if not _api_ok(request, key):
        return _deny()
    if not title:
        return JSONResponse({"error": "ต้องมี ?title="}, status_code=400)
    sess.add(db.TeamLog(author=author[:48], category=category[:24], title=title[:200], body=body))
    sess.commit()
    return {"ok": True, "logged": title[:200]}


@app.get("/api/help")
def api_help(request: Request, key: str | None = None):
    """รายการ endpoints ทั้งหมด (ต้องมี key เพื่อยืนยันว่าตั้งค่าถูก)"""
    if not _api_ok(request, key):
        return _deny()
    return {"endpoints": {
        "/api/summary":   "ภาพรวม: NAV, P&L, sleeve, alerts, VIX, F&G, holdings",
        "/api/portfolio": "portfolio เต็ม (rows + zones + trim calc)",
        "/api/prices?symbols=NVDA,SPY": "ราคา + RSI + MACD ต่อ symbol",
        "/api/income":    "ประมาณการปันผล + ที่รับจริงรายเดือน/ปี",
        "/api/calendar":  "FOMC/CPI/NFP + earnings + XD",
        "/api/news":      "ข่าวตลาดหลายสำนัก + ข่าวรายตัว",
        "/api/scanner":   "ผล momentum scan ล่าสุด (?run=1 รันใหม่ ~30วิ)",
        "/api/log?title=..&body=..&author=..": "เขียน team log ผ่าน GET",
    }, "auth": "ทุก endpoint ต้องมี ?key=API_KEY"}
