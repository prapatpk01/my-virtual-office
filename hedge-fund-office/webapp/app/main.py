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

from . import db, market, governance, auth

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
    prices = market.fetch_many([h.ticker for h in holdings] + ["SPY"]) if holdings else {"SPY": market.fetch("SPY")}
    pf = governance.build_portfolio(holdings, prices)

    peak = float(db.get_meta(sess, "peak_nav", "0") or 0)
    if pf["nav"] > peak:
        db.set_meta(sess, "peak_nav", str(pf["nav"]))
        peak = pf["nav"]
    dd = governance.drawdown_status(pf["nav"], peak)

    return templates.TemplateResponse(request, "dashboard.html", {
        "fund": FUND_NAME, "user": auth.current_user(request),
        "pf": pf, "dd": dd, "spy": prices.get("SPY", {}), "asof": market.now_iso(),
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


# ── JSON API ──────────────────────────────────────────────────────────────
@app.get("/api/portfolio")
def api_portfolio(request: Request, sess=Depends(get_db)):
    if not auth.current_user(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    holdings = sess.query(db.Holding).all()
    prices = market.fetch_many([h.ticker for h in holdings]) if holdings else {}
    return governance.build_portfolio(holdings, prices)


@app.get("/api/prices")
def api_prices(request: Request, symbols: str = "SPY"):
    if not auth.current_user(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    return {"fetched_at": market.now_iso(), "data": market.fetch_many(symbols.split(","))}
