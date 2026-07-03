# Sentinel Global Fund — Web App

FastAPI + Postgres (Railway) / SQLite (dev). Shared team password, no per-user accounts.

Dashboard (NAV, P&L, sleeve allocation, trim-zone alerts), holdings CRUD + trade log,
9-gate governance checklist, team log + watchlist. Live prices via yfinance
(cached 5 min); falls back to `DATA UNAVAILABLE` if a symbol can't be fetched — never
guesses a price (Governance Rule #5).

## Local development

```bash
cd hedge-fund-office/webapp
pip install -r requirements.txt
cp .env.example .env   # edit APP_PASSWORD + SECRET_KEY
export $(grep -v '^#' .env | xargs)   # or use a tool like direnv/python-dotenv
uvicorn app.main:app --reload --port 8000
```

Visit `http://localhost:8000`. No `DATABASE_URL` set → falls back to a local
`fund.db` SQLite file (gitignored).

## Deploy to Railway

1. **New Project → Deploy from GitHub repo**, point it at this repo.
2. **Root Directory**: set to `hedge-fund-office/webapp` (Railway service setting,
   not a file) — `railway.json` / `Procfile` / `requirements.txt` all live there.
3. **Add a Postgres plugin** to the project. Railway injects `DATABASE_URL`
   automatically — the app converts `postgres://` → `postgresql+psycopg2://` itself.
4. **Set environment variables** on the service:
   - `APP_PASSWORD` — shared team login password
   - `SECRET_KEY` — long random string (session cookie signing)
   - `API_KEY` — optional; enables key-based GET API for AI agents (unset = API disabled)
   - `FUND_NAME` — optional, defaults to "Sentinel Global Fund"
5. Deploy. Railway runs `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   (see `railway.json` / `Procfile`) and health-checks `/healthz`.

Tables are created automatically on startup (`db.init_db()` — `Base.metadata.create_all`).
No manual migration step for first deploy; schema changes later will need a
migration tool (Alembic) if the shape of existing tables changes.

## Notes

- yfinance requires open egress to Yahoo Finance — blocked in some sandboxed
  environments (e.g. this Claude Code web session), works normally on Railway.
- Auth is a single shared password (`APP_PASSWORD`), not per-user accounts —
  the display name typed at login is only used to attribute trade/log entries,
  it is not a security boundary.
- Peak NAV for drawdown tracking (`fund_meta.peak_nav`) is updated on every
  dashboard load when NAV exceeds the stored peak.
