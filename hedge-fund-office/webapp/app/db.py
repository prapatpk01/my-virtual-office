"""
Database layer — SQLAlchemy 2.0.
Postgres via DATABASE_URL (Railway) หรือ fallback SQLite ในเครื่อง dev.
"""
import os
from datetime import datetime, timezone

from sqlalchemy import (
    create_engine, String, Float, Integer, DateTime, Text, func,
)
from sqlalchemy.orm import (
    DeclarativeBase, Mapped, mapped_column, sessionmaker,
)


def _database_url() -> str:
    url = os.environ.get("DATABASE_URL", "").strip()
    if not url:
        # dev fallback — ไฟล์ SQLite ข้าง ๆ
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return f"sqlite:///{os.path.join(here, 'fund.db')}"
    # Railway/Heroku ให้ 'postgres://' — SQLAlchemy ต้อง 'postgresql+psycopg2://'
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+psycopg2://", 1)
    elif url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg2://", 1)
    return url


DATABASE_URL = _database_url()
_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, echo=False, pool_pre_ping=True, connect_args=_connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Holding(Base):
    __tablename__ = "holdings"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticker: Mapped[str] = mapped_column(String(16), unique=True, index=True)
    shares: Mapped[float] = mapped_column(Float, default=0.0)
    cost_basis: Mapped[float] = mapped_column(Float, default=0.0)  # ต้นทุน/หุ้น
    sleeve: Mapped[str] = mapped_column(String(24), default="Growth")  # Growth/Income/Cash
    note: Mapped[str] = mapped_column(Text, default="")
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)


class Trade(Base):
    __tablename__ = "trades"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    action: Mapped[str] = mapped_column(String(8))  # BUY / SELL / TRIM
    shares: Mapped[float] = mapped_column(Float)
    price: Mapped[float] = mapped_column(Float)
    notes: Mapped[str] = mapped_column(Text, default="")
    created_by: Mapped[str] = mapped_column(String(48), default="team")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class TeamLog(Base):
    __tablename__ = "team_log"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    author: Mapped[str] = mapped_column(String(48), default="team")
    category: Mapped[str] = mapped_column(String(24), default="note")  # note/meeting/decision
    title: Mapped[str] = mapped_column(String(200))
    body: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Watch(Base):
    __tablename__ = "watchlist"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    theme: Mapped[str] = mapped_column(String(64), default="")
    catalyst: Mapped[str] = mapped_column(Text, default="")
    catalyst_date: Mapped[str] = mapped_column(String(24), default="")
    added_by: Mapped[str] = mapped_column(String(48), default="team")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class FundMeta(Base):
    """ค่า single-row สำหรับ peak NAV tracking + settings"""
    __tablename__ = "fund_meta"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    key: Mapped[str] = mapped_column(String(48), unique=True, index=True)
    value: Mapped[str] = mapped_column(Text, default="")
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)


def init_db() -> None:
    Base.metadata.create_all(engine)


def get_meta(db, key: str, default: str = "") -> str:
    row = db.query(FundMeta).filter_by(key=key).first()
    return row.value if row else default


def set_meta(db, key: str, value: str) -> None:
    row = db.query(FundMeta).filter_by(key=key).first()
    if row:
        row.value = value
    else:
        db.add(FundMeta(key=key, value=value))
    db.commit()
