"""Task 5 - database trail. Works with SQLite (local) or PostgreSQL (Compose / cloud)
through one DATABASE_URL. Every prediction is stored: what was decided, when,
by which model, and how confidently."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    desc,
    insert,
    select,
    text,
)

metadata = MetaData()

predictions = Table(
    "predictions", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("filename", String(255), nullable=False),
    Column("image_sha256", String(64), nullable=False),
    Column("width", Integer),
    Column("height", Integer),
    Column("label", String(32), nullable=False),
    Column("prob_ai", Float, nullable=False),
    Column("confidence", Float, nullable=False),
    Column("threshold", Float, nullable=False),
    Column("model_name", String(100), nullable=False),
    Column("model_version", String(100), nullable=False),
    Column("backend", String(16), nullable=False),
    Column("latency_ms", Float),
    Column("client_ip", String(64)),
)


class Database:
    def __init__(self, url: str):
        self.engine = create_engine(url, pool_pre_ping=True, future=True)
        metadata.create_all(self.engine)

    def add(self, **row) -> int:
        row.setdefault("created_at", datetime.now(timezone.utc))
        with self.engine.begin() as conn:
            return conn.execute(insert(predictions).values(**row)).inserted_primary_key[0]

    def recent(self, limit: int = 100) -> list[dict]:
        with self.engine.connect() as conn:
            rows = conn.execute(select(predictions).order_by(desc(predictions.c.id)).limit(limit))
            return [dict(r._mapping) for r in rows]

    def ping(self) -> bool:
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception:  # noqa: BLE001
            return False
