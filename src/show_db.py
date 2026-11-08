"""Task 5 - query the predictions table directly (no Flask), for evidence.

    python -m src.show_db              # schema + last 10 rows
    python -m src.show_db --limit 25
Uses DATABASE_URL from .env (SQLite locally, Postgres under Compose).
"""
import argparse
import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, text

COLS = "id, created_at, filename, label, prob_ai, confidence, threshold, model_name, model_version, backend, latency_ms"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=10)
    a = ap.parse_args()

    load_dotenv()
    url = os.getenv("DATABASE_URL", "sqlite:///visionguard.db")
    engine = create_engine(url)
    print(f"Database: {engine.url.render_as_string(hide_password=True)}\n")

    print("Schema of table 'predictions':")
    for c in inspect(engine).get_columns("predictions"):
        print(f"  {c['name']:<14} {str(c['type']):<26} {'NULL' if c['nullable'] else 'NOT NULL'}")

    with engine.connect() as conn:
        total = conn.execute(text("SELECT COUNT(*) FROM predictions")).scalar()
        sql = f"SELECT {COLS} FROM predictions ORDER BY id DESC LIMIT :n"
        print(f"\n{sql.replace(':n', str(a.limit))};   -- {total} rows in total\n")
        rows = conn.execute(text(sql), {"n": a.limit}).all()

    headers = [h.strip() for h in COLS.split(",")]
    cells = [[str(v)[:19] if i == 1 else str(v) for i, v in enumerate(r)] for r in rows]
    widths = [max(len(h), *(len(c[i]) for c in cells)) if cells else len(h) for i, h in enumerate(headers)]
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("  ".join("-" * w for w in widths))
    for c in cells:
        print("  ".join(v.ljust(w) for v, w in zip(c, widths)))


if __name__ == "__main__":
    main()
