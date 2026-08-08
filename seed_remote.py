"""Seed the deployed database from your local machine.

Use this when the hosting platform gives you no shell (Render's free tier does
not). It connects to the remote Postgres directly and creates the demo accounts.

Usage
-----
1. Render dashboard -> intellibank-db -> Connections -> copy the
   **External Database URL** (it starts with postgres:// and ends with
   .oregon-postgres.render.com/...). The Internal URL will NOT work from your
   machine.

2. Run, pasting the URL in quotes:

       cd backend
       .venv\\Scripts\\python.exe ..\\seed_remote.py "postgres://user:pass@host/db"

The script is safe to re-run: it reports how many users already exist and does
nothing unless the database is empty, so it cannot overwrite real accounts.
Pass --force to reseed anyway.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BACKEND = Path(__file__).parent / "backend"
if not BACKEND.exists():
    BACKEND = Path(__file__).parent

args = [a for a in sys.argv[1:] if a != "--force"]
force = "--force" in sys.argv

if not args:
    print(__doc__)
    raise SystemExit(1)

database_url = args[0].strip().strip('"').strip("'")

if not database_url.startswith(("postgres://", "postgresql://", "postgresql+psycopg://")):
    print(f"ERROR: that does not look like a Postgres URL:\n  {database_url[:60]}...")
    raise SystemExit(1)

if "internal" in database_url or database_url.count("@") == 0:
    print("WARNING: this looks like an Internal URL, which is only reachable from")
    print("         inside Render. Use the External Database URL instead.\n")

# Point the app at the remote database before importing anything that reads
# settings, since configuration is resolved at import time.
os.environ["DATABASE_URL"] = database_url
os.environ["APP_ENV"] = "development"  # skip production guards for this CLI run

os.chdir(BACKEND)
sys.path.insert(0, str(BACKEND))

from sqlalchemy import func, select  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.core.database import SessionLocal, init_db  # noqa: E402
from app.models.user import User  # noqa: E402

host = settings.normalised_database_url().split("@")[-1].split("/")[0]
print("=" * 72)
print("SEEDING REMOTE DATABASE")
print("=" * 72)
print(f"  host: {host}")

print("\n  ensuring schema exists...")
init_db()

with SessionLocal() as db:
    existing = db.execute(select(func.count(User.id))).scalar_one()
print(f"  existing users: {existing}")

if existing and not force:
    print("\n  Database already has users - nothing to do.")
    print("  (Re-run with --force to wipe and reseed.)")
    with SessionLocal() as db:
        for email in db.execute(select(User.email).order_by(User.email)).scalars():
            print(f"    - {email}")
    raise SystemExit(0)

print("\n  seeding demo data (this scores transactions through the ML models)...")
from app.seed import seed  # noqa: E402

seed(reset=force)

with SessionLocal() as db:
    total = db.execute(select(func.count(User.id))).scalar_one()
    emails = db.execute(select(User.email).order_by(User.email)).scalars().all()

print("\n" + "=" * 72)
print(f"DONE - {total} users")
print("=" * 72)
for email in emails:
    print(f"  {email}")
print("\n  Sign in with:")
print("    priya@intellibank.dev / Demo@Pass123")
print("    admin@intellibank.dev / Admin@Pass123")
