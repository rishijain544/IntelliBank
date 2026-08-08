"""Seed the deployed database from your local machine.

Use this when the hosting platform gives you no shell (Render's free tier does
not). It connects to the remote Postgres directly and creates the demo accounts.

Usage
-----
1. Render dashboard -> intellibank-db -> Connections -> copy the
   **External Database URL** (it starts with postgres:// and contains
   .oregon-postgres.render.com). The Internal URL will NOT work from your
   machine.

2. Run, pasting the URL in quotes:

       .\\backend\\.venv\\Scripts\\python.exe seed_remote.py "postgres://user:pass@host/db"

What it does
------------
By default it is **additive**: it creates only the demo accounts, and leaves
every other user in the database alone. Re-running it refreshes the demo data
in place without touching real signups, because the underlying seed scopes its
deletes to the demo email addresses.

Flags
-----
--clean-test  also delete throwaway @example.com probe accounts left behind by
              deployment smoke tests. Never touches non-example.com addresses.
--force       DESTRUCTIVE. Drops and recreates every table, deleting all real
              accounts too. Requires typing a confirmation.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BACKEND = Path(__file__).parent / "backend"
if not BACKEND.exists():
    BACKEND = Path(__file__).parent

FLAGS = {"--force", "--clean-test"}
args = [a for a in sys.argv[1:] if a not in FLAGS]
force = "--force" in sys.argv
clean_test = "--clean-test" in sys.argv

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

from sqlalchemy import delete, func, select  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.core.database import SessionLocal, init_db  # noqa: E402
from app.models.system import AuditLog  # noqa: E402
from app.models.user import User  # noqa: E402
from app.seed import ADMIN_PASSWORD, DEMO_PASSWORD, DEMO_USERS  # noqa: E402

DEMO_EMAILS = [u["email"] for u in DEMO_USERS] + ["admin@intellibank.dev"]

# Throwaway accounts created by deployment smoke tests. Deliberately narrow:
# example.com is the RFC 2606 reserved domain, so this cannot match a real user.
TEST_PATTERN = re.compile(r"^(check|live|probe|smoke|test)[-\w]*@example\.com$", re.I)

host = settings.normalised_database_url().split("@")[-1].split("/")[0]
print("=" * 72)
print("SEEDING REMOTE DATABASE")
print("=" * 72)
print(f"  host: {host}")

print("\n  ensuring schema exists...")
init_db()

with SessionLocal() as db:
    all_emails = db.execute(select(User.email).order_by(User.email)).scalars().all()

present_demo = [e for e in all_emails if e in DEMO_EMAILS]
other = [e for e in all_emails if e not in DEMO_EMAILS]
test_accounts = [e for e in other if TEST_PATTERN.match(e)]
real_accounts = [e for e in other if e not in test_accounts]

print(f"  existing users: {len(all_emails)}")
print(f"    demo accounts present : {len(present_demo)}/{len(DEMO_EMAILS)}")
print(f"    other accounts        : {len(other)}")

if force:
    print("\n" + "!" * 72)
    print("  --force DROPS EVERY TABLE. These accounts will be permanently deleted:")
    for email in real_accounts:
        print(f"    - {email}")
    print("!" * 72)
    reply = input('\n  Type "DELETE EVERYTHING" to confirm: ').strip()
    if reply != "DELETE EVERYTHING":
        print("  Aborted - nothing was changed.")
        raise SystemExit(1)

# Optional cleanup of smoke-test leftovers so the admin panel looks clean.
if clean_test and test_accounts and not force:
    with SessionLocal() as db:
        rows = db.execute(select(User).where(User.email.in_(test_accounts))).scalars().all()
        for user in rows:
            db.execute(delete(AuditLog).where(AuditLog.actor_id == user.id))
            db.execute(delete(AuditLog).where(AuditLog.target_user_id == user.id))
            db.delete(user)
        db.commit()
    print(f"\n  removed {len(test_accounts)} throwaway test account(s):")
    for email in test_accounts:
        print(f"    - {email}")
elif test_accounts and not force:
    print(f"\n  note: {len(test_accounts)} throwaway test account(s) found.")
    print("        re-run with --clean-test to remove them.")

if real_accounts:
    print(f"\n  preserving {len(real_accounts)} real account(s):")
    for email in real_accounts:
        print(f"    - {email}")

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
    marker = "demo" if email in DEMO_EMAILS else "existing"
    print(f"  [{marker:8}] {email}")
print("\n  Sign in with:")
print(f"    priya@intellibank.dev / {DEMO_PASSWORD}     (customer)")
print(f"    admin@intellibank.dev / {ADMIN_PASSWORD}    (admin)")
