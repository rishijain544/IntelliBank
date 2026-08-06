"""Pre-deployment checklist: verify configuration is deployment-ready.

Run this before pushing the "Deploy" button on Render or Vercel. It catches
problems that would otherwise cause a failed deploy or silent runtime breakage.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent
BACKEND = ROOT / "backend"
FRONTEND = ROOT / "frontend"

print("=" * 78)
print("PRE-DEPLOYMENT CHECKLIST")
print("=" * 78)

errors: list[str] = []
warnings: list[str] = []

# -------------------------------------------------------------------------
# 1. Required files exist
# -------------------------------------------------------------------------
required = {
    "render.yaml": ROOT / "render.yaml",
    "frontend/vercel.json": FRONTEND / "vercel.json",
    "DEPLOYMENT.md": ROOT / "DEPLOYMENT.md",
    "backend/requirements.txt": BACKEND / "requirements.txt",
    "backend/.env.example": BACKEND / ".env.example",
}

for name, path in required.items():
    if not path.exists():
        errors.append(f"MISSING: {name}")

# -------------------------------------------------------------------------
# 2. Secrets are NOT committed
# -------------------------------------------------------------------------
committed = (ROOT / ".git").exists()
if committed:
    import subprocess

    tracked = subprocess.run(
        ["git", "ls-tree", "-r", "HEAD", "--name-only"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.splitlines()
    # Exact match: backend/.env or frontend/.env, not .env.example variants
    for line in tracked:
        if line in ("backend/.env", "frontend/.env"):
            errors.append(f"{line} IS COMMITTED (contains secrets)")

# -------------------------------------------------------------------------
# 3. Deployment dependencies present
# -------------------------------------------------------------------------
reqs = (BACKEND / "requirements.txt").read_text(encoding="utf-8")
for dep in ["psycopg", "google-genai", "gunicorn"]:
    if dep not in reqs:
        errors.append(f"{dep} missing from requirements.txt")

# -------------------------------------------------------------------------
# 4. Frontend env var matches code
# -------------------------------------------------------------------------
api_ts = (FRONTEND / "src" / "lib" / "api.ts").read_text(encoding="utf-8")
if "VITE_API_URL" not in api_ts:
    errors.append("api.ts does not read VITE_API_URL")

env_prod = (FRONTEND / ".env.production").read_text(encoding="utf-8")
if "VITE_API_URL" not in env_prod:
    errors.append(".env.production does not define VITE_API_URL")
if "/api/v1" not in env_prod:
    warnings.append(".env.production may be missing the /api/v1 suffix")

# -------------------------------------------------------------------------
# 5. render.yaml structure
# -------------------------------------------------------------------------
render_yaml = (ROOT / "render.yaml").read_text(encoding="utf-8")
if "intellibank-api" not in render_yaml:
    errors.append("render.yaml missing service name intellibank-api")
if "intellibank-db" not in render_yaml:
    errors.append("render.yaml missing database intellibank-db")
if "JWT_SECRET" not in render_yaml:
    errors.append("render.yaml missing JWT_SECRET env var")
if "CORS_ORIGINS" not in render_yaml:
    errors.append("render.yaml missing CORS_ORIGINS env var")

# -------------------------------------------------------------------------
# 6. ML artifacts committed
# -------------------------------------------------------------------------
ml_dir = BACKEND / "ml_artifacts"
if not ml_dir.exists():
    errors.append("backend/ml_artifacts/ missing")
else:
    models = list(ml_dir.glob("*.joblib"))
    if len(models) != 3:
        errors.append(f"Expected 3 .joblib models, found {len(models)}")

# -------------------------------------------------------------------------
# 7. gitignore covers .env
# -------------------------------------------------------------------------
gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
if ".env" not in gitignore or "!.env.example" not in gitignore:
    errors.append(".gitignore does not properly exclude .env while allowing .env.example")

# -------------------------------------------------------------------------
# Report
# -------------------------------------------------------------------------
print()
if errors:
    print("ERRORS (must fix before deploying):")
    for e in errors:
        print(f"  ✗ {e}")
    print()

if warnings:
    print("WARNINGS (review recommended):")
    for w in warnings:
        print(f"  ⚠ {w}")
    print()

if not errors and not warnings:
    print("✓ All checks passed")
    print()
    print("Next steps:")
    print("  1. Push to GitHub: git push origin main")
    print("  2. Deploy backend: https://dashboard.render.com/blueprints")
    print("  3. Deploy frontend: https://vercel.com/new")
    print("  4. Follow DEPLOYMENT.md for env vars + CORS wiring")
    print()
    sys.exit(0)
elif errors:
    print(f"\n{len(errors)} ERROR(S) — deployment will fail")
    sys.exit(1)
else:
    print(f"\n{len(warnings)} WARNING(S) — review before deploying")
    sys.exit(0)
