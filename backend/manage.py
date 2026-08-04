"""Developer convenience commands.

Usage::

    python manage.py train        # train all three ML models
    python manage.py seed         # seed demo data (preserves schema)
    python manage.py reset        # drop schema, recreate, seed
    python manage.py serve        # run the API with autoreload
    python manage.py test         # run the test suite
    python manage.py bootstrap    # train + reset + report (first-time setup)
    python manage.py check        # verify environment and artifacts
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

BACKEND = Path(__file__).parent


def _venv_python() -> str:
    """Resolve the project virtualenv's interpreter.

    `sys.executable` is deliberately not used as the default: running
    `python manage.py serve` with a global interpreter would inherit that
    interpreter, which does not have the project dependencies installed, and the
    server would fail with a confusing `ModuleNotFoundError: No module named
    'jwt'` from deep inside uvicorn's importer.

    Preferring the venv means the commands work whether or not the shell has the
    environment activated. If the venv is missing we fall back to the current
    interpreter and say so, rather than failing silently.
    """
    candidates = [
        BACKEND / ".venv" / "Scripts" / "python.exe",  # Windows
        BACKEND / ".venv" / "bin" / "python",          # macOS / Linux
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    print(
        "WARNING: no virtualenv found at backend/.venv — falling back to the\n"
        f"         current interpreter ({sys.executable}).\n"
        "         If imports fail, create it with:\n"
        "             python -m venv .venv\n"
        "             .venv\\Scripts\\pip install -r requirements.txt",
        file=sys.stderr,
    )
    return sys.executable


PY = _venv_python()


def _run(args: list[str]) -> int:
    print(f"$ {' '.join(args)}")
    return subprocess.call(args, cwd=BACKEND)


def train(argv: list[str]) -> int:
    return _run([PY, "-m", "app.ml.train", "--all", *argv])


def seed(argv: list[str]) -> int:
    return _run([PY, "-m", "app.seed", *argv])


def reset(argv: list[str]) -> int:
    return _run([PY, "-m", "app.seed", "--reset"])


def serve(argv: list[str]) -> int:
    port = argv[0] if argv else "8000"
    return _run(
        [PY, "-m", "uvicorn", "app.main:app", "--reload", "--host", "127.0.0.1", "--port", port]
    )


def test(argv: list[str]) -> int:
    return _run([PY, "-m", "pytest", "-q", "-p", "no:warnings", *argv])


def check(argv: list[str]) -> int:
    """Report on the environment: imports, DB, artifacts, model metrics.

    Dependency checks run in a subprocess using the venv interpreter, not this
    one. Importing them here would report on whichever Python launched the
    script, which is misleading when the venv is not activated.
    """
    print("=" * 62)
    print("IntelliBank environment check")
    print("=" * 62)
    print(f"launcher    : {sys.executable}")
    print(f"venv python : {PY}")
    print(f"using venv  : {PY != sys.executable}")

    probe = (
        "import sys;"
        "mods={};"
        "\nfor name, attr in ("
        "  ('fastapi','__version__'), ('pydantic','VERSION'),"
        "  ('sqlalchemy','__version__'), ('sklearn','__version__'),"
        "  ('xgboost','__version__'), ('jwt','__version__'),"
        "  ('bcrypt','__version__'), ('google.genai',None)):\n"
        "    try:\n"
        "        m=__import__(name, fromlist=['x'])\n"
        "        v=getattr(m, attr, 'ok') if attr else 'ok'\n"
        "        print(f'{name:14s}: {v}')\n"
        "    except ImportError as e:\n"
        "        print(f'{name:14s}: MISSING ({e})'); sys.exit(1)\n"
    )
    result = subprocess.run(
        [PY, "-c", probe], cwd=BACKEND, capture_output=True, text=True
    )
    print(result.stdout.rstrip())
    if result.returncode != 0:
        if result.stderr.strip():
            print(result.stderr.rstrip())
        print("\n  Install dependencies with:")
        print(f'    "{PY}" -m pip install -r requirements.txt')
        return 1

    from app.core.cache import get_kv
    from app.core.config import settings

    print(f"database    : {'sqlite' if settings.is_sqlite else 'postgresql'}")
    print(f"cache       : {get_kv().name}")
    if settings.JWT_SECRET.startswith("dev-only"):
        print("  WARNING: JWT_SECRET is still the insecure default")

    from app.ml.registry import list_artifacts, read_metrics

    artifacts = list_artifacts()
    print(f"artifacts   : {len(artifacts)} found in {settings.artifact_path}")
    if not artifacts:
        print("  no models trained yet -> python manage.py train")
        return 0

    for name in artifacts:
        meta = read_metrics(name) or {}
        # The sidecar nests held-out results under metrics.test; latency lives
        # under metrics.latency.
        metrics = meta.get("metrics", {})
        test_metrics = metrics.get("test", metrics)
        headline = []
        for key in ("roc_auc", "pr_auc", "recall", "precision", "gini", "ece"):
            if test_metrics.get(key) is not None:
                headline.append(f"{key}={test_metrics[key]}")
        lat = metrics.get("latency", {}) or meta.get("latency", {})
        if lat.get("p95_ms") is not None:
            headline.append(f"p95={lat['p95_ms']}ms")
        print(f"  {name:20s} {' '.join(headline) or 'no metrics'}")
    return 0


def bootstrap(argv: list[str]) -> int:
    """First-time setup: train models, then build a populated demo database."""
    if train([]) != 0:
        print("training failed")
        return 1
    if reset([]) != 0:
        print("seeding failed")
        return 1
    check([])
    print("\nready. start the API with:  python manage.py serve")
    return 0


COMMANDS = {
    "train": train,
    "seed": seed,
    "reset": reset,
    "serve": serve,
    "test": test,
    "check": check,
    "bootstrap": bootstrap,
}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        print("commands:", ", ".join(COMMANDS))
        return 1
    return COMMANDS[sys.argv[1]](sys.argv[2:])


if __name__ == "__main__":
    sys.exit(main())
