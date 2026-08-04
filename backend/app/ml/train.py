"""Training CLI for all three models.

Usage::

    python -m app.ml.train --all                 # train everything (synthetic data)
    python -m app.ml.train --fraud --credit      # train a subset
    python -m app.ml.train --all --kaggle        # also use real Kaggle data where present
    python -m app.ml.train --all --fresh         # ignore cached datasets and regenerate

Generated datasets are cached as CSV under ``data/generated/`` so that retraining
after a hyper-parameter change does not pay the simulation cost again. Caches are
keyed by generator parameters, so changing a size or seed produces a new cache
file rather than silently reusing stale data.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from app.core.config import DATA_DIR, settings
from app.ml.datasets import (
    dataset_summary,
    generate_anomaly_dataset,
    generate_credit_dataset,
    generate_fraud_dataset,
    kaggle_available,
    kaggle_notes,
    load_kaggle_credit,
    load_kaggle_fraud,
)
from app.ml.registry import ModelArtifact, save_artifact
from app.ml.train_anomaly import train_anomaly_model
from app.ml.train_credit import train_credit_model
from app.ml.train_fraud import train_fraud_benchmark, train_fraud_model

CACHE_DIR = DATA_DIR / "generated"

# Defaults sized to produce enough minority-class rows for stable metrics while
# still training in a couple of minutes on a laptop.
DEFAULTS = {
    "fraud_users": 2200,
    "fraud_days": 180,
    "fraud_rate": 0.004,
    "credit_samples": 30_000,
    "anomaly_users": 1400,
    "anomaly_days": 150,
}


def _cached(name: str, builder, *, fresh: bool, verbose: bool = True) -> pd.DataFrame:
    """Return a generated dataset, using the CSV cache when it exists."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{name}.csv"
    if path.exists() and not fresh:
        df = pd.read_csv(path)
        if verbose:
            print(f"  cache hit: {path.name} ({len(df):,} rows)")
        return df

    t0 = time.perf_counter()
    df = builder()
    df.to_csv(path, index=False)
    if verbose:
        print(f"  generated in {time.perf_counter() - t0:.1f}s -> {path.name}")
    return df


def _banner(text: str) -> None:
    print(f"\n{'=' * 68}\n{text}\n{'=' * 68}")


def run_fraud(args) -> ModelArtifact:
    _banner("1/3  FRAUD DETECTION  (XGBoost, supervised, imbalanced)")
    key = f"fraud_u{args.fraud_users}_d{args.fraud_days}_r{args.fraud_rate}_s{args.seed}"
    df = _cached(
        key,
        lambda: generate_fraud_dataset(
            n_users=args.fraud_users,
            days=args.fraud_days,
            fraud_rate=args.fraud_rate,
            seed=args.seed,
        ),
        fresh=args.fresh,
    )
    print(f"  {dataset_summary(df, 'dataset')}")
    artifact = train_fraud_model(df, seed=args.seed)
    save_artifact(artifact)
    print(f"  saved artifact: {artifact.name} v{artifact.version}")

    if args.kaggle:
        if kaggle_available("fraud"):
            print("\n  --- Kaggle benchmark (PCA feature space, not served) ---")
            kdf, kcols = load_kaggle_fraud()
            print(f"  {dataset_summary(kdf, 'kaggle')}")
            bench = train_fraud_benchmark(kdf, kcols, seed=args.seed)
            save_artifact(bench)
            print(f"  saved artifact: {bench.name}")
        else:
            print("\n  Kaggle fraud dataset not found; skipping benchmark.")
    return artifact


def run_credit(args) -> ModelArtifact:
    _banner("2/3  CREDIT SCORING  (XGBoost + isotonic calibration, monotone)")
    if args.kaggle and kaggle_available("credit"):
        print("  using Kaggle German Credit data (mapped onto the deployed schema)")
        df = load_kaggle_credit()
        # German Credit has only 1000 rows, which is too thin for a stable
        # threshold; blend it with simulated applicants so both signals contribute.
        synth = _cached(
            f"credit_n{args.credit_samples}_s{args.seed}",
            lambda: generate_credit_dataset(n_samples=args.credit_samples, seed=args.seed),
            fresh=args.fresh,
        )
        df = pd.concat([df, synth], ignore_index=True)
        print(f"  blended with {len(synth):,} simulated applicants")
    else:
        if args.kaggle:
            print("  Kaggle German Credit not found; using simulated applicants.")
        df = _cached(
            f"credit_n{args.credit_samples}_s{args.seed}",
            lambda: generate_credit_dataset(n_samples=args.credit_samples, seed=args.seed),
            fresh=args.fresh,
        )
    print(f"  {dataset_summary(df, 'dataset')}")
    artifact = train_credit_model(df, seed=args.seed)
    save_artifact(artifact)
    print(f"  saved artifact: {artifact.name} v{artifact.version}")
    return artifact


def run_anomaly(args) -> ModelArtifact:
    _banner("3/3  ANOMALY DETECTION  (Isolation Forest, unsupervised)")
    key = f"anomaly_u{args.anomaly_users}_d{args.anomaly_days}_s{args.seed}"
    df = _cached(
        key,
        lambda: generate_anomaly_dataset(
            n_users=args.anomaly_users, days=args.anomaly_days, seed=args.seed + 1
        ),
        fresh=args.fresh,
    )
    print(f"  {dataset_summary(df, 'dataset')}")
    artifact = train_anomaly_model(df, seed=args.seed)
    save_artifact(artifact)
    print(f"  saved artifact: {artifact.name} v{artifact.version}")
    return artifact


def write_summary(artifacts: list[ModelArtifact]) -> Path:
    """Write the combined summary the admin dashboard and README both read."""
    out = {
        "generated_at": datetime.now(UTC).isoformat(),
        "models": {
            a.name: {
                "version": a.version,
                "n_features": a.n_features,
                "threshold": a.threshold,
                "threshold_rationale": a.threshold_rationale,
                "algorithm": a.training_info.get("algorithm"),
                "metrics": a.metrics.get("test", {}),
                "latency": a.metrics.get("latency", {}),
            }
            for a in artifacts
        },
    }
    path = settings.artifact_path / "training_summary.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return path


def print_headline(artifacts: list[ModelArtifact]) -> None:
    _banner("RESULTS")
    for a in artifacts:
        test = a.metrics.get("test", {})
        lat = a.metrics.get("latency", {})
        print(f"\n{a.name}  v{a.version}  ({a.training_info.get('algorithm')})")
        print(f"  features: {a.n_features}   threshold: {a.threshold:.4f}")
        if "roc_auc" in test and test.get("roc_auc") is not None:
            print(f"  ROC-AUC : {test['roc_auc']}")
        if test.get("pr_auc") is not None:
            print(f"  PR-AUC  : {test['pr_auc']}")
        if test.get("recall") is not None:
            print(f"  recall  : {test['recall']}    precision: {test.get('precision')}")
        if test.get("gini") is not None:
            print(f"  Gini    : {test['gini']}    KS: {test.get('ks_statistic')}")
        if test.get("ece") is not None:
            print(f"  ECE     : {test['ece']} (calibration error)")
        op = test.get("operating_point")
        if op:
            print(
                f"  operating point: precision={op['precision']} recall={op['recall']} "
                f"flag_rate={op['flag_rate']}"
            )
        if lat:
            print(f"  latency : p50={lat.get('p50_ms')}ms p95={lat.get('p95_ms')}ms p99={lat.get('p99_ms')}ms")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m app.ml.train",
        description="Train the IntelliBank ML models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=kaggle_notes(),
    )
    p.add_argument("--all", action="store_true", help="train all three models")
    p.add_argument("--fraud", action="store_true")
    p.add_argument("--credit", action="store_true")
    p.add_argument("--anomaly", action="store_true")
    p.add_argument("--kaggle", action="store_true", help="use real Kaggle datasets when present")
    p.add_argument("--fresh", action="store_true", help="regenerate datasets, ignoring the cache")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fraud-users", type=int, default=DEFAULTS["fraud_users"])
    p.add_argument("--fraud-days", type=int, default=DEFAULTS["fraud_days"])
    p.add_argument("--fraud-rate", type=float, default=DEFAULTS["fraud_rate"])
    p.add_argument("--credit-samples", type=int, default=DEFAULTS["credit_samples"])
    p.add_argument("--anomaly-users", type=int, default=DEFAULTS["anomaly_users"])
    p.add_argument("--anomaly-days", type=int, default=DEFAULTS["anomaly_days"])
    p.add_argument("--quick", action="store_true", help="small sizes for a fast smoke run")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.quick:
        args.fraud_users, args.fraud_days = 400, 120
        args.credit_samples = 6000
        args.anomaly_users, args.anomaly_days = 250, 100

    if args.all:
        args.fraud = args.credit = args.anomaly = True
    if not any((args.fraud, args.credit, args.anomaly)):
        build_parser().print_help()
        return 1

    print(f"artifact dir: {settings.artifact_path}")
    started = time.perf_counter()
    artifacts: list[ModelArtifact] = []

    if args.fraud:
        artifacts.append(run_fraud(args))
    if args.credit:
        artifacts.append(run_credit(args))
    if args.anomaly:
        artifacts.append(run_anomaly(args))

    print_headline(artifacts)
    summary = write_summary(artifacts)
    print(f"\ntotal time: {time.perf_counter() - started:.1f}s")
    print(f"summary: {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
