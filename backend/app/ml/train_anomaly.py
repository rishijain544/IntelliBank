"""Anomaly detection trainer (unsupervised).

Framing that keeps this distinct from the fraud model: anomalies are lifestyle
deviations — "this doesn't look like you" — not theft. No money is ever blocked
on an anomaly score; it only produces an Insights nudge.

Design decisions:

* **Fitted on unlabelled data.** Labels exist in the simulated dataset purely to
  evaluate ranking quality. Feeding them to the estimator would make this a
  supervised model wearing an unsupervised label.
* **Split by user.** Same leakage argument as fraud: per-user baselines must not
  straddle the train/test boundary.
* **Scaling is mandatory.** Isolation Forest splits on raw axis values, so an
  unscaled ``amount_log`` (0-20) would dominate ``is_weekend`` (0-1). A scaler is
  persisted inside the artifact so serving applies the identical transform.
* **Scores are mapped to [0,1] via the training distribution.** ``score_samples``
  returns an unbounded, sign-flipped quantity that is meaningless in a UI, so the
  artifact stores the percentile mapping needed to express "top 3% most unusual".
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from app.ml.metrics import (
    anomaly_metrics,
    latency_summary,
    lift_at_decile,
    precision_at_k,
    psi_baseline_profile,
)
from app.ml.registry import ModelArtifact
from app.ml.schema import ANOMALY_FEATURES

MODEL_NAME = "anomaly_iforest"
MODEL_VERSION = "1.0.0"

CONTAMINATION = 0.03

BASE_PARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "max_samples": 512,
    "contamination": CONTAMINATION,
    "max_features": 0.85,
    "bootstrap": False,
    "n_jobs": -1,
    "random_state": 42,
}


def _split_by_user(df: pd.DataFrame, *, test_size: float, seed: int):
    if "user_id" not in df.columns:
        rng = np.random.default_rng(seed)
        mask = rng.random(len(df)) >= test_size
        return df[mask], df[~mask]
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(df, groups=df["user_id"]))
    return df.iloc[train_idx], df.iloc[test_idx]


def normalise_scores(raw: np.ndarray, ref_min: float, ref_max: float) -> np.ndarray:
    """Map Isolation Forest ``score_samples`` output to a 0-1 "unusualness" scale.

    ``score_samples`` is higher for *normal* points, so the sign is flipped:
    0 = perfectly typical, 1 = most unusual seen during training.
    """
    span = ref_max - ref_min
    if span <= 0:
        return np.full_like(raw, 0.5, dtype=float)
    return np.clip((ref_max - raw) / span, 0.0, 1.0)


def train_anomaly_model(
    df: pd.DataFrame, *, seed: int = 42, verbose: bool = True
) -> ModelArtifact:
    feature_names = ANOMALY_FEATURES.names
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise ValueError(f"dataset missing anomaly features: {missing}")

    train, test = _split_by_user(df, test_size=0.25, seed=seed)

    X_tr = train[feature_names].to_numpy(dtype=np.float64)
    X_te = test[feature_names].to_numpy(dtype=np.float64)
    y_te = test["label"].to_numpy(dtype=int)

    if verbose:
        print(f"  train={len(X_tr):,} (unlabelled) test={len(X_te):,}")

    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_te_s = scaler.transform(X_te)

    model = IsolationForest(**BASE_PARAMS)
    model.fit(X_tr_s)  # labels intentionally never passed

    # Reference range from the training distribution drives the 0-1 mapping.
    train_raw = model.score_samples(X_tr_s)
    ref_min, ref_max = float(train_raw.min()), float(train_raw.max())

    test_raw = model.score_samples(X_te_s)
    test_scores = normalise_scores(test_raw, ref_min, ref_max)

    metrics_out = anomaly_metrics(y_te, test_scores, contamination=CONTAMINATION)
    metrics_out["lift_top_decile"] = lift_at_decile(y_te, test_scores, 1)
    metrics_out["precision_at_50"] = round(precision_at_k(y_te, test_scores, 50), 4)

    # Alert threshold: the score at the (1 - contamination) training percentile.
    # Anchoring on training data keeps the alert *rate* stable in production
    # instead of letting a quiet week flood users with nudges.
    train_scores = normalise_scores(train_raw, ref_min, ref_max)
    alert_threshold = float(np.quantile(train_scores, 1 - CONTAMINATION))

    flagged = test_scores >= alert_threshold
    metrics_out["operating_point"] = {
        "threshold": round(alert_threshold, 6),
        "flag_rate": round(float(flagged.mean()), 5),
        "precision": round(float(y_te[flagged].mean()), 4) if flagged.any() else 0.0,
        "recall": round(float(flagged[y_te == 1].mean()), 4) if (y_te == 1).any() else 0.0,
    }

    samples: list[float] = []
    for _ in range(5):
        model.score_samples(scaler.transform(X_te[:1]))
    for i in range(min(300, len(X_te))):
        row = X_te[i : i + 1]
        t0 = time.perf_counter()
        model.score_samples(scaler.transform(row))
        samples.append((time.perf_counter() - t0) * 1000)
    latency = latency_summary(samples)

    if verbose:
        print(
            f"  test: ROC-AUC={metrics_out.get('roc_auc')} PR-AUC={metrics_out.get('pr_auc')} "
            f"lift={metrics_out['lift_top_decile']}x"
        )
        print(
            f"  operating point: precision={metrics_out['operating_point']['precision']} "
            f"recall={metrics_out['operating_point']['recall']} "
            f"flag_rate={metrics_out['operating_point']['flag_rate']}"
        )
        print(f"  latency p95={latency.get('p95_ms')}ms")

    return ModelArtifact(
        name=MODEL_NAME,
        version=MODEL_VERSION,
        estimator=model,
        scaler=scaler,
        feature_names=feature_names,
        threshold=alert_threshold,
        threshold_rationale=(
            f"score at the {(1 - CONTAMINATION):.0%} percentile of the training "
            "distribution, which holds the production alert rate near "
            f"{CONTAMINATION:.0%}"
        ),
        metrics={
            "test": metrics_out,
            "latency": latency,
            "score_normalisation": {"ref_min": ref_min, "ref_max": ref_max},
        },
        psi_baseline={
            **psi_baseline_profile(train_scores),
            "mean_score": round(float(train_scores.mean()), 6),
            "p95_score": round(float(np.percentile(train_scores, 95)), 6),
        },
        training_info={
            "algorithm": "IsolationForest",
            "params": {k: v for k, v in BASE_PARAMS.items() if k != "n_jobs"},
            "supervision": "unsupervised; labels used only for evaluation",
            "split": "GroupShuffleSplit by user_id",
            "score_normalisation": {"ref_min": ref_min, "ref_max": ref_max},
            "rows": {"train": int(len(X_tr)), "test": int(len(X_te))},
            "n_features": len(feature_names),
        },
    )
