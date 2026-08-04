"""Fraud detection trainer.

Design decisions worth defending:

* **Split by user, not by row.** A random row split would put a user's warm-up
  history in train and their fraud episode in test, leaking per-user behavioural
  baselines across the boundary and inflating every metric.
* **Imbalance strategy is chosen empirically, not assumed.** The trainer fits
  three candidates (class weighting, SMOTE, Borderline-SMOTE) and keeps the best
  by PR-AUC on a validation fold. Resampling is applied to *training folds only*;
  validation and test keep the true 0.4% prior, because a model tuned against a
  balanced validation set optimises a distribution that never occurs in production.
* **Threshold comes from the PR curve under a precision floor**, since analyst
  review capacity is the real constraint.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier

from app.ml.metrics import (
    classification_metrics,
    latency_summary,
    lift_at_decile,
    precision_at_k,
    psi_baseline_profile,
    threshold_for_min_precision,
)
from app.ml.registry import ModelArtifact
from app.ml.sampling import borderline_smote, imbalance_report, scale_pos_weight, smote
from app.ml.schema import FRAUD_FEATURES

MODEL_NAME = "fraud_xgb"
MODEL_VERSION = "1.0.0"

# Analyst review capacity is the binding constraint, so precision is a floor and
# recall is the objective underneath it.
MIN_PRECISION = 0.80

BASE_PARAMS: dict[str, Any] = {
    "n_estimators": 400,
    "max_depth": 5,
    "learning_rate": 0.06,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "min_child_weight": 2,
    "gamma": 0.4,
    "reg_lambda": 2.0,
    "reg_alpha": 0.3,
    "eval_metric": "aucpr",
    "tree_method": "hist",
    "n_jobs": -1,
    "random_state": 42,
}


@dataclass(slots=True)
class _Candidate:
    label: str
    pr_auc: float
    estimator: Any
    detail: dict[str, Any]


def _split_by_user(
    df: pd.DataFrame, *, test_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Group-aware split so no user appears on both sides of the boundary."""
    if "user_id" not in df.columns:
        rng = np.random.default_rng(seed)
        mask = rng.random(len(df)) >= test_size
        return df[mask], df[~mask]

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(df, df["label"], groups=df["user_id"]))
    return df.iloc[train_idx], df.iloc[test_idx]


def _fit_candidate(
    label: str,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> _Candidate:
    """Fit one imbalance strategy and score it on the untouched validation fold."""
    params = dict(BASE_PARAMS)
    detail: dict[str, Any] = {"strategy": label}

    if label == "class_weight":
        params["scale_pos_weight"] = scale_pos_weight(y_tr)
        Xf, yf = X_tr, y_tr
        detail["scale_pos_weight"] = round(params["scale_pos_weight"], 2)
    elif label == "smote":
        # 0.30 rather than full balance: over-synthesising a 0.4% class produces
        # mostly-interpolated training data and a badly calibrated model.
        Xf, yf = smote(X_tr, y_tr, target_ratio=0.30, k_neighbors=5, random_state=42)
        detail["target_ratio"] = 0.30
    elif label == "borderline_smote":
        Xf, yf = borderline_smote(X_tr, y_tr, target_ratio=0.30, random_state=42)
        detail["target_ratio"] = 0.30
    else:
        raise ValueError(label)

    detail["train_distribution"] = imbalance_report(yf)

    model = XGBClassifier(**params)
    model.fit(Xf, yf, eval_set=[(X_val, y_val)], verbose=False)

    val_scores = model.predict_proba(X_val)[:, 1]
    val_metrics = classification_metrics(y_val, val_scores, threshold=0.5)
    pr_auc = val_metrics.get("pr_auc") or 0.0
    detail["validation"] = {
        "pr_auc": pr_auc,
        "roc_auc": val_metrics.get("roc_auc"),
    }
    return _Candidate(label=label, pr_auc=float(pr_auc), estimator=model, detail=detail)


def train_fraud_model(
    df: pd.DataFrame,
    *,
    seed: int = 42,
    verbose: bool = True,
) -> ModelArtifact:
    """Train, select an imbalance strategy, tune the threshold and package the artifact."""
    feature_names = FRAUD_FEATURES.names
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise ValueError(f"dataset missing fraud features: {missing}")

    # Three-way split: train / validation (strategy + threshold) / test (reported once).
    train_val, test = _split_by_user(df, test_size=0.20, seed=seed)
    train, val = _split_by_user(train_val, test_size=0.25, seed=seed + 1)

    X_tr = train[feature_names].to_numpy(dtype=np.float32)
    y_tr = train["label"].to_numpy(dtype=int)
    X_val = val[feature_names].to_numpy(dtype=np.float32)
    y_val = val["label"].to_numpy(dtype=int)
    X_te = test[feature_names].to_numpy(dtype=np.float32)
    y_te = test["label"].to_numpy(dtype=int)

    if verbose:
        print(f"  train={len(y_tr):,} val={len(y_val):,} test={len(y_te):,}")
        print(f"  train imbalance: {imbalance_report(y_tr)}")

    if y_tr.sum() < 5 or y_val.sum() < 2 or y_te.sum() < 2:
        raise ValueError(
            f"too few fraud samples to train reliably "
            f"(train={y_tr.sum()}, val={y_val.sum()}, test={y_te.sum()}); "
            "increase n_users or fraud_rate"
        )

    # ---- compare imbalance strategies on the validation fold ----
    candidates = [
        _fit_candidate(name, X_tr, y_tr, X_val, y_val)
        for name in ("class_weight", "smote", "borderline_smote")
    ]
    for c in candidates:
        if verbose:
            print(f"    {c.label:18s} val PR-AUC={c.pr_auc:.4f}")

    best = max(candidates, key=lambda c: c.pr_auc)
    model = best.estimator
    if verbose:
        print(f"  selected strategy: {best.label}")

    # ---- threshold from the validation fold (never the test fold) ----
    val_scores = model.predict_proba(X_val)[:, 1]
    choice = threshold_for_min_precision(y_val, val_scores, min_precision=MIN_PRECISION)

    # ---- single held-out evaluation ----
    test_scores = model.predict_proba(X_te)[:, 1]
    test_metrics = classification_metrics(y_te, test_scores, threshold=choice.threshold)
    test_metrics["lift_top_decile"] = lift_at_decile(y_te, test_scores, 1)
    test_metrics["precision_at_100"] = round(precision_at_k(y_te, test_scores, 100), 4)

    # ---- latency: single-row inference, which is what production does ----
    samples: list[float] = []
    single = X_te[:1]
    for _ in range(5):  # warm up JIT/thread pools so p50 is not first-call noise
        model.predict_proba(single)
    for i in range(min(300, len(X_te))):
        row = X_te[i : i + 1]
        t0 = time.perf_counter()
        model.predict_proba(row)
        samples.append((time.perf_counter() - t0) * 1000)
    latency = latency_summary(samples)

    importances = sorted(
        (
            {"feature": f, "importance": round(float(v), 5)}
            for f, v in zip(feature_names, model.feature_importances_)
        ),
        key=lambda d: -d["importance"],
    )

    metrics = {
        "test": test_metrics,
        "validation_threshold": {
            "threshold": round(choice.threshold, 6),
            "precision": round(choice.precision, 4),
            "recall": round(choice.recall, 4),
            "rationale": choice.rationale,
        },
        "imbalance_strategy": best.detail,
        "strategy_comparison": [
            {"strategy": c.label, "val_pr_auc": round(c.pr_auc, 4)} for c in candidates
        ],
        "latency": latency,
        "top_features": importances[:12],
    }

    if verbose:
        print(
            f"  test: PR-AUC={test_metrics.get('pr_auc')} ROC-AUC={test_metrics.get('roc_auc')} "
            f"recall={test_metrics['recall']} precision={test_metrics['precision']}"
        )
        print(f"  latency p95={latency.get('p95_ms')}ms")

    return ModelArtifact(
        name=MODEL_NAME,
        version=MODEL_VERSION,
        estimator=model,
        feature_names=feature_names,
        threshold=float(choice.threshold),
        threshold_rationale=choice.rationale,
        metrics=metrics,
        psi_baseline={
            **psi_baseline_profile(test_scores),
            "mean_score": round(float(test_scores.mean()), 6),
            "p95_score": round(float(np.percentile(test_scores, 95)), 6),
        },
        training_info={
            "algorithm": "XGBClassifier",
            "params": {k: v for k, v in BASE_PARAMS.items() if k != "n_jobs"},
            "imbalance_strategy": best.label,
            "split": "GroupShuffleSplit by user_id (no user crosses the boundary)",
            "rows": {"train": int(len(y_tr)), "val": int(len(y_val)), "test": int(len(y_te))},
            "n_features": len(feature_names),
        },
    )


def train_fraud_benchmark(
    df: pd.DataFrame, feature_cols: list[str], *, seed: int = 42, verbose: bool = True
) -> ModelArtifact:
    """Benchmark model on the raw Kaggle PCA feature space.

    Kept separate from the served model: ``creditcard.csv`` ships anonymised
    components V1..V28 that cannot be mapped to named banking signals, so a model
    trained on them can report a headline number but cannot explain *why* a
    transaction was flagged — which the Fraud Center UI requires.
    """
    from sklearn.model_selection import train_test_split

    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=int)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tr, y_tr, test_size=0.25, stratify=y_tr, random_state=seed + 1
    )

    Xr, yr = smote(X_tr, y_tr, target_ratio=0.30, random_state=seed)
    params = dict(BASE_PARAMS)
    model = XGBClassifier(**params)
    model.fit(Xr, yr, eval_set=[(X_val, y_val)], verbose=False)

    val_scores = model.predict_proba(X_val)[:, 1]
    choice = threshold_for_min_precision(y_val, val_scores, min_precision=MIN_PRECISION)
    test_scores = model.predict_proba(X_te)[:, 1]
    test_metrics = classification_metrics(y_te, test_scores, threshold=choice.threshold)

    if verbose:
        print(
            f"  [kaggle benchmark] PR-AUC={test_metrics.get('pr_auc')} "
            f"recall={test_metrics['recall']} precision={test_metrics['precision']}"
        )

    return ModelArtifact(
        name="fraud_xgb_kaggle_benchmark",
        version=MODEL_VERSION,
        estimator=model,
        feature_names=feature_cols,
        threshold=float(choice.threshold),
        threshold_rationale=choice.rationale,
        metrics={"test": test_metrics, "imbalance_strategy": {"strategy": "smote", "target_ratio": 0.30}},
        psi_baseline=psi_baseline_profile(test_scores),
        training_info={
            "algorithm": "XGBClassifier",
            "dataset": "Kaggle Credit Card Fraud Detection (mlg-ulb)",
            "note": "PCA-anonymised features; benchmark only, not served by the API",
            "rows": {"train": int(len(y_tr)), "val": int(len(y_val)), "test": int(len(y_te))},
        },
    )
