"""Evaluation metrics for the three models.

Two deliberate choices worth defending in a review:

* **Fraud is judged by recall at a precision floor, not accuracy.** With a 0.4%
  positive rate a model that predicts "never fraud" scores 99.6% accuracy and is
  worthless. The operating threshold is therefore chosen from the PR curve.
* **Drift is measured with PSI against the training score distribution.** Live
  labels arrive late (an analyst must review the alert), so score-distribution
  shift is the only signal available in near-real time.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _as1d(a) -> np.ndarray:
    return np.asarray(a).ravel()


# --------------------------------------------------------------------------- #
# Threshold selection
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class ThresholdChoice:
    threshold: float
    precision: float
    recall: float
    f1: float
    rationale: str


def threshold_for_min_precision(
    y_true, y_score, *, min_precision: float = 0.80
) -> ThresholdChoice:
    """Highest-recall threshold that still meets a precision floor.

    This is how a fraud team actually tunes a model: the review queue has finite
    analyst capacity, so precision is a hard operational constraint and recall is
    what you maximise underneath it.
    """
    y_true, y_score = _as1d(y_true), _as1d(y_score)
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    # precision/recall have one more element than thresholds
    best: ThresholdChoice | None = None
    for p, r, t in zip(precision[:-1], recall[:-1], thresholds):
        if p >= min_precision and (best is None or r > best.recall):
            best = ThresholdChoice(
                threshold=float(t),
                precision=float(p),
                recall=float(r),
                f1=float(2 * p * r / (p + r)) if (p + r) else 0.0,
                rationale=f"max recall subject to precision >= {min_precision:.2f}",
            )
    if best is not None:
        return best

    # Precision floor unreachable: fall back to best F1 and say so loudly.
    f1s = np.divide(
        2 * precision[:-1] * recall[:-1],
        precision[:-1] + recall[:-1],
        out=np.zeros_like(precision[:-1]),
        where=(precision[:-1] + recall[:-1]) > 0,
    )
    i = int(np.argmax(f1s)) if len(f1s) else 0
    t = float(thresholds[i]) if len(thresholds) else 0.5
    return ThresholdChoice(
        threshold=t,
        precision=float(precision[i]),
        recall=float(recall[i]),
        f1=float(f1s[i]) if len(f1s) else 0.0,
        rationale=(
            f"precision floor {min_precision:.2f} unreachable; fell back to best F1"
        ),
    )


def threshold_for_min_recall(
    y_true, y_score, *, min_recall: float = 0.90
) -> ThresholdChoice:
    """Highest-precision threshold that still catches ``min_recall`` of fraud.

    The mirror-image policy: used when the business mandates a catch rate and is
    willing to absorb the false-positive cost.
    """
    y_true, y_score = _as1d(y_true), _as1d(y_score)
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    best: ThresholdChoice | None = None
    for p, r, t in zip(precision[:-1], recall[:-1], thresholds):
        if r >= min_recall and (best is None or p > best.precision):
            best = ThresholdChoice(
                threshold=float(t),
                precision=float(p),
                recall=float(r),
                f1=float(2 * p * r / (p + r)) if (p + r) else 0.0,
                rationale=f"max precision subject to recall >= {min_recall:.2f}",
            )
    if best is not None:
        return best
    return threshold_for_min_precision(y_true, y_score, min_precision=0.5)


# --------------------------------------------------------------------------- #
# Classification report
# --------------------------------------------------------------------------- #


def classification_metrics(
    y_true, y_score, *, threshold: float = 0.5, positive_label: int = 1
) -> dict[str, Any]:
    y_true, y_score = _as1d(y_true), _as1d(y_score)
    y_pred = (y_score >= threshold).astype(int)

    n_pos = int((y_true == positive_label).sum())
    n_neg = int(len(y_true) - n_pos)

    out: dict[str, Any] = {
        "threshold": round(float(threshold), 6),
        "support": {"total": int(len(y_true)), "positives": n_pos, "negatives": n_neg},
        "positive_rate": round(n_pos / len(y_true), 6) if len(y_true) else 0.0,
    }

    # Ranking metrics are threshold-independent and the honest headline numbers.
    if n_pos and n_neg:
        out["roc_auc"] = round(float(roc_auc_score(y_true, y_score)), 4)
        out["pr_auc"] = round(float(average_precision_score(y_true, y_score)), 4)
        out["brier"] = round(float(brier_score_loss(y_true, y_score)), 6)
        # Gini/Somers' D — the coefficient credit risk teams actually quote.
        out["gini"] = round(2 * out["roc_auc"] - 1, 4)
        out["ks_statistic"] = round(ks_statistic(y_true, y_score), 4)
    else:
        out["roc_auc"] = out["pr_auc"] = out["brier"] = out["gini"] = None
        out["ks_statistic"] = None

    out["precision"] = round(float(precision_score(y_true, y_pred, zero_division=0)), 4)
    out["recall"] = round(float(recall_score(y_true, y_pred, zero_division=0)), 4)
    out["f1"] = round(float(f1_score(y_true, y_pred, zero_division=0)), 4)
    out["accuracy"] = round(float((y_pred == y_true).mean()), 4)

    if n_pos and n_neg:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        out["confusion_matrix"] = {
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
        }
        out["specificity"] = round(float(tn / (tn + fp)), 4) if (tn + fp) else 0.0
        out["false_positive_rate"] = round(float(fp / (tn + fp)), 4) if (tn + fp) else 0.0
        # Alert volume per 1000 transactions: the operational cost of the model.
        out["alerts_per_1000"] = round(float((tp + fp) / len(y_true) * 1000), 2)
    return out


def ks_statistic(y_true, y_score) -> float:
    """Kolmogorov-Smirnov separation between good and bad score distributions."""
    y_true, y_score = _as1d(y_true), _as1d(y_score)
    pos = np.sort(y_score[y_true == 1])
    neg = np.sort(y_score[y_true == 0])
    if not len(pos) or not len(neg):
        return 0.0
    grid = np.sort(np.unique(y_score))
    cdf_pos = np.searchsorted(pos, grid, side="right") / len(pos)
    cdf_neg = np.searchsorted(neg, grid, side="right") / len(neg)
    return float(np.max(np.abs(cdf_pos - cdf_neg)))


def recall_at_precision(y_true, y_score, min_precision: float) -> float:
    return threshold_for_min_precision(y_true, y_score, min_precision=min_precision).recall


def precision_at_k(y_true, y_score, k: int) -> float:
    """Precision within the top-k highest-risk cases.

    Directly answers "if an analyst reviews 100 alerts a day, how many are real?"
    """
    y_true, y_score = _as1d(y_true), _as1d(y_score)
    k = min(k, len(y_true))
    if k <= 0:
        return 0.0
    top = np.argsort(-y_score)[:k]
    return float(y_true[top].mean())


def lift_at_decile(y_true, y_score, decile: int = 1) -> float:
    """How many times better than random the top decile is."""
    y_true, y_score = _as1d(y_true), _as1d(y_score)
    base = y_true.mean()
    if base == 0:
        return 0.0
    n = max(int(len(y_true) * decile / 10), 1)
    top = np.argsort(-y_score)[:n]
    return round(float(y_true[top].mean() / base), 2)


# --------------------------------------------------------------------------- #
# Calibration — matters for credit, because the PD becomes an interest rate
# --------------------------------------------------------------------------- #


def calibration_curve_points(y_true, y_score, n_bins: int = 10) -> list[dict[str, float]]:
    """Reliability diagram data: predicted PD vs realised default rate per bin.

    A credit model whose ranking is good but whose probabilities are inflated
    would systematically overprice loans, so calibration is a first-class metric
    here rather than an afterthought.
    """
    y_true, y_score = _as1d(y_true), _as1d(y_score)
    edges = np.quantile(y_score, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        return []
    idx = np.clip(np.digitize(y_score, edges[1:-1], right=True), 0, len(edges) - 2)
    points: list[dict[str, float]] = []
    for b in range(len(edges) - 1):
        mask = idx == b
        if not mask.any():
            continue
        points.append(
            {
                "bin": b,
                "n": int(mask.sum()),
                "mean_predicted": round(float(y_score[mask].mean()), 4),
                "observed_rate": round(float(y_true[mask].mean()), 4),
            }
        )
    return points


def expected_calibration_error(y_true, y_score, n_bins: int = 10) -> float:
    pts = calibration_curve_points(y_true, y_score, n_bins)
    if not pts:
        return 0.0
    total = sum(p["n"] for p in pts)
    return round(
        sum(p["n"] / total * abs(p["mean_predicted"] - p["observed_rate"]) for p in pts), 4
    )


# --------------------------------------------------------------------------- #
# Drift — Population Stability Index
# --------------------------------------------------------------------------- #

PSI_BINS = 10


def population_stability_index(
    expected, actual, *, bins: int = PSI_BINS, edges: list[float] | None = None
) -> float:
    """PSI between a baseline (training) and a live distribution.

    Conventional reading: < 0.10 stable, 0.10-0.25 watch, > 0.25 drifting.
    """
    expected, actual = _as1d(expected), _as1d(actual)
    if len(expected) < 2 or len(actual) < 2:
        return 0.0

    if edges is None:
        cut = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
        if len(cut) < 3:
            return 0.0
        inner = cut[1:-1]
    else:
        inner = np.asarray(edges, dtype=float)

    e_counts = np.bincount(np.digitize(expected, inner, right=True), minlength=len(inner) + 1)
    a_counts = np.bincount(np.digitize(actual, inner, right=True), minlength=len(inner) + 1)

    # Laplace smoothing keeps empty live bins from producing infinite PSI.
    e_pct = (e_counts + 0.5) / (e_counts.sum() + 0.5 * len(e_counts))
    a_pct = (a_counts + 0.5) / (a_counts.sum() + 0.5 * len(a_counts))

    return round(float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct))), 4)


def psi_against_baseline(
    actual, edges: list[float], *, expected_pct: list[float] | None = None
) -> float:
    """PSI of a live score distribution against the training baseline.

    ``edges`` are the training-score bin boundaries persisted in the artifact.
    ``expected_pct`` is the training share per bin, which must be supplied
    whenever the edges are not exact deciles: for zero-inflated fraud scores the
    quantile cuts get deduplicated, so assuming a uniform 1/n share would report
    large drift on a perfectly stable model. It falls back to uniform only when
    the baseline shares were not recorded.
    """
    actual = _as1d(actual)
    if len(actual) < 2 or not edges:
        return 0.0

    inner = np.asarray(edges, dtype=float)
    n_bins = len(inner) + 1
    a_counts = np.bincount(np.digitize(actual, inner, right=True), minlength=n_bins)

    if expected_pct and len(expected_pct) == n_bins:
        e_pct = np.asarray(expected_pct, dtype=float)
        e_pct = e_pct / e_pct.sum()
        # Smooth so an empty training bin cannot produce an infinite ratio.
        e_pct = (e_pct + 1e-6) / (e_pct + 1e-6).sum()
    else:
        e_pct = np.full(n_bins, 1.0 / n_bins)

    a_pct = (a_counts + 0.5) / (a_counts.sum() + 0.5 * n_bins)
    return round(float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct))), 4)


def psi_baseline_profile(scores, bins: int = PSI_BINS) -> dict[str, list[float]]:
    """Bin edges plus the training share per bin, for drift monitoring."""
    scores = _as1d(scores)
    edges = psi_bin_edges(scores, bins)
    if not edges:
        return {"edges": [], "expected_pct": []}
    inner = np.asarray(edges, dtype=float)
    n_bins = len(inner) + 1
    counts = np.bincount(np.digitize(scores, inner, right=True), minlength=n_bins)
    total = max(int(counts.sum()), 1)
    return {
        "edges": [float(e) for e in inner],
        "expected_pct": [round(float(c) / total, 8) for c in counts],
    }


def drift_status(psi: float) -> str:
    if psi < 0.10:
        return "stable"
    if psi < 0.25:
        return "watch"
    return "drifting"


def psi_bin_edges(scores, bins: int = PSI_BINS) -> list[float]:
    """Persisted with the model so live PSI uses the training bin boundaries.

    Fraud scores are extremely zero-inflated (99%+ of transactions are obviously
    legitimate), so plain quantile cuts collapse into a handful of
    indistinguishable values near zero. ``np.unique`` then yields far fewer edges
    than requested and the "uniform per bin" assumption behind
    ``psi_against_baseline`` breaks, reporting drift where none exists.

    Deduplicating on rounded values and requiring strictly increasing edges keeps
    the bins meaningful; the returned count may legitimately be < ``bins``, and
    ``psi_against_baseline`` derives its expected shares from the actual length.
    """
    scores = _as1d(scores)
    if len(scores) < bins:
        return []
    raw = np.quantile(scores, np.linspace(0, 1, bins + 1))
    # 6dp avoids near-duplicate edges that would create empty, unstable bins.
    cut = np.unique(np.round(raw, 6))
    if len(cut) < 3:
        return []
    return [float(x) for x in cut[1:-1]]


# --------------------------------------------------------------------------- #
# Latency
# --------------------------------------------------------------------------- #


def latency_summary(samples_ms: list[float]) -> dict[str, float]:
    """p50/p95/p99 of inference latency. Means hide tail behaviour, and the
    resume claim is a *latency budget*, which is inherently a tail statement."""
    if not samples_ms:
        return {}
    arr = np.asarray(samples_ms, dtype=float)
    return {
        "n": int(arr.size),
        "mean_ms": round(float(arr.mean()), 3),
        "p50_ms": round(float(np.percentile(arr, 50)), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "p99_ms": round(float(np.percentile(arr, 99)), 3),
        "max_ms": round(float(arr.max()), 3),
    }


# --------------------------------------------------------------------------- #
# Unsupervised scoring
# --------------------------------------------------------------------------- #


def anomaly_metrics(y_true, y_score, *, contamination: float) -> dict[str, Any]:
    """Evaluate an unsupervised detector against held-out labels.

    The labels are never used for fitting; they exist only to prove the detector
    ranks genuine deviations above routine activity. Precision@k at the
    contamination rate is the metric that matches how the alert feed is consumed.
    """
    y_true, y_score = _as1d(y_true), _as1d(y_score)
    k = max(int(len(y_true) * contamination), 1)
    out: dict[str, Any] = {
        "support": {"total": int(len(y_true)), "positives": int(y_true.sum())},
        "contamination": contamination,
        f"precision_at_{int(contamination * 100)}pct": round(precision_at_k(y_true, y_score, k), 4),
        "lift_top_decile": lift_at_decile(y_true, y_score, 1),
    }
    if y_true.sum() and (len(y_true) - y_true.sum()):
        out["roc_auc"] = round(float(roc_auc_score(y_true, y_score)), 4)
        out["pr_auc"] = round(float(average_precision_score(y_true, y_score)), 4)
    return out
