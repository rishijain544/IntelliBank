"""SMOTE and class-imbalance utilities.

Implemented in-repo rather than depending on ``imbalanced-learn``, which lags
scikit-learn releases and would pin us to an older sklearn. This is a faithful
implementation of Chawla et al. (2002) plus a borderline variant.
"""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


def smote(
    X: np.ndarray,
    y: np.ndarray,
    *,
    k_neighbors: int = 5,
    target_ratio: float = 1.0,
    minority_label: int = 1,
    random_state: int | None = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic Minority Over-sampling Technique.

    New minority points are interpolated along the segment joining a minority
    sample and one of its k nearest minority neighbours::

        x_new = x_i + lambda * (x_zi - x_i),   lambda ~ U(0, 1)

    Parameters
    ----------
    target_ratio:
        Desired ``n_minority / n_majority`` after resampling. ``1.0`` fully
        balances the classes; ``0.3`` is often preferable for fraud because it
        preserves some of the true prior while still giving the learner signal.

    Returns
    -------
    ``(X_resampled, y_resampled)`` with the synthetic rows appended.
    """
    rng = np.random.default_rng(random_state)
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).ravel()

    minority_mask = y == minority_label
    X_min = X[minority_mask]
    n_min, n_maj = int(minority_mask.sum()), int((~minority_mask).sum())

    if n_min < 2 or n_maj == 0:
        return X, y

    n_needed = int(round(target_ratio * n_maj)) - n_min
    if n_needed <= 0:
        return X, y

    k = min(k_neighbors, n_min - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X_min)
    # Column 0 is the point itself; drop it.
    neighbours = nn.kneighbors(X_min, return_distance=False)[:, 1:]

    base_idx = rng.integers(0, n_min, size=n_needed)
    neighbour_pick = rng.integers(0, k, size=n_needed)
    lam = rng.random((n_needed, 1))

    origin = X_min[base_idx]
    target = X_min[neighbours[base_idx, neighbour_pick]]
    synthetic = origin + lam * (target - origin)

    X_out = np.vstack([X, synthetic])
    y_out = np.concatenate([y, np.full(n_needed, minority_label, dtype=y.dtype)])

    shuffle = rng.permutation(len(y_out))
    return X_out[shuffle], y_out[shuffle]


def borderline_smote(
    X: np.ndarray,
    y: np.ndarray,
    *,
    k_neighbors: int = 5,
    m_neighbors: int = 10,
    target_ratio: float = 1.0,
    minority_label: int = 1,
    random_state: int | None = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Borderline-SMOTE1: oversample only minority points near the decision boundary.

    A minority point is "in danger" when at least half—but not all—of its m
    nearest neighbours in the full dataset belong to the majority class. Those
    are the samples that actually shape the boundary; interior points add little.
    Falls back to plain SMOTE when no borderline set can be identified.
    """
    rng = np.random.default_rng(random_state)
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).ravel()

    minority_mask = y == minority_label
    X_min = X[minority_mask]
    n_min, n_maj = int(minority_mask.sum()), int((~minority_mask).sum())
    if n_min < 2 or n_maj == 0:
        return X, y

    m = min(m_neighbors, len(X) - 1)
    nn_all = NearestNeighbors(n_neighbors=m + 1).fit(X)
    neigh_all = nn_all.kneighbors(X_min, return_distance=False)[:, 1:]
    majority_counts = (y[neigh_all] != minority_label).sum(axis=1)
    in_danger = (majority_counts >= m / 2) & (majority_counts < m)

    if in_danger.sum() < 2:
        return smote(
            X, y,
            k_neighbors=k_neighbors,
            target_ratio=target_ratio,
            minority_label=minority_label,
            random_state=random_state,
        )

    X_danger = X_min[in_danger]
    n_needed = int(round(target_ratio * n_maj)) - n_min
    if n_needed <= 0:
        return X, y

    k = min(k_neighbors, n_min - 1)
    nn_min = NearestNeighbors(n_neighbors=k + 1).fit(X_min)
    neigh_min = nn_min.kneighbors(X_danger, return_distance=False)[:, 1:]

    base_idx = rng.integers(0, len(X_danger), size=n_needed)
    neighbour_pick = rng.integers(0, k, size=n_needed)
    lam = rng.random((n_needed, 1))

    origin = X_danger[base_idx]
    target = X_min[neigh_min[base_idx, neighbour_pick]]
    synthetic = origin + lam * (target - origin)

    X_out = np.vstack([X, synthetic])
    y_out = np.concatenate([y, np.full(n_needed, minority_label, dtype=y.dtype)])
    shuffle = rng.permutation(len(y_out))
    return X_out[shuffle], y_out[shuffle]


def scale_pos_weight(y: np.ndarray, minority_label: int = 1) -> float:
    """XGBoost's ``scale_pos_weight`` = n_negative / n_positive.

    An alternative to resampling: reweights the loss instead of inventing rows.
    The fraud trainer compares both strategies.
    """
    y = np.asarray(y).ravel()
    pos = int((y == minority_label).sum())
    neg = int((y != minority_label).sum())
    return float(neg / pos) if pos else 1.0


def imbalance_report(y: np.ndarray, minority_label: int = 1) -> dict[str, float | int]:
    y = np.asarray(y).ravel()
    pos = int((y == minority_label).sum())
    total = int(len(y))
    neg = total - pos
    return {
        "total": total,
        "positives": pos,
        "negatives": neg,
        "positive_rate": round(pos / total, 6) if total else 0.0,
        "imbalance_ratio": round(neg / pos, 2) if pos else float("inf"),
    }
