"""Model artifact registry.

An artifact bundles four things that must travel together:

* the fitted estimator,
* the exact feature order it was trained on,
* the chosen operating threshold and its rationale,
* the training-time metrics plus the PSI baseline used for drift monitoring.

Persisting them separately is how a serving stack ends up feeding features in the
wrong order into a model whose threshold no longer applies, so they are written
and loaded as one unit.
"""
from __future__ import annotations

import json
import platform
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib

from app.core.config import settings

ARTIFACT_VERSION = "1.0.0"


@dataclass(slots=True)
class ModelArtifact:
    name: str
    version: str
    estimator: Any
    feature_names: list[str]
    threshold: float
    threshold_rationale: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    psi_baseline: dict[str, Any] = field(default_factory=dict)
    training_info: dict[str, Any] = field(default_factory=dict)
    # Optional preprocessing step (e.g. StandardScaler for Isolation Forest).
    scaler: Any = None
    trained_at: str = ""

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "n_features": self.n_features,
            "threshold": self.threshold,
            "threshold_rationale": self.threshold_rationale,
            "trained_at": self.trained_at,
            "metrics": self.metrics,
            "training_info": self.training_info,
        }


def _paths(name: str) -> tuple[Path, Path]:
    base = settings.artifact_path
    return base / f"{name}.joblib", base / f"{name}_metrics.json"


def save_artifact(artifact: ModelArtifact) -> tuple[Path, Path]:
    """Persist the bundle plus a human-readable metrics sidecar.

    The JSON sidecar exists so the admin dashboard and the README can read
    training metrics without unpickling (and therefore without importing) the
    estimator.
    """
    artifact.trained_at = artifact.trained_at or datetime.now(UTC).isoformat()
    artifact.training_info.setdefault("python", platform.python_version())
    artifact.training_info.setdefault("artifact_schema", ARTIFACT_VERSION)

    model_path, metrics_path = _paths(artifact.name)
    joblib.dump(
        {
            "name": artifact.name,
            "version": artifact.version,
            "estimator": artifact.estimator,
            "scaler": artifact.scaler,
            "feature_names": artifact.feature_names,
            "threshold": artifact.threshold,
            "threshold_rationale": artifact.threshold_rationale,
            "metrics": artifact.metrics,
            "psi_baseline": artifact.psi_baseline,
            "training_info": artifact.training_info,
            "trained_at": artifact.trained_at,
            "artifact_schema": ARTIFACT_VERSION,
        },
        model_path,
        compress=3,
    )
    metrics_path.write_text(
        json.dumps(
            {
                **artifact.summary(),
                "psi_baseline_bins": artifact.psi_baseline.get("edges", []),
                # Training share per bin; required for correct PSI on
                # zero-inflated score distributions.
                "psi_baseline_expected_pct": artifact.psi_baseline.get("expected_pct", []),
                "feature_names": artifact.feature_names,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return model_path, metrics_path


def load_artifact(name: str) -> ModelArtifact:
    model_path, _ = _paths(name)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact '{name}' not found at {model_path}. "
            "Run: python -m app.ml.train --all"
        )
    blob = joblib.load(model_path)
    return ModelArtifact(
        name=blob["name"],
        version=blob["version"],
        estimator=blob["estimator"],
        scaler=blob.get("scaler"),
        feature_names=blob["feature_names"],
        threshold=blob["threshold"],
        threshold_rationale=blob.get("threshold_rationale", ""),
        metrics=blob.get("metrics", {}),
        psi_baseline=blob.get("psi_baseline", {}),
        training_info=blob.get("training_info", {}),
        trained_at=blob.get("trained_at", ""),
    )


def artifact_exists(name: str) -> bool:
    return _paths(name)[0].exists()


def read_metrics(name: str) -> dict[str, Any] | None:
    _, metrics_path = _paths(name)
    if not metrics_path.exists():
        return None
    try:
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def list_artifacts() -> list[str]:
    base = settings.artifact_path
    return sorted(p.stem for p in base.glob("*.joblib"))
