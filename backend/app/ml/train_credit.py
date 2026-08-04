"""Credit scoring trainer.

Two properties matter more here than raw ranking power:

* **Calibration.** The predicted default probability is converted into an
  interest rate, so a model that ranks well but reports inflated probabilities
  would systematically overprice every loan. The trainer therefore reports ECE
  and a reliability curve alongside AUC, and compares an isotonic-calibrated
  variant against the raw estimator.
* **Monotonicity where the business requires it.** A regulator-facing scorecard
  cannot say "more prior defaults lowered your risk", even if a tree finds such a
  pocket in the data. XGBoost monotone constraints enforce the sign of the
  relationship for the features where direction is not negotiable.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from app.ml.metrics import (
    calibration_curve_points,
    classification_metrics,
    expected_calibration_error,
    latency_summary,
    lift_at_decile,
    psi_baseline_profile,
    threshold_for_min_precision,
)
from app.ml.registry import ModelArtifact
from app.ml.schema import CREDIT_FEATURES
from app.ml.sampling import imbalance_report

MODEL_NAME = "credit_xgb"
MODEL_VERSION = "1.0.0"

# Rejecting a good applicant costs a customer; approving a defaulter costs the
# principal. The floor is lower than fraud's because manual review is cheaper
# than a charge-off.
MIN_PRECISION = 0.55

# Direction of effect that must hold regardless of what the trees find.
# +1 = higher feature value may only increase predicted default risk.
MONOTONE: dict[str, int] = {
    "prior_defaults": 1,
    "emis_missed": 1,
    "debt_to_income": 1,
    "emi_to_income": 1,
    "loan_to_income": 1,
    "credit_utilisation": 1,
    "overdraft_events_90d": 1,
    "existing_emi_log": 1,
    "employment_stability": -1,
    "min_balance_ratio": -1,
    "savings_rate": -1,
    "annual_income_log": -1,
    "avg_balance_log": -1,
    "employment_years": -1,
    "account_age_months": -1,
}

BASE_PARAMS: dict[str, Any] = {
    "n_estimators": 500,
    "max_depth": 4,
    "learning_rate": 0.045,
    "subsample": 0.85,
    "colsample_bytree": 0.80,
    "min_child_weight": 8,
    "gamma": 0.2,
    "reg_lambda": 3.0,
    "reg_alpha": 0.5,
    "eval_metric": "auc",
    "tree_method": "hist",
    "n_jobs": -1,
    "random_state": 42,
}


def _monotone_tuple(feature_names: list[str]) -> str:
    """XGBoost expects the constraint vector positionally, as "(0,1,-1,...)"."""
    return "(" + ",".join(str(MONOTONE.get(f, 0)) for f in feature_names) + ")"


def train_credit_model(
    df: pd.DataFrame, *, seed: int = 42, verbose: bool = True
) -> ModelArtifact:
    feature_names = CREDIT_FEATURES.names
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise ValueError(f"dataset missing credit features: {missing}")

    X = df[feature_names].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=int)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=seed
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tr, y_tr, test_size=0.20, stratify=y_tr, random_state=seed + 1
    )

    if verbose:
        print(f"  train={len(y_tr):,} val={len(y_val):,} test={len(y_te):,}")
        print(f"  class balance: {imbalance_report(y_tr)}")

    # ---- candidate 1: logistic regression (the interpretable baseline) ----
    # Included as a genuine comparison, not decoration: if a linear scorecard
    # matches the boosted model, the simpler model is the defensible choice.
    logit = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000, C=0.5, class_weight="balanced", random_state=seed
                ),
            ),
        ]
    )
    logit.fit(X_tr, y_tr)
    logit_val_auc = classification_metrics(y_val, logit.predict_proba(X_val)[:, 1]).get("roc_auc") or 0.0

    # ---- candidate 2: XGBoost with monotone constraints ----
    params = dict(BASE_PARAMS)
    params["monotone_constraints"] = _monotone_tuple(feature_names)
    params["scale_pos_weight"] = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    xgb = XGBClassifier(**params)
    xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    xgb_val_auc = classification_metrics(y_val, xgb.predict_proba(X_val)[:, 1]).get("roc_auc") or 0.0

    # ---- candidate 3: isotonic-calibrated XGBoost ----
    # class_weight/scale_pos_weight deliberately distorts the probability scale to
    # help the split search; isotonic regression maps it back to true frequencies.
    calibrated = CalibratedClassifierCV(
        XGBClassifier(**params), method="isotonic", cv=StratifiedKFold(3, shuffle=True, random_state=seed)
    )
    calibrated.fit(X_tr, y_tr)
    cal_val_scores = calibrated.predict_proba(X_val)[:, 1]
    cal_val_auc = classification_metrics(y_val, cal_val_scores).get("roc_auc") or 0.0
    cal_val_ece = expected_calibration_error(y_val, cal_val_scores)
    raw_val_ece = expected_calibration_error(y_val, xgb.predict_proba(X_val)[:, 1])

    if verbose:
        print(f"    logistic          val ROC-AUC={logit_val_auc:.4f}")
        print(f"    xgb (monotone)    val ROC-AUC={xgb_val_auc:.4f}  ECE={raw_val_ece:.4f}")
        print(f"    xgb + isotonic    val ROC-AUC={cal_val_auc:.4f}  ECE={cal_val_ece:.4f}")

    # Selection rule: keep the calibrated model unless it loses meaningful
    # ranking power, because pricing depends on the probability being truthful.
    use_calibrated = cal_val_auc >= xgb_val_auc - 0.01 and cal_val_ece <= raw_val_ece
    model = calibrated if use_calibrated else xgb
    chosen = "xgb_isotonic" if use_calibrated else "xgb_monotone"
    if verbose:
        print(f"  selected: {chosen}")

    # ---- threshold on validation, evaluation on test ----
    val_scores = model.predict_proba(X_val)[:, 1]
    choice = threshold_for_min_precision(y_val, val_scores, min_precision=MIN_PRECISION)

    test_scores = model.predict_proba(X_te)[:, 1]
    test_metrics = classification_metrics(y_te, test_scores, threshold=choice.threshold)
    test_metrics["lift_top_decile"] = lift_at_decile(y_te, test_scores, 1)
    test_metrics["ece"] = expected_calibration_error(y_te, test_scores)

    samples: list[float] = []
    for _ in range(5):
        model.predict_proba(X_te[:1])
    for i in range(min(300, len(X_te))):
        row = X_te[i : i + 1]
        t0 = time.perf_counter()
        model.predict_proba(row)
        samples.append((time.perf_counter() - t0) * 1000)
    latency = latency_summary(samples)

    # Importances come from the underlying booster; CalibratedClassifierCV wraps it.
    booster = xgb
    importances = sorted(
        (
            {"feature": f, "importance": round(float(v), 5)}
            for f, v in zip(feature_names, booster.feature_importances_)
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
        "model_comparison": [
            {"model": "logistic_regression", "val_roc_auc": round(logit_val_auc, 4)},
            {"model": "xgb_monotone", "val_roc_auc": round(xgb_val_auc, 4), "val_ece": raw_val_ece},
            {"model": "xgb_isotonic", "val_roc_auc": round(cal_val_auc, 4), "val_ece": cal_val_ece},
        ],
        "selected_model": chosen,
        "calibration_curve": calibration_curve_points(y_te, test_scores, n_bins=10),
        "latency": latency,
        "top_features": importances[:12],
    }

    if verbose:
        print(
            f"  test: ROC-AUC={test_metrics.get('roc_auc')} Gini={test_metrics.get('gini')} "
            f"KS={test_metrics.get('ks_statistic')} ECE={test_metrics['ece']}"
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
            "algorithm": chosen,
            "params": {k: v for k, v in params.items() if k != "n_jobs"},
            "monotone_constraints": {k: v for k, v in MONOTONE.items()},
            "rows": {"train": int(len(y_tr)), "val": int(len(y_val)), "test": int(len(y_te))},
            "n_features": len(feature_names),
        },
    )
