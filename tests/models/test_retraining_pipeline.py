"""Unit tests for RetrainingPipeline, RetrainDecision, and RetrainingResult."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src.models.drift_detector import DriftReport
from src.models.model_store import ModelStore
from src.models.retraining_pipeline import (
    RetrainingPipeline,
    RetrainingResult,
)
from src.models.training_pipeline import TrainingPipeline

# ==============================================================================
# RetrainDecision logic
# ==============================================================================


def test_needs_retraining_no_model(tmp_path) -> None:
    """Assert returns should_retrain=True when model does not exist in store."""
    store = ModelStore(root=tmp_path / "models")
    tp = TrainingPipeline()
    pipeline = RetrainingPipeline(training_pipeline=tp, model_store=store)

    decision = pipeline.needs_retraining("nonexistent_model")
    assert decision.should_retrain
    assert "no_model" in decision.reasons


def test_needs_retraining_age(trained_model_setup) -> None:
    """Assert returns should_retrain=True when existing model exceeds age threshold."""
    service, store, builder, _ = trained_model_setup
    tp = TrainingPipeline(dataset_builder=builder, model_store=store)
    pipeline = RetrainingPipeline(
        training_pipeline=tp, model_store=store, retrain_interval_days=7
    )

    # Check 1 day later -> not needed
    now_fresh = datetime.now(timezone.utc) + timedelta(days=1)
    dec_fresh = pipeline.needs_retraining("test_linear", now=now_fresh)
    assert not dec_fresh.should_retrain

    # Check 10 days later -> needed due to age
    now_old = datetime.now(timezone.utc) + timedelta(days=10)
    dec_old = pipeline.needs_retraining("test_linear", now=now_old)
    assert dec_old.should_retrain
    assert any("model_age_days=" in r for r in dec_old.reasons)


def test_needs_retraining_drift(tmp_path) -> None:
    """Assert feature drift flags should_retrain=True."""
    store = ModelStore(root=tmp_path / "models")
    tp = TrainingPipeline()
    pipeline = RetrainingPipeline(training_pipeline=tp, model_store=store)

    # Model exists
    from src.models.linear_model import LinearModel
    from src.models.model_store import ModelMetadata

    m = LinearModel()
    m.fit(pd.DataFrame({"f1": [1.0, 2.0]}), pd.Series([0.1, 0.2]))
    meta = ModelMetadata(
        model_name="test_linear",
        model_type="linear",
        version="v1",
        created_at=datetime.now(timezone.utc).isoformat(),
        horizon=1,
        target_column="future_return_1d",
        feature_columns=("f1",),
        feature_set_version="v1",
        dataset_version="v1",
        hyperparameters={},
        val_metrics={"ic": 0.05},
        test_metrics={"ic": 0.05},
    )
    store.save(m, meta, promote=True)

    report_nodrift = DriftReport(
        feature_psi={"f1": 0.01},
        drifted_features=[],
        max_psi=0.01,
        drift_detected=False,
        n_reference=100,
        n_current=100,
    )
    dec_nodrift = pipeline.needs_retraining("test_linear", drift_report=report_nodrift)
    assert not dec_nodrift.should_retrain

    report_drift = DriftReport(
        feature_psi={"f1": 0.5},
        drifted_features=["f1"],
        max_psi=0.5,
        drift_detected=True,
        n_reference=100,
        n_current=100,
    )
    dec_drift = pipeline.needs_retraining("test_linear", drift_report=report_drift)
    assert dec_drift.should_retrain
    assert "feature_drift" in dec_drift.reasons


# ==============================================================================
# RetrainingPipeline.run()
# ==============================================================================


def test_retraining_run_not_needed(tmp_path, trained_model_setup) -> None:
    """Assert run returns trained=False and promoted=False when no retraining needed."""
    service, model_store, dataset_builder, _ = trained_model_setup
    tp = TrainingPipeline(dataset_builder=dataset_builder, model_store=model_store)
    pipeline = RetrainingPipeline(
        training_pipeline=tp,
        model_store=model_store,
        dataset_builder=dataset_builder,
        retrain_interval_days=30,
    )

    result = pipeline.run(model_type="linear", horizon=1, model_name="test_linear")
    assert isinstance(result, RetrainingResult)
    assert not result.trained
    assert not result.promoted


def test_retraining_run_forced_and_promoted(tmp_path, trained_model_setup) -> None:
    """Assert force=True triggers retraining and promotes candidate."""
    service, model_store, dataset_builder, _ = trained_model_setup
    tp = TrainingPipeline(dataset_builder=dataset_builder, model_store=model_store)
    pipeline = RetrainingPipeline(
        training_pipeline=tp,
        model_store=model_store,
        dataset_builder=dataset_builder,
        promotion_tolerance=0.05,
    )

    old_champion = model_store.latest_version("test_linear")
    result = pipeline.run(
        model_type="linear",
        horizon=1,
        model_name="test_linear",
        force=True,
    )

    assert result.trained
    assert "forced" in result.decision.reasons
    # Candidate was trained
    assert result.candidate_version is not None
    assert result.champion_version == old_champion


def test_retraining_invalid_promotion_metric_raises(tmp_path) -> None:
    """Assert invalid promotion_metric raises ValueError."""
    store = ModelStore(root=tmp_path / "models")
    tp = TrainingPipeline()
    with pytest.raises(ValueError, match="Unknown promotion_metric 'unknown_metric'"):
        RetrainingPipeline(
            training_pipeline=tp,
            model_store=store,
            promotion_metric="unknown_metric",
        )


def test_retraining_candidate_rejected_when_degraded(
    tmp_path, trained_model_setup, mocker
) -> None:
    """Assert candidate model is not promoted if its performance degrades beyond tolerance."""
    service, model_store, dataset_builder, _ = trained_model_setup
    tp = TrainingPipeline(dataset_builder=dataset_builder, model_store=model_store)
    pipeline = RetrainingPipeline(
        training_pipeline=tp,
        model_store=model_store,
        dataset_builder=dataset_builder,
        promotion_metric="ic",
        promotion_tolerance=0.01,
    )

    # Force candidate to produce very poor IC
    orig_run = tp.run

    def mock_run(*args, **kwargs):
        res = orig_run(*args, **kwargs)
        res["ic"] = -0.50  # Strongly negative IC compared to champion
        return res

    mocker.patch.object(tp, "run", side_effect=mock_run)

    old_champion = model_store.latest_version("test_linear")
    result = pipeline.run(
        model_type="linear",
        horizon=1,
        model_name="test_linear",
        force=True,
    )

    assert result.trained
    assert not result.promoted
    # Champion remains unchanged
    assert model_store.latest_version("test_linear") == old_champion
