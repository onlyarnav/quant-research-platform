"""
Tests for PredictionService: scope filtering, model loading, and persistence.
"""

from __future__ import annotations

import pandas as pd
import pytest

from config.settings import settings
from src.data_storage.parquet_writer import LAYER_PATH_MAP, DataLayer, ParquetWriter
from src.models.dataset_builder import DatasetBuilder
from src.models.linear_model import LinearModel
from src.models.model_store import ModelMetadata, ModelStore
from src.models.prediction_service import PredictionService
from src.models.training_pipeline import TrainingPipeline


def test_prediction_service_scope_test(trained_model_setup) -> None:
    """Assert scope='test' predicts only on dates strictly after val_end_date."""
    service, store, builder, _ = trained_model_setup
    _, meta = store.load("test_linear")
    assert meta.val_end_date is not None

    preds = service.run("test_linear", scope="test")
    assert not preds.empty
    val_end = pd.Timestamp(meta.val_end_date)
    assert (preds["date"] > val_end).all()
    assert preds["model_version"].iloc[0] == meta.version
    assert set(preds.columns) == {
        "symbol",
        "date",
        "predicted_return",
        "model_name",
        "model_version",
    }


def test_prediction_service_scope_oos(trained_model_setup) -> None:
    """Assert scope='oos' predicts only on dates strictly after train_end_date."""
    service, store, _, _ = trained_model_setup
    _, meta = store.load("test_linear")
    preds = service.run("test_linear", scope="oos")
    train_end = pd.Timestamp(meta.train_end_date)
    assert (preds["date"] > train_end).all()


def test_prediction_service_scope_all(trained_model_setup) -> None:
    """Assert scope='all' returns every non-NaN feature row."""
    service, _, builder, _ = trained_model_setup
    preds = service.run("test_linear", scope="all")
    raw = builder.load_all_features()
    assert len(preds) == len(raw)


def test_prediction_service_scope_latest(trained_model_setup) -> None:
    """Assert scope='latest' returns exactly one row per symbol and appends on repeated runs."""
    service, _, _, pred_dir = trained_model_setup
    preds = service.run("test_linear", scope="latest")
    assert len(preds) == 4  # 4 symbols in feature_frame

    # Running a second time with latest appends
    service.run("test_linear", scope="latest")
    writer = ParquetWriter()
    df_read = writer.read(
        layer=DataLayer.PREDICTIONS,
        asset_class="equity",
        symbol="INFY",
        timeframe="daily",
    )
    assert (
        len(df_read) == 1
    )  # Deduplicated on identical (symbol, date) by ParquetWriter


def test_prediction_service_invalid_scope_raises(trained_model_setup) -> None:
    """Assert invalid scope raises ValueError with available scopes in message."""
    service, _, _, _ = trained_model_setup
    with pytest.raises(ValueError, match="scope must be one of"):
        service.run("test_linear", scope="invalid_scope")


def test_prediction_service_missing_model_raises(trained_model_setup) -> None:
    """Assert missing model raises FileNotFoundError."""
    service, _, _, _ = trained_model_setup
    with pytest.raises(FileNotFoundError, match="No trained model found"):
        service.run("nonexistent_model")


def test_prediction_service_missing_feature_columns_raises(
    tmp_path, feature_dir_factory, monkeypatch
) -> None:
    """Assert missing feature columns raises ValueError with missing columns."""
    df_missing = pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "date": [pd.Timestamp("2023-01-01")],
            "other_col": [1.0],
        }
    )
    feat_dir = feature_dir_factory(df_missing)
    builder = DatasetBuilder(features_path=feat_dir)
    store = ModelStore(root=tmp_path / "models")

    # Manually store a model expecting feature 'f1'
    model = LinearModel()
    model._model = LinearModel()
    model._is_fitted = True
    meta = ModelMetadata(
        model_name="need_f1",
        model_type="linear",
        version="v1",
        created_at="2023-01-01T00:00:00Z",
        horizon=1,
        target_column="future_return_1d",
        feature_columns=("f1",),
        feature_set_version="v1",
        dataset_version="hash1",
        hyperparameters={"alpha": 1.0},
        val_metrics={},
        test_metrics={},
    )
    store.save(model, meta)

    service = PredictionService(model_store=store, dataset_builder=builder)
    with pytest.raises(ValueError, match="missing required columns"):
        service.run("need_f1", scope="all")


def test_prediction_service_no_split_dates_raises(
    tmp_path, feature_frame, feature_dir_factory
) -> None:
    """Assert scope='test' raises when model metadata has no split dates."""
    feat_dir = feature_dir_factory(feature_frame)
    builder = DatasetBuilder(features_path=feat_dir)
    store = ModelStore(root=tmp_path / "models")

    model = LinearModel()
    model.fit(feature_frame[["f1"]], pd.Series([0.1] * len(feature_frame)))
    meta = ModelMetadata(
        model_name="no_dates",
        model_type="linear",
        version="v1",
        created_at="2023-01-01T00:00:00Z",
        horizon=1,
        target_column="future_return_1d",
        feature_columns=("f1",),
        feature_set_version="v1",
        dataset_version="hash1",
        hyperparameters={"alpha": 1.0},
        val_metrics={},
        test_metrics={},
        val_end_date=None,  # Missing
    )
    store.save(model, meta)

    service = PredictionService(model_store=store, dataset_builder=builder)
    with pytest.raises(ValueError, match="Model metadata has no split dates"):
        service.run("no_dates", scope="test")


def test_prediction_service_timezone_aware(
    tmp_path, feature_frame_tz, feature_dir_factory, monkeypatch
) -> None:
    """Assert PredictionService works with timezone-aware data seamlessly."""
    pred_dir = tmp_path / "data" / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(settings, "DATA_PREDICTIONS_PATH", pred_dir)
    monkeypatch.setitem(LAYER_PATH_MAP, DataLayer.PREDICTIONS, pred_dir)

    feat_dir = feature_dir_factory(feature_frame_tz)
    builder = DatasetBuilder(features_path=feat_dir)
    store = ModelStore(root=tmp_path / "models")
    pipeline = TrainingPipeline(dataset_builder=builder, model_store=store)

    model = LinearModel()
    pipeline.run(model, horizon=1, model_name="tz_model", promote=True)

    service = PredictionService(model_store=store, dataset_builder=builder)
    preds = service.run("tz_model", scope="test")
    assert str(preds["date"].dt.tz) == "Asia/Kolkata"
