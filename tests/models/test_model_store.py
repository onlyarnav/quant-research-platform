"""
Tests for ModelStore and ModelMetadata.
"""

from __future__ import annotations

import datetime

import numpy as np
import pytest

from src.models.linear_model import LinearModel
from src.models.model_store import ModelMetadata, ModelStore


def _sample_metadata(
    model_name: str = "LinearModel",
    version: str = "20230101T000000Z",
    val_ic: float | None = 0.05,
) -> ModelMetadata:
    """Helper to construct ModelMetadata."""
    return ModelMetadata(
        model_name=model_name,
        model_type="linear",
        version=version,
        created_at="2023-01-01T00:00:00Z",
        horizon=1,
        target_column="future_return_1d",
        feature_columns=("f1", "f2"),
        feature_set_version="v1",
        dataset_version="hash12345678",
        hyperparameters={"alpha": 1.0},
        val_metrics={"ic": val_ic, "rmse": 0.02},
        test_metrics={"ic": 0.04, "rmse": 0.03},
        train_end_date="2022-12-01",
        val_end_date="2022-12-15",
        test_end_date="2022-12-31",
        mlflow_run_id="run-123",
    )


# =====================================================================
# ModelStore Tests
# =====================================================================


def test_new_version_format_and_lexicographical_order() -> None:
    """Assert new_version produces UTC timestamp string sorting in chronological order."""
    t1 = datetime.datetime(2023, 1, 1, 10, 0, 0, tzinfo=datetime.timezone.utc)
    t2 = datetime.datetime(2023, 1, 1, 10, 0, 5, tzinfo=datetime.timezone.utc)

    v1 = ModelStore.new_version(t1)
    v2 = ModelStore.new_version(t2)

    assert v1 == "20230101T100000Z"
    assert v2 == "20230101T100005Z"
    assert v1 < v2


def test_save_creates_files_and_promotes_by_default(tmp_path, regression_data) -> None:
    """Assert save writes model.joblib, metadata.json, and LATEST pointer."""
    X_train, y_train, _, _, _, _ = regression_data
    model = LinearModel(alpha=1.0)
    model.fit(X_train, y_train)

    store = ModelStore(root=tmp_path / "models")
    meta = _sample_metadata(model_name="test_model", version="20230101T120000Z")

    version_dir = store.save(model, meta, promote=True)

    assert (version_dir / "model.joblib").exists()
    assert (version_dir / "metadata.json").exists()
    latest_file = tmp_path / "models" / "test_model" / "LATEST"
    assert latest_file.exists()
    assert latest_file.read_text(encoding="utf-8").strip() == "20230101T120000Z"


def test_save_with_promote_false_does_not_create_latest(
    tmp_path, regression_data
) -> None:
    """Assert promote=False leaves LATEST untouched or uncreated."""
    X_train, y_train, _, _, _, _ = regression_data
    model = LinearModel()
    model.fit(X_train, y_train)

    store = ModelStore(root=tmp_path / "models")
    meta = _sample_metadata(model_name="test_unpromoted", version="20230101T120000Z")

    store.save(model, meta, promote=False)
    assert not (tmp_path / "models" / "test_unpromoted" / "LATEST").exists()


def test_load_latest_restores_identical_model_and_metadata(
    tmp_path, regression_data
) -> None:
    """Assert load() loads LATEST model with matching predictions and metadata."""
    X_train, y_train, _, _, X_test, _ = regression_data
    model = LinearModel(alpha=0.5)
    model.fit(X_train, y_train)
    orig_preds = model.predict(X_test)

    store = ModelStore(root=tmp_path / "models")
    meta = _sample_metadata(model_name="my_linear_model", version="20230101T100000Z")
    store.save(model, meta, promote=True)

    loaded_model, loaded_meta = store.load("my_linear_model")
    loaded_preds = loaded_model.predict(X_test)

    np.testing.assert_allclose(orig_preds, loaded_preds)
    assert loaded_model.model_name == "my_linear_model"
    assert loaded_meta.version == meta.version
    assert loaded_meta.feature_columns == meta.feature_columns


def test_promote_switches_latest_version(tmp_path, regression_data) -> None:
    """Assert promote() updates the LATEST pointer to the requested version."""
    X_train, y_train, _, _, _, _ = regression_data
    model = LinearModel()
    model.fit(X_train, y_train)

    store = ModelStore(root=tmp_path / "models")
    m1 = _sample_metadata(model_name="versioned_model", version="20230101T100000Z")
    m2 = _sample_metadata(model_name="versioned_model", version="20230102T100000Z")

    store.save(model, m1, promote=True)
    assert store.latest_version("versioned_model") == "20230101T100000Z"

    store.save(model, m2, promote=False)
    assert store.latest_version("versioned_model") == "20230101T100000Z"

    store.promote("versioned_model", "20230102T100000Z")
    assert store.latest_version("versioned_model") == "20230102T100000Z"


def test_load_unknown_model_raises_filenotfound(tmp_path) -> None:
    """Assert loading nonexistent model raises FileNotFoundError with documented message."""
    store = ModelStore(root=tmp_path / "models")
    with pytest.raises(
        FileNotFoundError, match="No trained model found for model_name='ghost'"
    ):
        store.load("ghost")


def test_load_unknown_version_raises_filenotfound(tmp_path, regression_data) -> None:
    """Assert loading nonexistent version raises FileNotFoundError with documented message."""
    X_train, y_train, _, _, _, _ = regression_data
    model = LinearModel()
    model.fit(X_train, y_train)

    store = ModelStore(root=tmp_path / "models")
    meta = _sample_metadata(model_name="known_model", version="20230101T100000Z")
    store.save(model, meta)

    with pytest.raises(FileNotFoundError, match="version='99999999T000000Z'"):
        store.load("known_model", version="99999999T000000Z")


def test_invalid_model_name_raises_valueerror(tmp_path) -> None:
    """Assert invalid model_name (path traversal attempt) raises ValueError."""
    store = ModelStore(root=tmp_path / "models")
    with pytest.raises(ValueError, match="Invalid model_name"):
        store.exists("../escape")


def test_save_unfitted_model_raises(tmp_path) -> None:
    """Assert saving an unfitted model raises RuntimeError containing Cannot save unfitted model."""
    model = LinearModel()
    store = ModelStore(root=tmp_path / "models")
    meta = _sample_metadata(model_name="unfitted")
    with pytest.raises(RuntimeError, match="Cannot save unfitted model"):
        store.save(model, meta)


def test_save_existing_version_raises_fileexistserror(
    tmp_path, regression_data
) -> None:
    """Assert saving duplicate version raises FileExistsError."""
    X_train, y_train, _, _, _, _ = regression_data
    model = LinearModel()
    model.fit(X_train, y_train)

    store = ModelStore(root=tmp_path / "models")
    meta = _sample_metadata(model_name="dup_model", version="20230101T100000Z")
    store.save(model, meta)

    with pytest.raises(FileExistsError, match="already exists"):
        store.save(model, meta)


def test_metadata_nan_metrics_serialization() -> None:
    """Assert NaN and inf metrics serialize as None and roundtrip accurately."""
    meta = _sample_metadata(val_ic=float("nan"))
    d = meta.to_dict()
    assert d["val_metrics"]["ic"] is None

    restored = ModelMetadata.from_dict(d)
    assert restored.val_metrics["ic"] is None
    assert restored.feature_columns == meta.feature_columns


def test_list_versions_ascending(tmp_path, regression_data) -> None:
    """Assert list_versions returns ascending sorted version strings."""
    X_train, y_train, _, _, _, _ = regression_data
    model = LinearModel()
    model.fit(X_train, y_train)

    store = ModelStore(root=tmp_path / "models")
    store.save(
        model,
        _sample_metadata(model_name="sorted_m", version="20230103T000000Z"),
        promote=False,
    )
    store.save(
        model,
        _sample_metadata(model_name="sorted_m", version="20230101T000000Z"),
        promote=False,
    )
    store.save(
        model,
        _sample_metadata(model_name="sorted_m", version="20230102T000000Z"),
        promote=False,
    )

    versions = store.list_versions("sorted_m")
    assert versions == ["20230101T000000Z", "20230102T000000Z", "20230103T000000Z"]
