"""Unit tests for ML CLI scripts."""

from __future__ import annotations

import pandas as pd

from config.settings import settings
from scripts.run_prediction import main as run_prediction_main
from scripts.run_retraining import main as run_retraining_main
from scripts.run_signals import main as run_signals_main
from scripts.run_training import main as run_training_main
from scripts.run_walk_forward import main as run_walk_forward_main
from src.data_storage.parquet_writer import LAYER_PATH_MAP, DataLayer, ParquetWriter
from src.models.dataset_builder import DatasetBuilder

# ==============================================================================
# run_training.py
# ==============================================================================


def test_run_training_main(tmp_path, feature_frame, monkeypatch) -> None:
    """Assert run_training.main succeeds with valid arguments."""
    monkeypatch.setattr(DatasetBuilder, "load_all_features", lambda self: feature_frame)
    store_dir = tmp_path / "models"
    monkeypatch.setattr(settings, "MODEL_ARTIFACT_PATH", store_dir)

    code = run_training_main(
        [
            "--model-type",
            "linear",
            "--horizon",
            "1",
            "--no-save",
        ]
    )
    assert code == 0


def test_run_training_main_failure(monkeypatch) -> None:
    """Assert run_training.main catches exceptions and returns exit code 1."""
    monkeypatch.setattr(
        DatasetBuilder,
        "load_all_features",
        lambda self: (_ for _ in ()).throw(RuntimeError("Disk failure")),
    )
    code = run_training_main(["--model-type", "linear"])
    assert code == 1


# ==============================================================================
# run_walk_forward.py
# ==============================================================================


def test_run_walk_forward_main(tmp_path, feature_frame, monkeypatch) -> None:
    """Assert run_walk_forward.main executes validation and outputs results."""
    monkeypatch.setattr(DatasetBuilder, "load_all_features", lambda self: feature_frame)
    csv_out = tmp_path / "wf_metrics.csv"

    code = run_walk_forward_main(
        [
            "--model-type",
            "linear",
            "--horizon",
            "1",
            "--n-splits",
            "2",
            "--min-train-dates",
            "20",
            "--output",
            str(csv_out),
        ]
    )
    assert code == 0
    assert csv_out.exists()


# ==============================================================================
# run_prediction.py
# ==============================================================================


def test_run_prediction_main(trained_model_setup) -> None:
    """Assert run_prediction.main generates predictions successfully."""
    code = run_prediction_main(
        [
            "--model-name",
            "test_linear",
            "--scope",
            "test",
        ]
    )
    assert code == 0


# ==============================================================================
# run_signals.py
# ==============================================================================


def test_run_signals_main(tmp_path, monkeypatch) -> None:
    """Assert run_signals.main reads predictions and writes signal files."""
    pred_dir = tmp_path / "predictions"
    sig_dir = tmp_path / "signals"
    pred_dir.mkdir(parents=True, exist_ok=True)
    sig_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(settings, "DATA_PREDICTIONS_PATH", pred_dir)
    monkeypatch.setitem(LAYER_PATH_MAP, DataLayer.PREDICTIONS, pred_dir)
    monkeypatch.setattr(settings, "DATA_SIGNALS_PATH", sig_dir)
    monkeypatch.setitem(LAYER_PATH_MAP, DataLayer.SIGNALS, sig_dir)

    writer = ParquetWriter()
    df = pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "date": [pd.Timestamp("2023-01-01")],
            "predicted_return": [0.05],
            "model_name": ["model"],
            "model_version": ["v1"],
        }
    )
    writer.write(
        df,
        layer=DataLayer.PREDICTIONS,
        asset_class="equity",
        symbol="AAPL",
        timeframe="daily",
    )

    code = run_signals_main(
        [
            "--strategy",
            "threshold",
            "--symbols",
            "AAPL",
        ]
    )
    assert code == 0

    code_topn = run_signals_main(
        [
            "--strategy",
            "top_n",
            "--top-n",
            "5",
            "--symbols",
            "AAPL",
        ]
    )
    assert code_topn == 0


# ==============================================================================
# run_retraining.py
# ==============================================================================


def test_run_retraining_main(trained_model_setup, feature_frame, monkeypatch) -> None:
    """Assert run_retraining.main evaluates retraining and logs decision."""
    service, store, builder, _ = trained_model_setup
    monkeypatch.setattr(DatasetBuilder, "load_all_features", lambda self: feature_frame)
    monkeypatch.setattr(settings, "MODEL_ARTIFACT_PATH", store.root)

    code = run_retraining_main(
        [
            "--model-type",
            "linear",
            "--horizon",
            "1",
            "--model-name",
            "test_linear",
            "--force",
        ]
    )
    assert code == 0
