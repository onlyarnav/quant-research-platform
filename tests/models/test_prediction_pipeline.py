import numpy as np
import pandas as pd
import pytest
from src.models.prediction_pipeline import PredictionPipeline


def test_run_raises_on_empty_features_df(fitted_dummy_model):
    pipeline = PredictionPipeline()
    with pytest.raises(ValueError, match="cannot be empty"):
        pipeline.run(
            model=fitted_dummy_model,
            features_df=pd.DataFrame(),
            feature_columns=["f1"],
            model_version="v1.0",
            asset_class="equity",
        )


def test_run_raises_on_missing_required_columns(fitted_dummy_model):
    pipeline = PredictionPipeline()
    features_df = pd.DataFrame({
        "symbol": ["AAPL"],
        "f1": [1.0],
    })  # missing 'date'
    with pytest.raises(ValueError, match="missing required columns"):
        pipeline.run(
            model=fitted_dummy_model,
            features_df=features_df,
            feature_columns=["f1"],
            model_version="v1.0",
            asset_class="equity",
        )


def test_run_returns_correct_schema(fitted_dummy_model):
    pipeline = PredictionPipeline()
    features_df = pd.DataFrame({
        "symbol": ["AAPL", "AAPL", "AAPL", "MSFT", "MSFT", "MSFT"],
        "date": pd.date_range("2024-01-01", periods=6),
        "f1": np.random.randn(6),
    })

    output_df = pipeline.run(
        model=fitted_dummy_model,
        features_df=features_df,
        feature_columns=["f1"],
        model_version="v1.0",
        asset_class="equity",
    )

    expected_cols = {"symbol", "date", "predicted_return", "model_name", "model_version"}
    assert set(output_df.columns) == expected_cols


def test_run_predicted_return_matches_model_output(fitted_dummy_model):
    pipeline = PredictionPipeline()
    features_df = pd.DataFrame({
        "symbol": ["AAPL", "MSFT"],
        "date": pd.date_range("2024-01-01", periods=2),
        "f1": [1.0, 2.0],
    })

    output_df = pipeline.run(
        model=fitted_dummy_model,
        features_df=features_df,
        feature_columns=["f1"],
        model_version="v1.0",
        asset_class="equity",
    )

    expected_predictions = fitted_dummy_model.predict(features_df[["f1"]])
    assert np.allclose(output_df["predicted_return"].values, expected_predictions)


def test_run_model_name_and_version_populated_correctly(fitted_dummy_model):
    pipeline = PredictionPipeline()
    features_df = pd.DataFrame({
        "symbol": ["AAPL", "MSFT"],
        "date": pd.date_range("2024-01-01", periods=2),
        "f1": [1.0, 2.0],
    })

    model_version = "v1.2.3"
    output_df = pipeline.run(
        model=fitted_dummy_model,
        features_df=features_df,
        feature_columns=["f1"],
        model_version=model_version,
        asset_class="equity",
    )

    assert (output_df["model_name"] == fitted_dummy_model.model_name).all()
    assert (output_df["model_version"] == model_version).all()


def test_run_preserves_row_count(fitted_dummy_model):
    pipeline = PredictionPipeline()
    features_df = pd.DataFrame({
        "symbol": ["AAPL", "AAPL", "MSFT", "MSFT"],
        "date": pd.date_range("2024-01-01", periods=4),
        "f1": np.random.randn(4),
    })

    output_df = pipeline.run(
        model=fitted_dummy_model,
        features_df=features_df,
        feature_columns=["f1"],
        model_version="v1.0",
        asset_class="equity",
    )

    assert len(output_df) == len(features_df)


def test_run_does_not_mutate_input_dataframe(fitted_dummy_model):
    pipeline = PredictionPipeline()
    features_df = pd.DataFrame({
        "symbol": ["AAPL", "MSFT"],
        "date": pd.date_range("2024-01-01", periods=2),
        "f1": [1.0, 2.0],
    })
    original_df = features_df.copy()

    pipeline.run(
        model=fitted_dummy_model,
        features_df=features_df,
        feature_columns=["f1"],
        model_version="v1.0",
        asset_class="equity",
    )

    pd.testing.assert_frame_equal(features_df, original_df)


def test_run_writes_one_file_per_symbol(monkeypatch, tmp_path, fitted_dummy_model):
    from src.data_storage import parquet_writer as pw_module

    # Patch settings and LAYER_PATH_MAP
    monkeypatch.setattr(pw_module.settings, "DATA_PREDICTIONS_PATH", tmp_path)
    monkeypatch.setitem(pw_module.LAYER_PATH_MAP, pw_module.DataLayer.PREDICTIONS, tmp_path)

    features_df = pd.DataFrame({
        "symbol": ["AAPL", "AAPL", "MSFT"],
        "date": pd.date_range("2024-01-01", periods=3),
        "f1": [1.0, 2.0, 3.0],
    })

    pipeline = PredictionPipeline(writer=pw_module.ParquetWriter())
    pipeline.run(
        model=fitted_dummy_model,
        features_df=features_df,
        feature_columns=["f1"],
        model_version="v1.0",
        asset_class="equity",
    )

    # Asset class is 'equity', timeframe defaults to 'daily'
    # Expected file naming: {asset_class}/{asset_class}_{symbol}_{timeframe}.parquet
    assert (tmp_path / "equity" / "equity_AAPL_daily.parquet").exists()
    assert (tmp_path / "equity" / "equity_MSFT_daily.parquet").exists()


def test_run_written_file_content_matches_symbol_subset(monkeypatch, tmp_path, fitted_dummy_model):
    from src.data_storage import parquet_writer as pw_module

    monkeypatch.setattr(pw_module.settings, "DATA_PREDICTIONS_PATH", tmp_path)
    monkeypatch.setitem(pw_module.LAYER_PATH_MAP, pw_module.DataLayer.PREDICTIONS, tmp_path)

    features_df = pd.DataFrame({
        "symbol": ["AAPL", "AAPL", "MSFT"],
        "date": pd.date_range("2024-01-01", periods=3),
        "f1": [1.0, 2.0, 3.0],
    })

    pipeline = PredictionPipeline(writer=pw_module.ParquetWriter())
    output_df = pipeline.run(
        model=fitted_dummy_model,
        features_df=features_df,
        feature_columns=["f1"],
        model_version="v1.0",
        asset_class="equity",
    )


    # Read back AAPL file
    aapl_df = pd.read_parquet(tmp_path / "equity" / "equity_AAPL_daily.parquet")

    # AAPL subset from output_df
    expected_aapl = output_df[output_df["symbol"] == "AAPL"].reset_index(drop=True)

    pd.testing.assert_frame_equal(aapl_df, expected_aapl, check_dtype=False)

def test_run_calls_predict_exactly_once(mocker, fitted_dummy_model):
    """Assert predict() is called once on the full DataFrame, not per symbol."""
    pipeline = PredictionPipeline()
    features_df = pd.DataFrame({
        "symbol": ["AAPL", "AAPL", "MSFT"],
        "date": pd.date_range("2024-01-01", periods=3),
        "f1": [1.0, 2.0, 3.0],
    })

    spy_predict = mocker.spy(fitted_dummy_model, "predict")

    pipeline.run(
        model=fitted_dummy_model,
        features_df=features_df,
        feature_columns=["f1"],
        model_version="v1.0",
        asset_class="equity",
    )

    spy_predict.assert_called_once()