"""
Feature engineering pipeline.

Responsibilities:
    - Load processed parquet datasets
    - Generate all feature families
    - Generate training targets
    - Merge outputs
    - Persist feature datasets following naming convention
    - Continue processing even if one symbol fails

Input:
    data/processed/<asset_class>/<asset_class>_<symbol>_<timeframe>.parquet

Output:
    data/features/<asset_class>/<asset_class>_<symbol>_<timeframe>.parquet
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.data_storage.parquet_writer import DataLayer, ParquetWriter
from src.feature_engineering.mean_reversion_features import MeanReversionFeatureGenerator
from src.feature_engineering.momentum_features import MomentumFeatureGenerator
from src.feature_engineering.target_generator import TargetGenerator
from src.feature_engineering.trend_features import TrendFeatureGenerator
from src.feature_engineering.volatility_features import VolatilityFeatureGenerator
from src.feature_engineering.volume_features import VolumeFeatureGenerator

logger = logging.getLogger(__name__)


@dataclass
class FeaturePipelineReport:
    """
    Aggregate report for a feature pipeline run.

    Attributes:
        total: Total symbols attempted.
        succeeded: Successfully processed symbols.
        failed: Failed symbols.
        failed_files: List of failed file names.
    """

    total: int = 0
    succeeded: int = 0
    failed: int = 0
    failed_files: list[str] = field(default_factory=list)


class FeaturePipeline:
    """
    End-to-end feature engineering pipeline.

    Reads processed parquet files, generates all feature families
    and targets, merges them, and writes feature parquet files
    following the platform naming convention.
    """

    def __init__(
        self,
        writer: ParquetWriter | None = None,
    ) -> None:
        self.writer = writer or ParquetWriter()

        self.trend_generator = TrendFeatureGenerator()
        self.momentum_generator = MomentumFeatureGenerator()
        self.volatility_generator = VolatilityFeatureGenerator()
        self.mean_reversion_generator = MeanReversionFeatureGenerator()
        self.volume_generator = VolumeFeatureGenerator()
        self.target_generator = TargetGenerator()

    def run(self) -> FeaturePipelineReport:
        """
        Execute feature engineering pipeline across all processed files.

        Returns:
            FeaturePipelineReport summarizing success and failure counts.
        """
        processed_root = self.writer.build_file_path(
            DataLayer.PROCESSED, "", "", ""
        ).parent.parent

        parquet_files = sorted(processed_root.rglob("*.parquet"))

        if not parquet_files:
            logger.warning(
                "No processed parquet files found in %s",
                processed_root,
            )
            return FeaturePipelineReport()

        logger.info("Found %s processed datasets.", len(parquet_files))

        report = FeaturePipelineReport()

        for file_path in parquet_files:
            report.total += 1
            try:
                self._process_file(file_path)
                report.succeeded += 1
            except Exception as exc:
                report.failed += 1
                report.failed_files.append(file_path.name)
                logger.exception(
                    "Failed processing %s: %s",
                    file_path.name,
                    exc,
                )

        logger.info(
            "Feature pipeline complete. succeeded=%s failed=%s",
            report.succeeded,
            report.failed,
        )

        if report.failed_files:
            logger.warning("Failed files: %s", report.failed_files)

        return report

    def _process_file(self, file_path: Path) -> None:
        """
        Process a single processed parquet file.

        Args:
            file_path: Path to the processed parquet file.
        """
        logger.info("Processing %s", file_path.name)

        df = pd.read_parquet(file_path)

        if df.empty:
            raise ValueError(f"{file_path.name} contains no rows.")

        # parse naming convention: <asset_class>_<symbol>_<timeframe>.parquet
        stem = file_path.stem
        parts = stem.split("_", 2)

        if len(parts) < 3:
            raise ValueError(
                f"Cannot parse naming convention from filename: {file_path.name}"
            )

        asset_class = parts[0]
        symbol = parts[1]
        timeframe = parts[2]

        feature_df = self._generate_features(df)

        self.writer.write(
            feature_df,
            layer=DataLayer.FEATURES,
            asset_class=asset_class,
            symbol=symbol,
            timeframe=timeframe,
            mode="overwrite",
        )

        logger.info(
            "Saved feature dataset for symbol=%s rows=%s",
            symbol,
            len(feature_df),
        )

    def _generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run all feature generators and merge outputs.

        Args:
            df: Processed market DataFrame.

        Returns:
            Merged feature DataFrame with all features and targets.
        """
        trend_df = self.trend_generator.generate(df)
        momentum_df = self.momentum_generator.generate(df)
        volatility_df = self.volatility_generator.generate(df)
        mean_reversion_df = self.mean_reversion_generator.generate(df)
        volume_df = self.volume_generator.generate(df)
        target_df = self.target_generator.generate(df)

        merged_df = (
            trend_df
            .merge(momentum_df, on=["symbol", "date"], how="inner")
            .merge(volatility_df, on=["symbol", "date"], how="inner")
            .merge(mean_reversion_df, on=["symbol", "date"], how="inner")
            .merge(volume_df, on=["symbol", "date"], how="inner")
            .merge(target_df, on=["symbol", "date"], how="inner")
        )

        merged_df = merged_df.sort_values(
            ["symbol", "date"]
        ).reset_index(drop=True)

        return merged_df