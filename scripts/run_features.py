"""
Feature engineering pipeline entry point.

Runs the full feature engineering pipeline over all processed datasets.

Usage:
    python scripts/run_features.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.feature_engineering.feature_pipeline import FeaturePipeline
from src.utils.logger import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


def main() -> None:
    logger.info("Starting feature engineering pipeline.")

    pipeline = FeaturePipeline()
    report = pipeline.run()

    logger.info("=" * 60)
    logger.info("FEATURE PIPELINE SUMMARY")
    logger.info(
        "total=%s succeeded=%s failed=%s",
        report.total,
        report.succeeded,
        report.failed,
    )
    logger.info("=" * 60)

    if report.failed_files:
        logger.warning("Failed files: %s", report.failed_files)

    sys.exit(0 if report.failed == 0 else 1)


if __name__ == "__main__":
    main()