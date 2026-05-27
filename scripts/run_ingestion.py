"""
Ingestion pipeline entry point.

Runs full historical ingestion and processing for the configured universe.

Usage:
    python scripts/run_ingestion.py
    python scripts/run_ingestion.py --mode full
    python scripts/run_ingestion.py --mode incremental
    python scripts/run_ingestion.py --universe equity
    python scripts/run_ingestion.py --universe index
    python scripts/run_ingestion.py --universe all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from src.data_pipeline.ingestion import IngestionPipeline
from src.data_pipeline.processing import ProcessingPipeline, ProcessingConfig
from src.data_pipeline.universe import (
    get_full_universe,
    get_nse_equity_universe,
    get_nse_index_universe,
)
from src.utils.logger import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run data ingestion and processing pipeline."
    )
    parser.add_argument(
        "--mode",
        choices=["full", "incremental"],
        default=settings.DATA_UPDATE_MODE,
        help="Ingestion mode.",
    )
    parser.add_argument(
        "--universe",
        choices=["equity", "index", "all"],
        default="all",
        help="Asset universe to ingest.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger.info(
        "Starting pipeline. mode=%s universe=%s",
        args.mode,
        args.universe,
    )

    settings.DATA_UPDATE_MODE = args.mode

    if args.universe == "equity":
        assets = get_nse_equity_universe()
    elif args.universe == "index":
        assets = get_nse_index_universe()
    else:
        assets = get_full_universe()

    logger.info("Universe loaded. total_assets=%s", len(assets))

    # ── Ingestion ─────────────────────────────────────────────
    ingestion_pipeline = IngestionPipeline()
    ingestion_report = ingestion_pipeline.run(assets)

    logger.info(
        "Ingestion complete. succeeded=%s failed=%s",
        ingestion_report.succeeded,
        ingestion_report.failed,
    )

    if ingestion_report.failed == ingestion_report.total:
        logger.error("All assets failed ingestion. Aborting.")
        sys.exit(1)

    # ── Processing ────────────────────────────────────────────
    successful_assets = [
        r.symbol for r in ingestion_report.results if r.success
    ]

    processing_configs = [
        ProcessingConfig(
            symbol=asset.symbol,
            asset_class=asset.asset_class,
            timeframe=asset.timeframe,
        )
        for asset in assets
        if asset.symbol in successful_assets
    ]

    processing_pipeline = ProcessingPipeline()
    processing_report = processing_pipeline.run(processing_configs)

    logger.info(
        "Processing complete. succeeded=%s failed=%s",
        processing_report.succeeded,
        processing_report.failed,
    )

    # ── Summary ───────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PIPELINE SUMMARY")
    logger.info("Ingestion  — succeeded=%s failed=%s", ingestion_report.succeeded, ingestion_report.failed)
    logger.info("Processing — succeeded=%s failed=%s", processing_report.succeeded, processing_report.failed)
    logger.info("=" * 60)

    if processing_report.failed > 0:
        failed = [r.symbol for r in processing_report.results if not r.success]
        logger.warning("Failed processing symbols: %s", failed)

    sys.exit(0 if processing_report.failed == 0 else 1)


if __name__ == "__main__":
    main()