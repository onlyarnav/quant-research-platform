"""CLI script for generating and storing trading signals."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from src.signals.rank_signal import RankSignalGenerator
from src.signals.signal_pipeline import SignalGenerator, SignalPipeline
from src.signals.threshold_signal import ThresholdSignalGenerator
from src.signals.topn_signal import TopNLongShortSignalGenerator
from src.utils.logger import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments for signal generation."""
    parser = argparse.ArgumentParser(
        description="Generate BUY/HOLD/SELL trading signals."
    )
    parser.add_argument(
        "--strategy",
        choices=["threshold", "rank", "topn", "top_n", "top-n"],
        default="threshold",
        help="Signal generation strategy.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=settings.SIGNAL_THRESHOLD,
        help="Return threshold for threshold strategy.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=settings.SIGNAL_TOP_N,
        help="Top-N long allocation count for rank/topn strategies.",
    )
    parser.add_argument(
        "--bottom-n",
        type=int,
        default=settings.SIGNAL_BOTTOM_N,
        help="Bottom-N short allocation count for rank/topn strategies.",
    )
    parser.add_argument(
        "--asset-class",
        type=str,
        default="equity",
        help="Asset class (default: 'equity').",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="daily",
        help="Data timeframe (default: 'daily').",
    )
    parser.add_argument(
        "--symbols",
        nargs="*",
        default=None,
        help="Optional specific symbols list. If omitted, discovers from predictions.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Execute signal generation workflow."""
    args = parse_args(argv)
    try:
        generator: SignalGenerator
        if args.strategy == "threshold":
            generator = ThresholdSignalGenerator(threshold=args.threshold)
        elif args.strategy == "rank":
            generator = RankSignalGenerator(top_n=args.top_n, bottom_n=args.bottom_n)
        elif args.strategy in ("topn", "top_n", "top-n"):
            bottom_n = args.bottom_n if args.bottom_n > 0 else args.top_n
            generator = TopNLongShortSignalGenerator(
                top_n=args.top_n, bottom_n=bottom_n
            )
        else:
            raise ValueError(f"Unknown strategy '{args.strategy}'.")

        pipeline = SignalPipeline()
        symbols = args.symbols
        if not symbols:
            symbols = pipeline.discover_symbols(
                asset_class=args.asset_class,
                timeframe=args.timeframe,
            )

        if not symbols:
            raise ValueError(
                f"No symbols found for asset_class='{args.asset_class}' "
                f"and timeframe='{args.timeframe}'."
            )

        signals_df = pipeline.run(
            generator=generator,
            symbols=symbols,
            asset_class=args.asset_class,
            timeframe=args.timeframe,
        )

        n_buy = (signals_df["signal"] == 1).sum()
        n_hold = (signals_df["signal"] == 0).sum()
        n_sell = (signals_df["signal"] == -1).sum()

        print("=" * 60)
        print(f"SIGNAL GENERATION SUMMARY: Strategy={args.strategy}")
        print(f"Total Symbols:  {signals_df['symbol'].nunique()}")
        print(f"Total Rows:     {len(signals_df)}")
        print(f"BUY Signals:    {n_buy} ({n_buy / len(signals_df):.1%})")
        print(f"HOLD Signals:   {n_hold} ({n_hold / len(signals_df):.1%})")
        print(f"SELL Signals:   {n_sell} ({n_sell / len(signals_df):.1%})")
        print("=" * 60)
        return 0
    except Exception:
        logger.exception("Signal generation pipeline failed.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
