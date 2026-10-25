"""Signal generation strategies and pipeline for the quantitative research platform."""

from __future__ import annotations

from src.signals.rank_signal import RankSignalGenerator
from src.signals.signal_pipeline import SignalGenerator, SignalPipeline
from src.signals.threshold_signal import ThresholdSignalGenerator
from src.signals.topn_signal import TopNLongShortSignalGenerator

__all__ = [
    "RankSignalGenerator",
    "SignalGenerator",
    "SignalPipeline",
    "ThresholdSignalGenerator",
    "TopNLongShortSignalGenerator",
]
