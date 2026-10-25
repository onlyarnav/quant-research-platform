"""
Feature engineering package.
"""

from __future__ import annotations

from src.feature_engineering.feature_pipeline import (
    FeaturePipeline,
    FeaturePipelineReport,
)
from src.feature_engineering.feature_schema import (
    FEATURE_COLUMNS,
    IDENTIFIER_COLUMNS,
    TARGET_COLUMNS,
    get_feature_columns,
    validate_feature_frame,
)
from src.feature_engineering.mean_reversion_features import (
    MeanReversionFeatureGenerator,
)
from src.feature_engineering.momentum_features import MomentumFeatureGenerator
from src.feature_engineering.target_generator import TargetGenerator
from src.feature_engineering.trend_features import TrendFeatureGenerator
from src.feature_engineering.volatility_features import VolatilityFeatureGenerator
from src.feature_engineering.volume_features import VolumeFeatureGenerator

__all__ = [
    "FEATURE_COLUMNS",
    "IDENTIFIER_COLUMNS",
    "TARGET_COLUMNS",
    "FeaturePipeline",
    "FeaturePipelineReport",
    "MeanReversionFeatureGenerator",
    "MomentumFeatureGenerator",
    "TargetGenerator",
    "TrendFeatureGenerator",
    "VolatilityFeatureGenerator",
    "VolumeFeatureGenerator",
    "get_feature_columns",
    "validate_feature_frame",
]
