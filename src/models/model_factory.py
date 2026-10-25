"""
Model factory and registry for instantiating and extending ML model wrappers.

Provides a unified creation interface and dynamic registration mechanism allowing
the platform to instantiate linear, tree, and future deep-learning architectures
consistently by model type string.
"""

from __future__ import annotations

from typing import Any

from src.models.base_model import BaseModel
from src.models.baseline_models import LassoModel, OLSModel
from src.models.lightgbm_model import LightGBMModel
from src.models.linear_model import LinearModel
from src.models.tree_models import GradientBoostingModel, RandomForestModel
from src.models.xgboost_model import XGBoostModel

MODEL_REGISTRY: dict[str, type[BaseModel]] = {
    "ols": OLSModel,
    "linear": LinearModel,
    "lasso": LassoModel,
    "random_forest": RandomForestModel,
    "gradient_boosting": GradientBoostingModel,
    "xgboost": XGBoostModel,
    "lightgbm": LightGBMModel,
}


def available_model_types() -> list[str]:
    """
    Return a sorted list of all registered model type identifiers.

    Returns:
        Alphabetically sorted list of registered type names.
    """
    return sorted(MODEL_REGISTRY.keys())


def create_model(model_type: str, **hyperparameters: Any) -> BaseModel:
    """
    Instantiate a model wrapper by registered model type.

    Args:
        model_type: Registered model type identifier (e.g. 'xgboost', 'linear').
        **hyperparameters: Model-specific hyperparameter keyword arguments.

    Returns:
        Instantiated BaseModel subclass.

    Raises:
        ValueError: If model_type is not registered.
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model_type='{model_type}'. Available: {available_model_types()}"
        )
    model_cls = MODEL_REGISTRY[model_type]
    return model_cls(**hyperparameters)


def register_model(model_type: str, model_cls: type[BaseModel]) -> None:
    """
    Register a new BaseModel subclass in the factory registry.

    Extension point for future deep learning or custom models.

    Args:
        model_type: Unique model type identifier string.
        model_cls: BaseModel subclass to associate with the identifier.

    Raises:
        TypeError: If model_cls is not a subclass of BaseModel.
        ValueError: If model_type is already registered.
    """
    if not isinstance(model_cls, type) or not issubclass(model_cls, BaseModel):
        raise TypeError(f"model_cls must be a subclass of BaseModel, got {model_cls}.")
    if model_type in MODEL_REGISTRY:
        raise ValueError(f"model_type '{model_type}' is already registered.")

    MODEL_REGISTRY[model_type] = model_cls


def default_model_name(model_type: str, horizon: int) -> str:
    """
    Generate the standard canonical name for a model and horizon combination.

    Args:
        model_type: Model type identifier.
        horizon: Prediction horizon in days.

    Returns:
        Formatted name string: '{model_type}_h{horizon}d'.
    """
    return f"{model_type}_h{horizon}d"
