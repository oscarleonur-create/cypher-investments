"""Machine learning modules for signal tracking, feature engineering, and prediction."""

from advisor.ml.features import FeatureEngine
from advisor.ml.models import MLModelTrainer, ModelType
from advisor.ml.pipeline import MLPipeline
from advisor.ml.preprocessing import FeaturePreprocessor
from advisor.ml.signal_generator import MLSignalGenerator

__all__ = [
    "FeatureEngine",
    "FeaturePreprocessor",
    "MLModelTrainer",
    "MLPipeline",
    "MLSignalGenerator",
    "ModelType",
]


# Lazy imports for optional modules
def __getattr__(name: str):
    if name == "RegimeDetector":
        from advisor.ml.regime import RegimeDetector

        return RegimeDetector
    if name == "HRPAllocator":
        from advisor.ml.hrp import HRPAllocator

        return HRPAllocator
    if name == "MetaLabeler":
        from advisor.ml.meta_label import MetaLabeler

        return MetaLabeler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
