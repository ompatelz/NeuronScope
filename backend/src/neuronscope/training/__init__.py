"""Public training interface for small deterministic NeuronScope experiments."""

from neuronscope.training.engine import train_model
from neuronscope.training.schemas import (
    EpochMetrics,
    OptimizerName,
    TrainingConfig,
    TrainingResult,
)

__all__ = [
    "EpochMetrics",
    "OptimizerName",
    "TrainingConfig",
    "TrainingResult",
    "train_model",
]
