"""Public training interface for small deterministic NeuronScope experiments."""

from neuronscope.training.engine import (
    TrainingRun,
    select_snapshot_epochs,
    train_model,
    train_model_with_snapshots,
)
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
    "TrainingRun",
    "TrainingResult",
    "select_snapshot_epochs",
    "train_model",
    "train_model_with_snapshots",
]
