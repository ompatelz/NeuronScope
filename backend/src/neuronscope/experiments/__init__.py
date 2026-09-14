"""Public orchestration interface for complete NeuronScope experiments."""

from neuronscope.experiments.boundary import compute_decision_boundary
from neuronscope.experiments.schemas import (
    DecisionBoundaryConfig,
    DecisionBoundaryResult,
    ExperimentRequest,
    ExperimentResponse,
    PlaybackConfig,
    PlaybackResult,
    PlaybackSnapshot,
)
from neuronscope.experiments.service import run_experiment

__all__ = [
    "DecisionBoundaryConfig",
    "DecisionBoundaryResult",
    "ExperimentRequest",
    "ExperimentResponse",
    "PlaybackConfig",
    "PlaybackResult",
    "PlaybackSnapshot",
    "compute_decision_boundary",
    "run_experiment",
]
