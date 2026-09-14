"""Public training-instrumentation interface."""

from neuronscope.instrumentation.collector import (
    TrainingInstrumentationCollector,
    summarize_activations,
)
from neuronscope.instrumentation.schemas import (
    ActivationStatistics,
    EpochInstrumentation,
    GradientStatistics,
    LayerInstrumentation,
)

__all__ = [
    "ActivationStatistics",
    "EpochInstrumentation",
    "GradientStatistics",
    "LayerInstrumentation",
    "TrainingInstrumentationCollector",
    "summarize_activations",
]
