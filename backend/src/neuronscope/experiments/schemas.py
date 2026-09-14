"""Stable request and response contracts for synchronous experiments."""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from neuronscope.datasets import DatasetConfig, DatasetResult
from neuronscope.diagnostics import DiagnosticResult, DiagnosticsConfig
from neuronscope.instrumentation import EpochInstrumentation
from neuronscope.models import MLPArchitecture, MLPConfig
from neuronscope.training import EpochMetrics, TrainingConfig, TrainingResult

MAX_EXPERIMENT_WORK_UNITS = 1_000_000_000


def _parameter_count(config: MLPConfig) -> int:
    """Return the number of trainable weights and biases in an MLP configuration."""

    widths = (config.input_size, *config.hidden_layers, config.output_size)
    return sum(
        input_width * output_width + output_width
        for input_width, output_width in zip(widths[:-1], widths[1:], strict=True)
    )


class DecisionBoundaryConfig(BaseModel):
    """Bounded controls for final model-grid inference."""

    model_config = ConfigDict(frozen=True)

    resolution: int = Field(default=48, ge=24, le=80)


class DecisionBoundaryResult(BaseModel):
    """Coordinates and row-major class-one probabilities for a square grid."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    resolution: int = Field(ge=24, le=80)
    x_coordinates: tuple[float, ...]
    y_coordinates: tuple[float, ...]
    probabilities: tuple[float, ...]


class PlaybackConfig(BaseModel):
    """Bounded controls for recorded training snapshots."""

    model_config = ConfigDict(frozen=True)

    max_snapshots: int = Field(default=12, ge=2, le=24)
    trace_sample_index: int = Field(default=0, ge=0)


class ForwardLayerTrace(BaseModel):
    """Observed values for one learned layer during a single forward pass."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    layer_name: str
    activation_name: str | None
    pre_activations: tuple[float | None, ...]
    activations: tuple[float | None, ...]


class ForwardPassTrace(BaseModel):
    """A bounded, input-specific trace through one recorded model state."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    sample_index: int = Field(ge=0)
    input_values: tuple[float, ...]
    expected_label: int = Field(ge=0, le=1)
    layers: tuple[ForwardLayerTrace, ...]
    output_logit: float | None
    predicted_probability: float
    predicted_label: int = Field(ge=0, le=1)


class PlaybackSnapshot(BaseModel):
    """Public playback data without model parameters or tensors."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    epoch: int = Field(ge=1)
    metrics: EpochMetrics
    instrumentation: EpochInstrumentation | None
    probabilities: tuple[float, ...]
    forward_pass: ForwardPassTrace


class PlaybackResult(BaseModel):
    """Fixed grid metadata and bounded recorded epochs."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    resolution: int = Field(ge=24, le=80)
    x_coordinates: tuple[float, ...]
    y_coordinates: tuple[float, ...]
    snapshots: tuple[PlaybackSnapshot, ...]


class ExperimentRequest(BaseModel):
    """Validated configuration for one generated binary-classification run."""

    model_config = ConfigDict(frozen=True)

    dataset: DatasetConfig
    model: MLPConfig
    training: TrainingConfig
    boundary: DecisionBoundaryConfig = Field(default_factory=DecisionBoundaryConfig)
    diagnostics: DiagnosticsConfig = Field(default_factory=DiagnosticsConfig)
    playback: PlaybackConfig = Field(default_factory=PlaybackConfig)

    @model_validator(mode="after")
    def validate_generated_dataset_shape(self) -> Self:
        """Require the model shape used by NeuronScope's generated 2D datasets."""

        if self.model.input_size != 2 or self.model.output_size != 1:
            raise ValueError(
                "Generated binary datasets require a model with input_size=2 and output_size=1."
            )
        if self.playback.trace_sample_index >= self.dataset.samples:
            raise ValueError("Trace sample index must refer to a generated dataset point.")

        parameters = _parameter_count(self.model)
        training_work = self.dataset.samples * self.training.epochs
        inference_work = self.boundary.resolution**2 * (self.playback.max_snapshots + 1)
        workload = parameters * (training_work + inference_work)
        if workload > MAX_EXPERIMENT_WORK_UNITS:
            raise ValueError(
                "Experiment workload exceeds the public execution budget; reduce samples, "
                "epochs, layer widths, playback snapshots, or boundary resolution."
            )
        return self


class ExperimentResponse(BaseModel):
    """Dataset, inspectable model structure, and observed training result."""

    model_config = ConfigDict(frozen=True)

    dataset: DatasetResult
    architecture: MLPArchitecture
    training: TrainingResult
    boundary: DecisionBoundaryResult
    diagnostics: tuple[DiagnosticResult, ...]
    playback: PlaybackResult
