"""Stable request and response contracts for synchronous experiments."""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from neuronscope.datasets import DatasetConfig, DatasetResult
from neuronscope.models import MLPArchitecture, MLPConfig
from neuronscope.training import TrainingConfig, TrainingResult


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


class ExperimentRequest(BaseModel):
    """Validated configuration for one generated binary-classification run."""

    model_config = ConfigDict(frozen=True)

    dataset: DatasetConfig
    model: MLPConfig
    training: TrainingConfig
    boundary: DecisionBoundaryConfig = Field(default_factory=DecisionBoundaryConfig)

    @model_validator(mode="after")
    def validate_generated_dataset_shape(self) -> Self:
        """Require the model shape used by NeuronScope's generated 2D datasets."""

        if self.model.input_size != 2 or self.model.output_size != 1:
            raise ValueError(
                "Generated binary datasets require a model with input_size=2 and output_size=1."
            )
        return self


class ExperimentResponse(BaseModel):
    """Dataset, inspectable model structure, and observed training result."""

    model_config = ConfigDict(frozen=True)

    dataset: DatasetResult
    architecture: MLPArchitecture
    training: TrainingResult
    boundary: DecisionBoundaryResult
